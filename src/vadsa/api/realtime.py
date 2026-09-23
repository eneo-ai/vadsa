"""GET /v1/realtime: vLLM's realtime transcription dialect on a WebSocket.

Event shapes follow vllm/entrypoints/speech_to_text/realtime/protocol.py. Every refusal
is an `error` event after accept, then a close; docs/realtime.md is the reference."""

import asyncio
import base64
import json
import logging
import math
import time
import uuid
from contextlib import suppress
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from vadsa.audio import pcm16_to_float32
from vadsa.auth import bearer_token, key_accepted
from vadsa.config import Settings
from vadsa.engine.scheduler import (
    AtCapacity,
    FallingBehind,
    ModelLoading,
    Scheduler,
    Stream,
    StreamEvent,
)

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_APPEND_BYTES = 1024 * 1024


class SessionEnd(Exception):
    """Ends a session with an `error` event and then a close with `close_code`."""

    def __init__(self, close_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.close_code = close_code
        self.code = code
        self.message = message


@router.websocket("/v1/realtime")
async def realtime(websocket: WebSocket) -> None:
    await websocket.accept()
    settings: Settings = websocket.app.state.settings
    scheduler: Scheduler = websocket.app.state.scheduler
    loop = asyncio.get_running_loop()
    events: asyncio.Queue[StreamEvent] = asyncio.Queue()

    def deliver(event: StreamEvent) -> None:
        # called from the GPU thread; the loop only closes when the server stops
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(events.put_nowait, event)

    stream: Stream | None = None
    try:
        token = bearer_token(websocket.headers.get("authorization"))
        if not key_accepted(token, settings.api_keys):
            raise SessionEnd(1008, "invalid_api_key", "Invalid API key.")
        try:
            stream = scheduler.open_stream(deliver)
        except ModelLoading:
            raise SessionEnd(1013, "model_loading", "The model is still loading.") from None
        except AtCapacity:
            raise SessionEnd(1013, "capacity_exceeded", "Too many realtime sessions.") from None
        await websocket.send_json(
            {
                "type": "session.created",
                "id": f"sess-{uuid.uuid4().hex}",
                "created": int(time.time()),
            }
        )
        text = await _run(websocket, scheduler, stream, events, settings)
        await websocket.send_json({"type": "transcription.done", "text": text, "usage": None})
        await websocket.close(1000)
    except SessionEnd as end:
        await _refuse(websocket, end)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("realtime session failed")
        await _refuse(websocket, SessionEnd(1011, "internal_error", "Internal server error."))
    finally:
        if stream is not None:
            scheduler.discard(stream)


async def _run(
    websocket: WebSocket,
    scheduler: Scheduler,
    stream: Stream,
    events: asyncio.Queue[StreamEvent],
    settings: Settings,
) -> str:
    """Relay client events and the stream's text until the final commit is decoded, or the
    client, a limit or a failure ends the session first.

    Plain asyncio.wait rather than a TaskGroup or gather: those can replace or swallow the
    CancelledError of a session that is being cancelled."""
    receiver = asyncio.create_task(_receive(websocket, scheduler, stream, settings))
    sender = asyncio.create_task(_send_deltas(websocket, events))
    try:
        # the receiver only ever ends with an error, the sender with the text or an error
        await asyncio.wait((receiver, sender), return_when=asyncio.FIRST_COMPLETED)
    finally:
        receiver.cancel()
        sender.cancel()
        # the other side stops before the session sends its last event
        await asyncio.wait((receiver, sender))
    for task in (receiver, sender):
        if not task.cancelled() and (error := task.exception()) is not None:
            raise error
    return sender.result()


async def _receive(
    websocket: WebSocket, scheduler: Scheduler, stream: Stream, settings: Settings
) -> None:
    """Client events until the session ends. After the final commit the socket is still
    read, so a disconnect or the time limit ends a session whose last text is pending."""
    loop = asyncio.get_running_loop()
    session_ends = loop.time() + settings.max_session_seconds
    idle_ends = loop.time() + settings.idle_timeout_seconds
    audio_ended = False
    while True:
        message = None
        with suppress(TimeoutError):
            async with asyncio.timeout_at(min(session_ends, idle_ends)):
                message = await websocket.receive()
        if loop.time() >= session_ends:
            raise SessionEnd(1000, "session_too_long", "The session reached its time limit.")
        if message is None:
            raise SessionEnd(1000, "idle_timeout", "No audio arrived in time.")
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(message.get("code", 1000))
        if audio_ended:
            continue

        event = _parse(message.get("text"))
        match event.get("type"):
            case "input_audio_buffer.append":
                try:
                    scheduler.append(stream, _audio(event))
                except FallingBehind:
                    raise SessionEnd(
                        1013, "falling_behind", "Too much audio is waiting to be transcribed."
                    ) from None
                idle_ends = loop.time() + settings.idle_timeout_seconds
            case "input_audio_buffer.commit":
                final = event.get("final", False)
                if not isinstance(final, bool):
                    raise SessionEnd(1008, "invalid_event", "`final` must be true or false.")
                if final:
                    scheduler.finish(stream)
                    # later events change nothing, and waiting for the text is not idling
                    audio_ended, idle_ends = True, math.inf
                # vLLM clients commit once before streaming; audio is decoded as it arrives
            case "session.update":
                model = event.get("model")
                if model != settings.served_model_name:
                    raise SessionEnd(
                        1008, "model_not_found", f"The model `{model}` does not exist."
                    )
            case other:
                raise SessionEnd(1008, "unknown_event", f"Unknown event type: {other}")


async def _send_deltas(websocket: WebSocket, events: asyncio.Queue[StreamEvent]) -> str:
    """Send each piece of committed text as a delta; return the whole text."""
    text = ""
    while (event := await events.get()) is not None:
        if isinstance(event, BaseException):
            raise SessionEnd(1011, "internal_error", "Transcription failed.") from event
        # the transcript starts without the word separator the model puts first
        delta = event if text else event.lstrip()
        if delta:
            text += delta
            await websocket.send_json({"type": "transcription.delta", "delta": delta})
    return text


def _parse(text: str | None) -> dict[str, Any]:
    if text is None:
        raise SessionEnd(1008, "invalid_event", "Send events as JSON text messages.")
    try:
        event = json.loads(text)
    except ValueError:
        raise SessionEnd(1008, "invalid_json", "Invalid JSON.") from None
    if not isinstance(event, dict):
        raise SessionEnd(1008, "invalid_event", "An event is a JSON object.")
    return event


def _audio(event: dict[str, Any]) -> np.ndarray:
    encoded = event.get("audio")
    if not isinstance(encoded, str):
        raise SessionEnd(1008, "invalid_event", "`audio` must be a base64 string.")
    try:
        pcm = base64.b64decode(encoded, validate=True)
    except ValueError:
        raise SessionEnd(1008, "invalid_audio", "`audio` is not valid base64.") from None
    if len(pcm) > MAX_APPEND_BYTES:
        raise SessionEnd(
            1009, "frame_too_large", f"An append carries at most {MAX_APPEND_BYTES} bytes of audio."
        )
    if len(pcm) % 2:
        raise SessionEnd(1008, "invalid_audio", "PCM16 audio has an even number of bytes.")
    return pcm16_to_float32(pcm)


async def _refuse(websocket: WebSocket, end: SessionEnd) -> None:
    # the client may already be gone
    with suppress(WebSocketDisconnect, RuntimeError):
        await websocket.send_json({"type": "error", "error": end.message, "code": end.code})
        await websocket.close(end.close_code)
