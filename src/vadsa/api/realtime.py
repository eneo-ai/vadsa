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

from vadsa.audio import SAMPLE_RATE, pcm16_to_float32
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
# how long the closing error and close may take to leave; the session's resources are
# already released by then
CLOSE_SECONDS = 5


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
    refusal: SessionEnd | None = None
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
        await _run(websocket, scheduler, stream, events, settings)
    except SessionEnd as end:
        refusal = end
    except WebSocketDisconnect:
        return
    except Exception:
        logger.exception("realtime session failed")
        refusal = SessionEnd(1011, "internal_error", "Internal server error.")
    finally:
        # before the closing message, which a client that stopped reading never takes
        if stream is not None:
            scheduler.discard(stream)
    await _close(websocket, refusal)


async def _run(
    websocket: WebSocket,
    scheduler: Scheduler,
    stream: Stream,
    events: asyncio.Queue[StreamEvent],
    settings: Settings,
) -> None:
    """Relay client events and the stream's text until `transcription.done` is sent, or the
    client, a limit or a failure ends the session first; the session's deadlines cover
    sending the text as well as decoding it.

    Plain asyncio.wait rather than a TaskGroup or gather: those can replace or swallow the
    CancelledError of a session that is being cancelled."""
    receiver = asyncio.create_task(_receive(websocket, scheduler, stream, settings))
    sender = asyncio.create_task(_send_text(websocket, events, stream))
    try:
        # the receiver only ever ends with an error, the sender once the text is out
        await asyncio.wait((receiver, sender), return_when=asyncio.FIRST_COMPLETED)
    finally:
        receiver.cancel()
        sender.cancel()
        # the other side stops before the session closes
        await asyncio.wait((receiver, sender))
    # once the text is out, the session ends as done, even if a deadline or the client's
    # disconnect ended the receiver in the same turn
    if not sender.cancelled() and sender.exception() is None:
        return
    for task in (receiver, sender):
        if not task.cancelled() and (error := task.exception()) is not None:
            raise error


async def _receive(
    websocket: WebSocket, scheduler: Scheduler, stream: Stream, settings: Settings
) -> None:
    """Client events until the session ends. The session's audio is limited; its time only
    by a backstop until the final commit, and after it by the time the final text may take.
    The socket is still read after the final commit, so a disconnect ends such a session."""
    loop = asyncio.get_running_loop()
    audio_left = int(settings.max_session_seconds * SAMPLE_RATE)
    session_ends = loop.time() + settings.max_session_wall_seconds
    idle_ends = loop.time() + settings.idle_timeout_seconds
    audio_ended = False
    while True:
        message = None
        with suppress(TimeoutError):
            async with asyncio.timeout_at(min(session_ends, idle_ends)):
                message = await websocket.receive()
        if loop.time() >= session_ends:
            if audio_ended:
                raise SessionEnd(1013, "finalize_timeout", "The final text was not sent in time.")
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
                samples = _audio(event)
                audio_left -= len(samples)
                if audio_left < 0:
                    raise SessionEnd(
                        1000, "session_too_long", "The session reached its limit of audio."
                    )
                try:
                    scheduler.append(stream, samples)
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
                    # later events change nothing, and waiting for the text is not idling;
                    # the text gets its own time, however long the session has been open
                    audio_ended, idle_ends = True, math.inf
                    session_ends = loop.time() + settings.finalize_seconds
                # vLLM clients commit once before streaming; audio is decoded as it arrives
            case "session.update":
                model = event.get("model")
                if model != settings.served_model_name:
                    raise SessionEnd(
                        1008, "model_not_found", f"The model `{model}` does not exist."
                    )
            case other:
                raise SessionEnd(1008, "unknown_event", f"Unknown event type: {other}")


async def _send_text(
    websocket: WebSocket, events: asyncio.Queue[StreamEvent], stream: Stream
) -> None:
    """Send each piece of committed text as a delta, then the whole text as done."""
    text = ""
    while (event := await events.get()) is not None:
        if isinstance(event, BaseException):
            raise SessionEnd(1011, "internal_error", "Transcription failed.") from event
        # the transcript starts without the word separator the model puts first
        delta = event.text if text else event.text.lstrip()
        if delta:
            text += delta
            await websocket.send_json(
                {
                    "type": "transcription.delta",
                    "delta": delta,
                    "audio_start": event.audio_start,
                    "audio_end": event.audio_end,
                }
            )
    await websocket.send_json(
        {
            "type": "transcription.done",
            "text": text,
            "usage": None,
            "audio_seconds": stream.received_samples / SAMPLE_RATE,
        }
    )


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


async def _close(websocket: WebSocket, refusal: SessionEnd | None) -> None:
    """Close after `transcription.done`, or send a refused session's error and close; a
    client that is gone or stopped reading gets CLOSE_SECONDS for both."""
    with suppress(WebSocketDisconnect, RuntimeError, TimeoutError):
        async with asyncio.timeout(CLOSE_SECONDS):
            if refusal is None:
                await websocket.close(1000)
                return
            error = {"type": "error", "error": refusal.message, "code": refusal.code}
            await websocket.send_json(error)
            await websocket.close(refusal.close_code)
