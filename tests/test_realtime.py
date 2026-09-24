import base64
import re
import threading
import time
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from conftest import AUTH, MODEL, GatedEngine, serve, speech, transcribe, wait_until
from vadsa.engine.fake import FakeEngine

PATH = "/v1/realtime?intent=transcription"


def append(ws: WebSocketTestSession, samples: np.ndarray) -> None:
    pcm = (samples * 32767).astype("<i2").tobytes()
    ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})


def start(ws: WebSocketTestSession) -> dict[str, Any]:
    """What a vLLM client sends first: the model, then a non-final commit."""
    created = ws.receive_json()
    ws.send_json({"type": "session.update", "model": MODEL})
    ws.send_json({"type": "input_audio_buffer.commit"})
    return created


def refusal(ws: WebSocketTestSession) -> tuple[str, int]:
    """The error event's code and the close code that follows it."""
    event = ws.receive_json()
    assert event.keys() == {"type", "error", "code"} and event["type"] == "error"
    assert isinstance(event["error"], str)
    with pytest.raises(WebSocketDisconnect) as closed:
        ws.receive_json()
    return event["code"], closed.value.code


def until_done(ws: WebSocketTestSession) -> list[dict[str, Any]]:
    events = [ws.receive_json()]
    while events[-1]["type"] != "transcription.done":
        events.append(ws.receive_json())
    return events


def opens(client: TestClient) -> bool:
    """Whether a new session gets a slot."""
    with client.websocket_connect(PATH, headers=AUTH) as ws:
        return ws.receive_json()["type"] == "session.created"


def test_a_session_streams_deltas_then_done_in_vllm_shapes(client: TestClient) -> None:
    with client.websocket_connect(PATH, headers=AUTH) as ws:
        created = start(ws)
        for _ in range(25):  # 2.5 s in 100 ms appends, like a live client
            append(ws, speech(0.1))
        ws.send_json({"type": "input_audio_buffer.commit", "final": True})
        events = until_done(ws)
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()

    assert created.keys() == {"type", "id", "created"} and created["type"] == "session.created"
    assert re.fullmatch(r"sess-[0-9a-f]{32}", created["id"])
    assert abs(created["created"] - time.time()) < 60
    assert events == [
        {"type": "transcription.delta", "delta": "ord0"},
        {"type": "transcription.delta", "delta": " ord1"},
        {"type": "transcription.delta", "delta": " ord2"},
        {"type": "transcription.done", "text": "ord0 ord1 ord2", "usage": None},
    ]
    assert closed.value.code == 1000


def test_a_final_commit_without_audio_is_done_with_no_text(client: TestClient) -> None:
    with client.websocket_connect(PATH, headers=AUTH) as ws:
        start(ws)
        ws.send_json({"type": "input_audio_buffer.commit", "final": True})
        assert until_done(ws) == [{"type": "transcription.done", "text": "", "usage": None}]


def test_a_wrong_key_is_refused_before_the_session(client: TestClient) -> None:
    with client.websocket_connect(PATH, headers={"Authorization": "Bearer wrong"}) as ws:
        assert refusal(ws) == ("invalid_api_key", 1008)


def test_another_model_is_refused(client: TestClient) -> None:
    with client.websocket_connect(PATH, headers=AUTH) as ws:
        ws.receive_json()
        ws.send_json({"type": "session.update", "model": "whisper-1"})
        assert refusal(ws) == ("model_not_found", 1008)


def test_sessions_beyond_the_limit_are_refused() -> None:
    with (
        serve(max_sessions=1) as client,
        client.websocket_connect(PATH, headers=AUTH) as first,
    ):
        start(first)
        with client.websocket_connect(PATH, headers=AUTH) as second:
            assert refusal(second) == ("capacity_exceeded", 1013)


def test_a_session_without_appends_ends_idle() -> None:
    with (
        serve(idle_timeout_seconds=0.2) as client,
        client.websocket_connect(PATH, headers=AUTH) as ws,
    ):
        start(ws)
        assert refusal(ws) == ("idle_timeout", 1000)


def test_a_session_open_longer_than_its_audio_limit_still_finishes() -> None:
    # the limit counts audio sent, so time spent without sending any is not held against it
    with (
        serve(max_session_seconds=1.0) as client,
        client.websocket_connect(PATH, headers=AUTH) as ws,
    ):
        start(ws)
        append(ws, speech(0.4))
        time.sleep(1.2)
        append(ws, speech(0.4))
        ws.send_json({"type": "input_audio_buffer.commit", "final": True})
        assert until_done(ws)[-1] == {"type": "transcription.done", "text": "ord0", "usage": None}


def test_audio_past_the_session_limit_is_refused() -> None:
    with (
        serve(max_session_seconds=1.0) as client,
        client.websocket_connect(PATH, headers=AUTH) as ws,
    ):
        start(ws)
        append(ws, speech(0.6))
        append(ws, speech(0.6))
        ws.send_json({"type": "input_audio_buffer.commit", "final": True})
        assert refusal(ws) == ("session_too_long", 1000)


def test_the_final_text_gets_its_own_time_after_the_final_commit() -> None:
    engine = GatedEngine()
    with (
        serve(engine, max_session_seconds=0.5) as client,
        client.websocket_connect(PATH, headers=AUTH) as ws,
    ):
        start(ws)
        append(ws, speech(0.3))
        ws.send_json({"type": "input_audio_buffer.commit", "final": True})
        # the last frame holds the GPU well past the session's audio length in wall time
        wait_until(lambda: engine.calls)
        time.sleep(0.8)
        engine.gate.set()
        assert until_done(ws)[-1] == {"type": "transcription.done", "text": "ord0", "usage": None}


def test_a_session_that_never_finishes_ends_at_the_wall_clock_backstop() -> None:
    # a trickle of silence keeps the session from idling and stays under its audio limit
    with (
        serve(max_session_seconds=0.2, max_session_wall_seconds=0.6) as client,
        client.websocket_connect(PATH, headers=AUTH) as ws,
    ):
        start(ws)
        for _ in range(10):
            append(ws, np.zeros(160, np.float32))
            time.sleep(0.1)
        assert refusal(ws) == ("session_too_long", 1000)


@pytest.mark.parametrize(
    ("ending", "settings"),
    [("disconnect", {}), ("grace", {"finalize_seconds": 0.5})],
    ids=["disconnect", "grace"],
)
def test_a_session_waiting_for_its_last_text_still_ends(
    ending: str, settings: dict[str, float]
) -> None:
    engine = GatedEngine()
    with serve(engine, max_sessions=1, **settings) as client:
        # a transcription holds the GPU, so the text is still due after the final commit
        batch = threading.Thread(target=transcribe, args=(client,))
        batch.start()
        wait_until(lambda: engine.calls)
        with client.websocket_connect(PATH, headers=AUTH) as ws:
            start(ws)
            append(ws, speech(2.5))
            ws.send_json({"type": "input_audio_buffer.commit", "final": True})
            if ending == "disconnect":
                ws.close()
            # the ended session drops its stream, which frees its slot
            wait_until(lambda: opens(client))
            if ending == "grace":
                assert refusal(ws) == ("finalize_timeout", 1013)
        engine.gate.set()
        batch.join()
    # none of the ended session's audio reached the model
    assert engine.calls == ["window"]


def test_a_session_whose_audio_outruns_the_gpu_is_ended() -> None:
    # the gated GPU never finishes a step, so appended audio only piles up
    with (
        serve(GatedEngine(), max_pending_seconds=2) as client,
        client.websocket_connect(PATH, headers=AUTH) as ws,
    ):
        start(ws)
        for _ in range(4):
            append(ws, speech(1))
        assert refusal(ws) == ("falling_behind", 1013)


def test_an_append_over_one_mib_is_refused(client: TestClient) -> None:
    with client.websocket_connect(PATH, headers=AUTH) as ws:
        start(ws)
        append(ws, np.zeros(512 * 1024 + 1, np.float32))
        assert refusal(ws) == ("frame_too_large", 1009)


@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("not json", "invalid_json"),
        ('{"type": "response.create"}', "unknown_event"),
        ('{"type": "input_audio_buffer.append", "audio": "not base64!"}', "invalid_audio"),
        ('{"type": "input_audio_buffer.append", "audio": "AA=="}', "invalid_audio"),
        ('{"type": "input_audio_buffer.commit", "final": "yes"}', "invalid_event"),
    ],
    ids=["json", "type", "base64", "odd-bytes", "final"],
)
def test_malformed_events_end_the_session(client: TestClient, message: str, code: str) -> None:
    with client.websocket_connect(PATH, headers=AUTH) as ws:
        start(ws)
        ws.send_text(message)
        assert refusal(ws) == (code, 1008)


def test_a_disconnect_clears_the_stream_state() -> None:
    engine = FakeEngine()
    with serve(engine) as client:
        with client.websocket_connect(PATH, headers=AUTH) as ws:
            start(ws)
            append(ws, speech(1.5))
            assert ws.receive_json() == {"type": "transcription.delta", "delta": "ord0"}
            assert engine.streams
        wait_until(lambda: not engine.streams)
