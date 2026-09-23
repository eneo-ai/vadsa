import asyncio
import gc
import json
import tempfile
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Message
from starlette.websockets import WebSocketDisconnect

from conftest import (
    AUTH,
    KEY,
    MODEL,
    GatedEngine,
    serve,
    speech,
    transcribe,
    wait_until,
    wav_bytes,
)
from vadsa.config import Settings
from vadsa.engine.fake import FakeEngine
from vadsa.main import create_app

BOUNDARY = "vadsa-boundary"
HEADERS = AUTH | {"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"}
WAV = wav_bytes(speech(3))


def part(name: str, value: str | bytes, filename: str | None = None) -> bytes:
    """One part of a multipart body; a file name makes it a file part."""
    disposition = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
    data = value.encode() if isinstance(value, str) else value
    return f"--{BOUNDARY}\r\nContent-Disposition: {disposition}\r\n\r\n".encode() + data + b"\r\n"


def form(*parts: bytes) -> bytes:
    return b"".join(parts) + f"--{BOUNDARY}--\r\n".encode()


async def call(
    app: ASGIApp, body: list[bytes], leave: asyncio.Event | None = None
) -> tuple[int, bytes]:
    """POST `body` to the transcription route straight through ASGI, one chunk per receive(),
    and return the status and the content; `body` keeps the chunks the app never asked for.

    The client then disconnects: right after the last chunk, before the body is complete,
    or, given `leave`, after a complete body once `leave` is set."""
    sent: list[Message] = []

    async def receive() -> Message:
        if body:
            chunk = body.pop(0)
            return {"type": "http.request", "body": chunk, "more_body": leave is None or bool(body)}
        if leave is not None:
            await leave.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/audio/transcriptions",
        "raw_path": b"/v1/audio/transcriptions",
        "root_path": "",
        "query_string": b"",
        "headers": [(key.lower().encode(), value.encode()) for key, value in HEADERS.items()],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    await app(scope, receive, send)
    return sent[0]["status"], b"".join(message.get("body", b"") for message in sent[1:])


@pytest.fixture
def uploads_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[[], list[Path]]]:
    """The upload files still on disk. The cyclic collector is off, so a file is only gone
    once the request itself has closed it."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    gc.disable()
    try:
        yield lambda: list(tmp_path.iterdir())
    finally:
        gc.enable()


def error_of(response: httpx2.Response) -> tuple[int, str | None]:
    error = response.json()["error"]
    assert error.keys() == {"message", "type", "param", "code"}
    return response.status_code, error["code"]


def test_while_the_model_loads_health_and_work_are_refused() -> None:
    loaded = threading.Event()

    def load() -> FakeEngine:
        loaded.wait()
        return FakeEngine()

    settings = Settings(environment="development", engine="fake", api_keys=KEY)
    with TestClient(create_app(settings, load_engine=load)) as client:
        assert client.get("/health").status_code == 503
        response = transcribe(client)
        assert error_of(response) == (503, "model_loading")
        assert response.headers["retry-after"]
        with client.websocket_connect("/v1/realtime", headers=AUTH) as ws:
            assert ws.receive_json()["code"] == "model_loading"
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_json()
            assert closed.value.code == 1013
        loaded.set()
        wait_until(lambda: client.get("/health").status_code == 200)


def test_models_lists_the_served_name_as_a_transcription_model(client: TestClient) -> None:
    body = client.get("/v1/models", headers=AUTH).json()
    [card] = body.pop("data")
    assert body == {"object": "list"}
    assert isinstance(card.pop("created"), int)
    assert card == {
        "id": MODEL,
        "object": "model",
        "owned_by": "vadsa",
        "model_type": "transcription",
    }


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong"}, {"Authorization": f"Basic {KEY}"}],
    ids=["missing", "wrong", "not-bearer"],
)
def test_v1_needs_a_valid_key(client: TestClient, headers: dict[str, str]) -> None:
    assert error_of(client.get("/v1/models", headers=headers)) == (401, "invalid_api_key")
    assert error_of(transcribe(client, headers=headers)) == (401, "invalid_api_key")


def test_without_keys_development_is_open() -> None:
    with serve(api_keys="") as client:
        assert client.get("/v1/models").status_code == 200


def test_json_is_the_default_format(client: TestClient) -> None:
    assert transcribe(client).json() == {"text": "ord0 ord1 ord2"}


def test_text_format_is_plain_text(client: TestClient) -> None:
    response = transcribe(client, response_format="text")
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "ord0 ord1 ord2"


@pytest.mark.parametrize(
    ("granularities", "present"),
    [([], {"segments"}), (["word"], {"words"}), (["word", "segment"], {"words", "segments"})],
    ids=["default", "word", "both"],
)
def test_verbose_json_includes_what_was_requested(
    client: TestClient, granularities: list[str], present: set[str]
) -> None:
    audio = wav_bytes(speech(3, silent=(1.0, 2.0)))
    fields: dict[str, Any] = {"response_format": "verbose_json"}
    if granularities:
        fields["timestamp_granularities[]"] = granularities
    body = transcribe(client, audio, **fields).json()
    assert {
        "task": "transcribe",
        "language": "sv",
        "duration": 3.0,
        "text": "ord0 ord2",
    }.items() <= body.items()
    assert body.keys() - {"task", "language", "duration", "text"} == present
    if "words" in present:
        assert body["words"] == [
            {"word": "ord0", "start": 0.0, "end": 1.0},
            {"word": "ord2", "start": 2.0, "end": 3.0},
        ]
    if "segments" in present:
        assert body["segments"] == [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "ord0"},
            {"id": 1, "start": 2.0, "end": 3.0, "text": "ord2"},
        ]


@pytest.mark.parametrize("language", ["sv", "auto", None])
def test_swedish_is_served(client: TestClient, language: str | None) -> None:
    fields = {"language": language} if language else {}
    assert transcribe(client, **fields).status_code == 200


@pytest.mark.parametrize(
    ("fields", "status", "code"),
    [
        ({"model": "whisper-1"}, 404, "model_not_found"),
        ({"language": "en"}, 400, "unsupported_language"),
        ({"response_format": "srt"}, 400, "unsupported_response_format"),
        ({"timestamp_granularities[]": ["token"]}, 400, None),
    ],
    ids=["model", "language", "format", "granularity"],
)
def test_requests_outside_the_contract_are_refused(
    client: TestClient, fields: dict[str, Any], status: int, code: str | None
) -> None:
    assert error_of(transcribe(client, **fields)) == (status, code)


def test_the_file_is_required(client: TestClient) -> None:
    # a multipart body with the model field only
    files = {"model": (None, MODEL)}
    response = client.post("/v1/audio/transcriptions", files=files, headers=AUTH)
    assert error_of(response) == (400, None)
    assert response.json()["error"]["param"] == "file"


@pytest.mark.parametrize("audio", [b"", b"not audio"], ids=["empty", "text"])
def test_undecodable_uploads_are_invalid_audio(client: TestClient, audio: bytes) -> None:
    assert error_of(transcribe(client, audio)) == (400, "invalid_audio")


def test_audio_over_the_duration_limit_is_refused() -> None:
    with serve(max_audio_seconds=2) as client:
        assert error_of(transcribe(client)) == (400, "audio_too_long")


@pytest.mark.parametrize("chunked", [False, True], ids=["content-length", "chunked"])
def test_uploads_over_the_size_limit_are_refused(chunked: bool) -> None:
    body = form(part("model", MODEL), part("file", WAV, "a.wav"))

    def chunks() -> Iterator[bytes]:
        for start in range(0, len(body), 4096):
            yield body[start : start + 4096]

    with serve(max_upload_bytes=50_000) as client:
        response = client.post(
            "/v1/audio/transcriptions", content=chunks() if chunked else body, headers=HEADERS
        )
        assert error_of(response) == (413, "file_too_large")


@pytest.mark.parametrize(
    ("body", "refused_after", "param"),
    [
        (
            form(part("model", MODEL), part("file", WAV, "a.wav"), part("file", WAV, "b.wav")),
            b'filename="b.wav"\r\n\r\n',
            "file",
        ),
        (
            form(part("model", MODEL), part("audio", WAV, "a.wav")),
            b'filename="a.wav"\r\n\r\n',
            "file",
        ),
        (
            form(*(part(f"field{n}", "value") for n in range(33)), part("file", WAV, "a.wav")),
            b'name="field32"\r\n\r\n',
            None,
        ),
        (
            form(part("prompt", "x" * (64 * 1024 + 1)), part("file", WAV, "a.wav")),
            b"x" * (64 * 1024 + 1),
            None,
        ),
    ],
    ids=["second-file", "file-not-named-file", "33-fields", "field-bytes"],
)
async def test_a_form_beyond_its_bounds_is_refused_as_soon_as_it_crosses_them(
    uploads_left: Callable[[], list[Path]], body: bytes, refused_after: bytes, param: str | None
) -> None:
    at = body.index(refused_after) + len(refused_after)
    tail = body[at:]
    chunks = [body[:at], tail]
    with serve() as client:
        status, content = await call(client.app, chunks)
    assert status == 400
    assert json.loads(content)["error"]["param"] == param
    # the rest of the body was never read, and nothing it started is left
    assert chunks == [tail]
    assert uploads_left() == []


async def test_a_client_that_disconnects_mid_upload_leaves_no_file(
    uploads_left: Callable[[], list[Path]],
) -> None:
    body = form(part("model", MODEL), part("file", WAV, "a.wav"))
    with serve() as client:
        status, _ = await call(client.app, [body[: len(body) // 2]])
    assert status == 499
    assert uploads_left() == []


def test_a_body_that_ends_before_its_closing_boundary_is_refused(
    uploads_left: Callable[[], list[Path]],
) -> None:
    # complete `model` and `file` parts, then the body stops where another part would start
    body = form(part("model", MODEL), part("file", WAV, "a.wav"))
    truncated = body.removesuffix(f"--{BOUNDARY}--\r\n".encode()) + f"--{BOUNDARY}\r\n".encode()
    with serve() as client:
        response = client.post("/v1/audio/transcriptions", content=truncated, headers=HEADERS)
    assert error_of(response) == (400, None)
    assert response.json()["error"]["message"] == "The multipart body is malformed."
    assert uploads_left() == []


def test_long_audio_is_decoded_in_windows_with_times_offset() -> None:
    # 20 s windows; the first cut lands in the silence at 17.0-17.4 s, at 17.1 s
    audio = wav_bytes(speech(25, silent=(17.0, 17.4)))
    with serve(window_seconds=20) as client:
        body = transcribe(
            client, audio, response_format="verbose_json", **{"timestamp_granularities[]": "word"}
        ).json()
    starts = [word["start"] for word in body["words"]]
    assert starts[:17] == [float(second) for second in range(17)]
    assert starts[17:] == [round(17.1 + second, 3) for second in range(8)]
    assert body["words"][-1]["end"] == 25.0


def test_batch_requests_beyond_the_limit_get_503() -> None:
    engine = GatedEngine()
    with serve(engine, max_batch_requests=1) as client:
        first: list[httpx2.Response] = []
        thread = threading.Thread(target=lambda: first.append(transcribe(client)))
        thread.start()
        wait_until(lambda: engine.calls)
        busy = transcribe(client)
        assert error_of(busy) == (503, "server_busy")
        assert busy.headers["retry-after"]
        engine.gate.set()
        thread.join()
        assert first[0].status_code == 200


async def test_a_request_whose_client_leaves_while_its_window_waits_is_never_transcribed() -> None:
    engine = GatedEngine()
    with serve(engine) as client:
        # the first request's window holds the GPU
        first: list[httpx2.Response] = []
        thread = threading.Thread(target=lambda: first.append(transcribe(client)))
        thread.start()
        await asyncio.to_thread(wait_until, lambda: engine.calls)
        leave = asyncio.Event()
        body = [form(part("model", MODEL), part("file", WAV, "a.wav"))]
        second = asyncio.ensure_future(call(client.app, body, leave))
        # the second request's window waits behind it when its client disconnects
        await asyncio.to_thread(wait_until, lambda: client.app.state.scheduler._windows)
        leave.set()
        # it ends at once, without waiting for the GPU
        status, _ = await asyncio.wait_for(second, 5)
        assert status == 499
        engine.gate.set()
        await asyncio.to_thread(thread.join)
    assert first[0].status_code == 200
    # the model only ever saw the first request's window
    assert engine.calls == ["window"]
