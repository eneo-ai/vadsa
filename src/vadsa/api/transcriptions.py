"""POST /v1/audio/transcriptions: OpenAI multipart in, an OpenAI transcription out."""

import asyncio
import tempfile
from collections.abc import Awaitable
from typing import IO, Literal

import numpy as np
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from python_multipart import MultipartParser
from python_multipart.exceptions import FormParserError
from python_multipart.multipart import parse_options_header
from starlette.requests import ClientDisconnect

from vadsa import audio
from vadsa.auth import require_api_key
from vadsa.config import Settings
from vadsa.engine.base import Segment, Transcript, Word
from vadsa.engine.scheduler import AtCapacity, ModelLoading, Scheduler
from vadsa.errors import ApiError, ErrorResponse

router = APIRouter()

_RETRY_AFTER = {"Retry-After": "5"}
_FORMATS = ("json", "text", "verbose_json")
_GRANULARITIES = ("word", "segment")
# the OpenAI form is a handful of short fields next to the file
_MAX_FIELDS = 32
_MAX_FIELD_BYTES = 64 * 1024


class Transcription(BaseModel):
    text: str


class TranscriptionWord(BaseModel):
    word: str
    start: float
    end: float


class TranscriptionSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class TranscriptionVerbose(BaseModel):
    task: Literal["transcribe"] = "transcribe"
    language: Literal["sv"] = "sv"
    duration: float
    text: str
    # each present only when requested through timestamp_granularities[]
    segments: list[TranscriptionSegment] | None = None
    words: list[TranscriptionWord] | None = None


_FORM = {
    "type": "object",
    "required": ["file", "model"],
    "properties": {
        "file": {
            "type": "string",
            "format": "binary",
            "description": "Audio in any format ffmpeg reads.",
        },
        "model": {"type": "string", "description": "The served model name."},
        "language": {
            "type": "string",
            "enum": ["sv", "auto"],
            "default": "sv",
            "description": "Only Swedish is served; auto means sv.",
        },
        "response_format": {"type": "string", "enum": list(_FORMATS), "default": "json"},
        "timestamp_granularities[]": {
            "type": "array",
            "items": {"type": "string", "enum": list(_GRANULARITIES)},
            "default": ["segment"],
            "description": "What verbose_json includes; other formats ignore it.",
        },
        "prompt": {"type": "string", "description": "Accepted and ignored."},
        "temperature": {"type": "number", "description": "Accepted and ignored."},
    },
}


@router.post(
    "/v1/audio/transcriptions",
    dependencies=[Depends(require_api_key)],
    response_model=Transcription | TranscriptionVerbose,
    responses={
        200: {"content": {"text/plain": {"schema": {"type": "string"}}}},
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    openapi_extra={
        "requestBody": {"required": True, "content": {"multipart/form-data": {"schema": _FORM}}}
    },
)
async def create_transcription(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    scheduler: Scheduler = request.app.state.scheduler
    # admission comes before the body is read, so a refused upload is never stored
    try:
        with scheduler.admit():
            return await _transcribe(request, settings, scheduler)
    except ClientDisconnect:
        # nginx's "client closed request"; nobody is left to read it
        return Response(status_code=499)
    except ModelLoading:
        raise ApiError(
            503, "The model is still loading.", code="model_loading", headers=_RETRY_AFTER
        ) from None
    except AtCapacity:
        raise ApiError(
            503, "Too many transcriptions in progress.", code="server_busy", headers=_RETRY_AFTER
        ) from None


async def _transcribe(request: Request, settings: Settings, scheduler: Scheduler) -> Response:
    # leaving the block deletes the upload, however the request ends
    with tempfile.NamedTemporaryFile() as upload:
        fields, has_file = await _receive_form(request, settings.max_upload_bytes, upload)
        response_format, granularities = _read_options(fields, settings)
        if not has_file:
            raise ApiError(400, "Attach the audio as the `file` field.", param="file")
        samples = await _decode(upload.name, settings.max_audio_seconds)

    parts = []
    window_samples = int(settings.window_seconds * audio.SAMPLE_RATE)
    for start, end in audio.windows(samples, window_samples):
        # a client that left while its audio was decoded gets no window queued
        if await request.is_disconnected():
            raise ClientDisconnect
        window = scheduler.transcribe(samples[start:end])
        parts.append((start / audio.SAMPLE_RATE, await _unless_disconnected(request, window)))
    transcript = _join(parts)

    if response_format == "text":
        return PlainTextResponse(transcript.text)
    if response_format == "json":
        return JSONResponse(Transcription(text=transcript.text).model_dump())
    verbose = TranscriptionVerbose(
        duration=round(len(samples) / audio.SAMPLE_RATE, 3), text=transcript.text
    )
    if "segment" in granularities:
        verbose.segments = [
            TranscriptionSegment(
                id=index, start=round(s.start, 3), end=round(s.end, 3), text=s.text
            )
            for index, s in enumerate(transcript.segments)
        ]
    if "word" in granularities:
        verbose.words = [
            TranscriptionWord(word=w.word, start=round(w.start, 3), end=round(w.end, 3))
            for w in transcript.words
        ]
    return JSONResponse(verbose.model_dump(exclude_none=True))


async def _unless_disconnected(request: Request, window: Awaitable[Transcript]) -> Transcript:
    """The window's transcript, or ClientDisconnect once the client has gone. The scheduler
    takes a cancelled window off its queue, or lets it finish on the GPU, before this raises,
    so the request keeps its admission slot until its audio is let go."""
    work = asyncio.ensure_future(window)
    gone = asyncio.ensure_future(_disconnected(request))
    try:
        await asyncio.wait((work, gone), return_when=asyncio.FIRST_COMPLETED)
    finally:
        gone.cancel()
        work.cancel()
        await asyncio.wait((work,))
    if work.cancelled():
        raise ClientDisconnect
    return work.result()


async def _disconnected(request: Request) -> None:
    # once the body is read, receive() answers when the client has gone
    while (await request.receive())["type"] != "http.disconnect":
        pass


async def _receive_form(
    request: Request, max_bytes: int, upload: IO[bytes]
) -> tuple[dict[str, list[str]], bool]:
    """Stream the multipart body: the `file` part into `upload`, the other fields into memory.

    One file part is taken and the fields are bounded while they arrive; a body is whole only
    at its closing boundary. Returns the fields and whether the file arrived."""
    content_type, params = parse_options_header(request.headers.get("content-type"))
    if content_type != b"multipart/form-data" or b"boundary" not in params:
        raise ApiError(400, "Send the request as multipart/form-data.")
    too_large = ApiError(
        413, f"The upload is larger than {max_bytes} bytes.", code="file_too_large", param="file"
    )
    if int(request.headers.get("content-length", 0)) > max_bytes:
        raise too_large
    malformed = ApiError(400, "The multipart body is malformed.")
    fields_over_limit = ApiError(
        400,
        f"The form takes at most {_MAX_FIELDS} fields besides the file, "
        f"of {_MAX_FIELD_BYTES} bytes together.",
    )

    fields: dict[str, list[str]] = {}
    field_count = field_bytes = 0
    header_name, header_value = bytearray(), bytearray()
    disposition = b""
    # the field being read, or None while the file is
    field: tuple[str, bytearray] | None = None
    has_file = ended = False

    def on_header_field(data: bytes, start: int, end: int) -> None:
        header_name.extend(data[start:end])

    def on_header_value(data: bytes, start: int, end: int) -> None:
        header_value.extend(data[start:end])

    def on_header_end() -> None:
        nonlocal disposition
        if header_name.lower() == b"content-disposition":
            disposition = bytes(header_value)
        header_name.clear()
        header_value.clear()

    def on_headers_finished() -> None:
        nonlocal disposition, field, field_count, has_file
        _, options = parse_options_header(disposition)
        disposition = b""
        name = options.get(b"name")
        if name is None:
            raise malformed
        if b"filename" in options:
            # any file part but the first `file` is refused as soon as it starts
            if has_file or name != b"file":
                raise ApiError(400, "Attach one file, as the `file` field.", param="file")
            has_file, field = True, None
            return
        field_count += 1
        if field_count > _MAX_FIELDS:
            raise fields_over_limit
        field = (name.decode("utf-8", "replace"), bytearray())

    def on_part_data(data: bytes, start: int, end: int) -> None:
        nonlocal field_bytes
        if field is None:
            upload.write(data[start:end])
            return
        field_bytes += end - start
        if field_bytes > _MAX_FIELD_BYTES:
            raise fields_over_limit
        field[1].extend(data[start:end])

    def on_part_end() -> None:
        if field is not None:
            name, value = field
            fields.setdefault(name, []).append(value.decode("utf-8", "replace"))

    def on_end() -> None:
        nonlocal ended
        ended = True

    received = 0
    try:
        parser = MultipartParser(
            params[b"boundary"],
            {
                "on_header_field": on_header_field,
                "on_header_value": on_header_value,
                "on_header_end": on_header_end,
                "on_headers_finished": on_headers_finished,
                "on_part_data": on_part_data,
                "on_part_end": on_part_end,
                "on_end": on_end,
            },
        )
        async for chunk in request.stream():
            received += len(chunk)
            if received > max_bytes:
                raise too_large
            parser.write(chunk)
    except FormParserError:
        raise malformed from None
    # the parser accepts a body that stops early; only the closing boundary ends it
    if not ended:
        raise malformed
    upload.flush()
    return fields, has_file


def _read_options(fields: dict[str, list[str]], settings: Settings) -> tuple[str, set[str]]:
    def last(name: str) -> str | None:
        values = fields.get(name)
        return values[-1] if values else None

    model = last("model")
    if model != settings.served_model_name:
        raise ApiError(
            404, f"The model `{model}` does not exist.", code="model_not_found", param="model"
        )
    if last("language") not in (None, "", "sv", "auto"):
        raise ApiError(
            400, "Only Swedish (sv) is served.", code="unsupported_language", param="language"
        )
    response_format = last("response_format") or "json"
    if response_format not in _FORMATS:
        raise ApiError(
            400,
            f"response_format must be one of {', '.join(_FORMATS)}.",
            code="unsupported_response_format",
            param="response_format",
        )
    granularities = set(fields.get("timestamp_granularities[]", ["segment"]))
    if not granularities <= set(_GRANULARITIES):
        raise ApiError(
            400,
            "timestamp_granularities[] takes word and segment.",
            param="timestamp_granularities[]",
        )
    return response_format, granularities


async def _decode(path: str, max_seconds: float) -> np.ndarray:
    try:
        return await audio.decode(path, max_seconds)
    except audio.InvalidAudio:
        raise ApiError(
            400, "The file is not audio ffmpeg can decode.", code="invalid_audio", param="file"
        ) from None
    except audio.AudioTooLong:
        raise ApiError(
            400,
            f"The audio is longer than {max_seconds:g} seconds.",
            code="audio_too_long",
            param="file",
        ) from None


def _join(parts: list[tuple[float, Transcript]]) -> Transcript:
    """One transcript from its windows, each window's times moved by its offset."""
    return Transcript(
        text=" ".join(part.text for _, part in parts if part.text),
        words=[
            Word(w.word, w.start + offset, w.end + offset)
            for offset, part in parts
            for w in part.words
        ],
        segments=[
            Segment(s.start + offset, s.end + offset, s.text)
            for offset, part in parts
            for s in part.segments
        ],
    )
