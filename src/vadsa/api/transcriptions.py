"""POST /v1/audio/transcriptions: OpenAI multipart in, an OpenAI transcription out."""

import os
import tempfile
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from python_multipart import FormParser
from python_multipart.exceptions import ParseError
from python_multipart.multipart import Field, File, parse_options_header

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
    except ModelLoading:
        raise ApiError(
            503, "The model is still loading.", code="model_loading", headers=_RETRY_AFTER
        ) from None
    except AtCapacity:
        raise ApiError(
            503, "Too many transcriptions in progress.", code="server_busy", headers=_RETRY_AFTER
        ) from None


async def _transcribe(request: Request, settings: Settings, scheduler: Scheduler) -> Response:
    fields, upload = await _receive_form(request, settings.max_upload_bytes)
    try:
        response_format, granularities = _read_options(fields, settings)
        if upload is None:
            raise ApiError(400, "Attach the audio as the `file` field.", param="file")
        samples = await _decode(upload, settings.max_audio_seconds)
    finally:
        if upload is not None:
            upload.close()

    parts = []
    window_samples = int(settings.window_seconds * audio.SAMPLE_RATE)
    for start, end in audio.windows(samples, window_samples):
        if await request.is_disconnected():
            # nginx's "client closed request"; nobody is left to read it
            return Response(status_code=499)
        parts.append((start / audio.SAMPLE_RATE, await scheduler.transcribe(samples[start:end])))
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


async def _receive_form(
    request: Request, max_bytes: int
) -> tuple[dict[str, list[str]], File | None]:
    """Stream the multipart body; the `file` part goes straight to a temp file."""
    content_type, params = parse_options_header(request.headers.get("content-type"))
    if content_type != b"multipart/form-data" or b"boundary" not in params:
        raise ApiError(400, "Send the request as multipart/form-data.")
    too_large = ApiError(
        413, f"The upload is larger than {max_bytes} bytes.", code="file_too_large", param="file"
    )
    if int(request.headers.get("content-length", 0)) > max_bytes:
        raise too_large

    fields: dict[str, list[str]] = {}
    files: list[File] = []

    def on_field(field: Field) -> None:
        name = field.field_name.decode("utf-8", "replace")
        fields.setdefault(name, []).append((field.value or b"").decode("utf-8", "replace"))

    parser = FormParser(
        "multipart/form-data",
        on_field,
        files.append,
        boundary=params[b"boundary"],
        config={"UPLOAD_DIR": tempfile.gettempdir(), "MAX_MEMORY_FILE_SIZE": 0},
    )
    received = 0
    try:
        async for chunk in request.stream():
            received += len(chunk)
            if received > max_bytes:
                raise too_large
            parser.write(chunk)
        parser.finalize()
    except ParseError:
        _close(files)
        raise ApiError(400, "The multipart body is malformed.") from None
    except BaseException:
        _close(files)
        raise

    upload = None
    for file in files:
        if file.field_name == b"file" and upload is None:
            upload = file
        else:
            file.close()
    return fields, upload


def _close(files: list[File]) -> None:
    for file in files:
        file.close()


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


async def _decode(upload: File, max_seconds: float) -> np.ndarray:
    try:
        # an empty upload never leaves memory, so it has no file name
        if upload.actual_file_name is None:
            raise audio.InvalidAudio
        return await audio.decode(os.fsdecode(upload.actual_file_name), max_seconds)
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
