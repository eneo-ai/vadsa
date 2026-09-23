"""Audio in: ffmpeg decoding, PCM16 conversion and windows for long recordings."""

import asyncio

import numpy as np

SAMPLE_RATE = 16_000
_QUIET_SAMPLES = SAMPLE_RATE // 5  # 200 ms
_SEARCH_SAMPLES = 10 * SAMPLE_RATE


class InvalidAudio(Exception):
    """ffmpeg could not decode any audio from the file."""


class AudioTooLong(Exception):
    """The decoded audio is longer than allowed."""


async def decode(path: str, max_seconds: float) -> np.ndarray:
    """Decode any file ffmpeg reads to 16 kHz mono float32.

    ffmpeg stops one second past the limit, so an over-long file is refused without
    decoding all of it. Cancelled or failing, decode kills and reaps ffmpeg before it
    returns."""
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        path,
        "-map",
        "0:a:0",
        "-t",
        str(max_seconds + 1),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        output, _ = await process.communicate()
    finally:
        if process.returncode is None:
            process.kill()
            # reading to the end closes the pipe, which the wait for the exit needs
            await process.communicate()
    if process.returncode != 0 or not output:
        raise InvalidAudio
    audio = np.frombuffer(output, dtype="<f4")
    if len(audio) > max_seconds * SAMPLE_RATE:
        raise AudioTooLong
    return audio


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Little-endian PCM16 (an even number of bytes) to float32 in [-1, 1)."""
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768


def windows(audio: np.ndarray, window_samples: int) -> list[tuple[int, int]]:
    """(start, end) sample bounds covering the audio in windows of at most window_samples.

    Each cut lands in the middle of the quietest 200 ms within the 10 s before the
    window would end, so a cut rarely splits a word."""
    bounds = []
    start = 0
    while len(audio) - start > window_samples:
        end = start + window_samples
        blocks = min(_SEARCH_SAMPLES, window_samples) // _QUIET_SAMPLES
        search_start = end - blocks * _QUIET_SAMPLES
        energy = np.square(audio[search_start:end]).reshape(blocks, _QUIET_SAMPLES).sum(axis=1)
        cut = search_start + int(np.argmin(energy)) * _QUIET_SAMPLES + _QUIET_SAMPLES // 2
        bounds.append((start, cut))
        start = cut
    bounds.append((start, len(audio)))
    return bounds
