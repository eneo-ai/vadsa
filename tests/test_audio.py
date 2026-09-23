import itertools

import numpy as np
import pytest

from conftest import speech, wav_bytes
from vadsa import audio
from vadsa.audio import SAMPLE_RATE


def test_audio_within_one_window_is_not_split() -> None:
    samples = speech(20)
    assert audio.windows(samples, 20 * SAMPLE_RATE) == [(0, len(samples))]


def test_cut_lands_in_the_quietest_200_ms_before_the_boundary() -> None:
    # silence at 17.0-17.4 s lies within the 10 s before the 20 s boundary
    samples = speech(25, silent=(17.0, 17.4))
    bounds = audio.windows(samples, 20 * SAMPLE_RATE)
    assert bounds == [(0, int(17.1 * SAMPLE_RATE)), (int(17.1 * SAMPLE_RATE), len(samples))]


def test_windows_cover_long_audio_without_gaps_or_oversize() -> None:
    samples = np.random.default_rng(0).uniform(-0.5, 0.5, 95 * SAMPLE_RATE).astype(np.float32)
    window = 20 * SAMPLE_RATE
    bounds = audio.windows(samples, window)
    assert bounds[0][0] == 0 and bounds[-1][1] == len(samples)
    assert all(end == next_start for (_, end), (next_start, _) in itertools.pairwise(bounds))
    assert all(0 < end - start <= window for start, end in bounds)


async def test_decode_resamples_to_16_khz_mono(tmp_path) -> None:
    path = tmp_path / "eight-khz.wav"
    path.write_bytes(wav_bytes(np.zeros(8000, np.float32), sample_rate=8000))
    samples = await audio.decode(str(path), max_seconds=10)
    assert samples.dtype == np.float32
    assert abs(len(samples) - SAMPLE_RATE) < 100


async def test_decode_refuses_what_is_not_audio(tmp_path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("not audio")
    with pytest.raises(audio.InvalidAudio):
        await audio.decode(str(path), max_seconds=10)


async def test_decode_refuses_audio_over_the_limit(tmp_path) -> None:
    path = tmp_path / "long.wav"
    path.write_bytes(wav_bytes(speech(3)))
    with pytest.raises(audio.AudioTooLong):
        await audio.decode(str(path), max_seconds=2)
