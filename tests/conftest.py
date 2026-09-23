import io
import time
import wave
from collections.abc import Callable

import numpy as np

from vadsa.audio import SAMPLE_RATE


def speech(seconds: float, silent: tuple[float, float] | None = None) -> np.ndarray:
    """A 440 Hz tone the fake engine hears as speech, optionally silent between two times."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    if silent:
        samples[int(silent[0] * SAMPLE_RATE) : int(silent[1] * SAMPLE_RATE)] = 0
    return samples


def wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes((samples * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def wait_until(condition: Callable[[], bool], timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)
