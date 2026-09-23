import io
import threading
import time
import wave
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vadsa.audio import SAMPLE_RATE
from vadsa.config import Settings
from vadsa.engine.base import Engine, Frame, Transcript
from vadsa.engine.fake import FakeEngine
from vadsa.main import create_app

KEY = "test-key"
MODEL = "KlangAI/pianissimo-sv"
AUTH = {"Authorization": f"Bearer {KEY}"}


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


class GatedEngine(FakeEngine):
    """A fake engine that holds the GPU thread in every call until `gate` opens."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.calls: list[str] = []

    def step(self, frames: list[Frame]) -> list[str]:
        self.calls.append("step")
        self.gate.wait()
        return super().step(frames)

    def transcribe(self, audio: np.ndarray) -> Transcript:
        self.calls.append("window")
        self.gate.wait()
        return super().transcribe(audio)


@contextmanager
def serve(engine: Engine | None = None, **overrides: object) -> Iterator[TestClient]:
    """A running app on the fake engine, with test keys and any setting overridden."""
    loaded = engine or FakeEngine()
    settings = Settings(
        **{"environment": "development", "engine": "fake", "api_keys": KEY} | overrides
    )
    app = create_app(settings, load_engine=lambda: loaded)
    with TestClient(app) as client:
        try:
            wait_until(lambda: app.state.scheduler.ready)
            yield client
        finally:
            # let a held GPU thread finish so it can see the stop
            if isinstance(loaded, GatedEngine):
                loaded.gate.set()


@pytest.fixture
def client() -> Iterator[TestClient]:
    with serve() as client:
        yield client
