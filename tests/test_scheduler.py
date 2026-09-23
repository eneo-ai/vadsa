import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import suppress

import numpy as np
import pytest

from conftest import wait_until
from vadsa.engine.base import Frame, Transcript
from vadsa.engine.fake import FakeEngine
from vadsa.engine.scheduler import AtCapacity, Scheduler, Stream

FRAME = 1600


class RecordingEngine(FakeEngine):
    """Records every call; each call waits until `gate` opens."""

    def __init__(self) -> None:
        super().__init__(frame_samples=FRAME)
        self.gate = threading.Event()
        self.calls: list[str] = []
        self.frames: list[Frame] = []
        self.discarded: list[int] = []
        self.fail = False

    def step(self, frames: list[Frame]) -> list[str]:
        self.calls.append("step")
        self.frames.extend(frames)
        self.gate.wait()
        if self.fail:
            raise RuntimeError("step failed")
        return super().step(frames)

    def discard(self, stream_id: int) -> None:
        self.discarded.append(stream_id)
        super().discard(stream_id)

    def transcribe(self, audio: np.ndarray) -> Transcript:
        self.calls.append("window")
        self.gate.wait()
        return super().transcribe(audio)


async def until(condition: Callable[[], object]) -> None:
    # the GPU thread changes what the condition reads, so poll off the event loop
    await asyncio.to_thread(wait_until, condition)


@pytest.fixture
def engine() -> RecordingEngine:
    return RecordingEngine()


@pytest.fixture
async def scheduler(engine: RecordingEngine) -> AsyncIterator[Scheduler]:
    scheduler = Scheduler(
        lambda: engine, max_sessions=1, max_pending_samples=100 * FRAME, max_batch_requests=4
    )
    scheduler.start()
    await until(lambda: scheduler.ready)
    yield scheduler
    engine.gate.set()
    scheduler.stop()


def window() -> np.ndarray:
    return np.zeros(FRAME, np.float32)


async def test_steps_and_windows_alternate_while_both_wait(
    scheduler: Scheduler, engine: RecordingEngine
) -> None:
    first = asyncio.ensure_future(scheduler.transcribe(window()))
    await until(lambda: engine.calls)
    # while the first window holds the GPU: four frames of one stream and two windows wait
    stream = scheduler.open_stream(lambda event: None)
    scheduler.append(stream, np.ones(4 * FRAME + 1, np.float32))
    queued = [asyncio.ensure_future(scheduler.transcribe(window())) for _ in range(2)]
    await asyncio.sleep(0)  # both tasks run up to their wait, so their windows are queued
    engine.gate.set()
    await asyncio.gather(first, *queued)
    await until(lambda: len(engine.calls) == 7)
    # batch progresses between steps; with only steps left they run back to back
    assert engine.calls == ["window", "step", "window", "step", "window", "step", "step"]


@pytest.mark.parametrize(
    ("samples", "frames"),
    [
        (0, []),
        (FRAME + FRAME // 2, [(FRAME, True, False), (FRAME // 2, False, True)]),
        (2 * FRAME, [(FRAME, True, False), (FRAME, False, True)]),
    ],
    ids=["no-audio", "partial-frame", "exact-multiple"],
)
async def test_final_commit_sends_what_is_left_as_the_last_frame(
    scheduler: Scheduler,
    engine: RecordingEngine,
    samples: int,
    frames: list[tuple[int, bool, bool]],
) -> None:
    engine.gate.set()
    events: list[object] = []
    stream = scheduler.open_stream(events.append)
    scheduler.append(stream, np.full(samples, 0.5, np.float32))
    # commit only once the GPU has taken every frame it may take before the final commit
    await until(lambda: len(engine.frames) == max(len(frames) - 1, 0))
    await asyncio.sleep(0.05)
    scheduler.finish(stream)
    await until(lambda: None in events)
    assert [(f.length, f.is_first, f.is_last) for f in engine.frames] == frames
    assert all(len(f.samples) == FRAME and not f.samples[f.length :].any() for f in engine.frames)
    assert events[-1] is None and engine.streams == {}


async def test_a_discarded_stream_holds_its_slot_until_its_state_is_cleared(
    scheduler: Scheduler, engine: RecordingEngine
) -> None:
    stream = scheduler.open_stream(lambda event: None)
    scheduler.append(stream, np.ones(FRAME + 1, np.float32))
    await until(lambda: engine.frames)
    scheduler.discard(stream)
    with pytest.raises(AtCapacity):
        scheduler.open_stream(lambda event: None)

    engine.gate.set()
    reopened: list[Stream] = []

    def reopen() -> bool:
        with suppress(AtCapacity):
            reopened.append(scheduler.open_stream(lambda event: None))
        return bool(reopened)

    await until(reopen)
    assert engine.discarded == [stream.id] and engine.streams == {}
    assert reopened[0].id > stream.id


async def test_a_failed_step_ends_its_streams_and_clears_their_state(
    scheduler: Scheduler, engine: RecordingEngine
) -> None:
    engine.fail = True
    engine.gate.set()
    events: list[object] = []
    stream = scheduler.open_stream(events.append)
    scheduler.append(stream, np.ones(FRAME + 1, np.float32))
    await until(lambda: engine.discarded)
    assert isinstance(events[0], RuntimeError)
    assert engine.discarded == [stream.id]
