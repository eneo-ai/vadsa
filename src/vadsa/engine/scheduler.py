"""The GPU owner: one thread makes every engine call.

NeMo pipelines are not thread-safe and both workloads share one device, so a single
thread runs realtime steps and batch windows. When both kinds wait it alternates one
step (every stream with a frame ready, batched) and one window, so batch always
progresses and realtime waits at most one window. With one kind waiting it runs that.

Streams get their results through loop.call_soon_threadsafe (their `deliver`); windows
through concurrent futures that asyncio.wrap_future hands back to the event loop the
same way."""

import asyncio
import itertools
import logging
import os
import signal
import threading
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field

import numpy as np

from vadsa.audio import SAMPLE_RATE
from vadsa.engine.base import Engine, Frame, Transcript

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StreamDelta:
    text: str
    audio_start: float
    audio_end: float


# What a stream's session receives: committed text after each step, then None once the
# last frame is decoded, or the exception that ended the stream.
StreamEvent = StreamDelta | BaseException | None


class ModelLoading(Exception):
    """The engine is still loading."""


class AtCapacity(Exception):
    """No room for another realtime session or batch request."""


class FallingBehind(Exception):
    """More audio waits for the GPU than a stream may queue."""


@dataclass(eq=False)
class Stream:
    id: int
    deliver: Callable[[StreamEvent], None]
    # samples not yet sent to the engine
    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    # the engine's lead-in of silence went in ahead of the first audio
    led_in: bool = False
    # a frame reached the engine, so the engine holds state for this id
    started: bool = False
    final: bool = False
    received_samples: int = 0
    steps: int = 0


@dataclass(frozen=True, eq=False)
class _Window:
    audio: np.ndarray
    future: Future[Transcript]


class Scheduler:
    def __init__(
        self,
        load: Callable[[], Engine],
        *,
        max_sessions: int,
        max_pending_samples: int,
        max_batch_requests: int,
    ) -> None:
        self._load = load
        self._max_sessions = max_sessions
        self._max_pending_samples = max_pending_samples
        self._max_batch_requests = max_batch_requests
        self._lock = threading.Condition()
        self._engine: Engine | None = None
        self._stopped = False
        self._streams: dict[int, Stream] = {}
        # ids whose engine state the GPU thread still has to clear
        self._discarded: list[int] = []
        self._windows: deque[_Window] = deque()
        self._batch_requests = 0
        self._stream_turn = True
        # never reused: the pipeline keeps the state of an id it has seen before
        self._ids = itertools.count(1)
        self._thread = threading.Thread(target=self._run, name="vadsa-gpu", daemon=True)

    @property
    def ready(self) -> bool:
        return self._engine is not None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            self._lock.notify()

    # Realtime sessions, called from the event loop.

    def open_stream(self, deliver: Callable[[StreamEvent], None]) -> Stream:
        with self._lock:
            if self._engine is None:
                raise ModelLoading
            # a discarded stream keeps its slot until its engine state is cleared
            if len(self._streams) + len(self._discarded) >= self._max_sessions:
                raise AtCapacity
            stream = Stream(next(self._ids), deliver)
            self._streams[stream.id] = stream
            return stream

    def append(self, stream: Stream, samples: np.ndarray) -> None:
        with self._lock:
            received = len(samples)
            if len(samples) and not stream.led_in and self._engine is not None:
                # only real audio gets the lead-in, so an empty session ends at once
                lead_in = np.zeros(self._engine.lead_in_samples, np.float32)
                samples = np.concatenate((lead_in, samples))
                stream.led_in = True
            stream.audio = np.concatenate((stream.audio, samples))
            if len(stream.audio) > self._max_pending_samples:
                raise FallingBehind
            stream.received_samples += received
            self._lock.notify()

    def finish(self, stream: Stream) -> None:
        """The final commit: send what is left, then deliver None."""
        with self._lock:
            stream.final = True
            if not stream.started and len(stream.audio) == 0:
                # nothing ever reached the engine, so there is nothing to decode
                self._streams.pop(stream.id, None)
                stream.deliver(None)
            self._lock.notify()

    def discard(self, stream: Stream) -> None:
        """End a stream without its last frame; a no-op for one that already ended."""
        with self._lock:
            if self._streams.pop(stream.id, None) is not None and stream.started:
                self._discarded.append(stream.id)
                self._lock.notify()

    # Batch requests, called from the event loop.

    @contextmanager
    def admit(self) -> Iterator[None]:
        """Hold one of the batch request slots, whether in progress or queued."""
        with self._lock:
            if self._engine is None:
                raise ModelLoading
            if self._batch_requests >= self._max_batch_requests:
                raise AtCapacity
            self._batch_requests += 1
        try:
            yield
        finally:
            with self._lock:
                self._batch_requests -= 1

    async def transcribe(self, audio: np.ndarray) -> Transcript:
        """Decode one window. Cancelled while queued, the window leaves the queue at once;
        cancelled on the GPU, it finishes first, so no caller outlives the use of its audio."""
        window = _Window(audio, Future())
        with self._lock:
            self._windows.append(window)
            self._lock.notify()
        try:
            return await asyncio.wrap_future(window.future)
        except asyncio.CancelledError:
            with self._lock:
                if window in self._windows:
                    self._windows.remove(window)
            # a window already running on the GPU cannot be cancelled; wait until it is done
            if not window.future.cancel():
                with suppress(Exception):
                    await asyncio.wrap_future(window.future)
            raise

    # The GPU thread.

    def _run(self) -> None:
        try:
            engine = self._load()
        except Exception:
            # without a model the server is useless; SIGTERM lets uvicorn shut down and
            # the container's restart policy try again
            logger.exception("the model failed to load")
            os.kill(os.getpid(), signal.SIGTERM)
            return
        with self._lock:
            self._engine = engine
        logger.info("model ready, realtime frames of %.2f s", engine.frame_samples / SAMPLE_RATE)
        while (work := self._next(engine)) is not None:
            if isinstance(work, _Window):
                self._decode(engine, work)
            else:
                self._step(engine, work)
            # a finished window may hold a whole decoded recording; drop it before waiting
            del work

    def _next(self, engine: Engine) -> _Window | list[tuple[Stream, Frame]] | None:
        with self._lock:
            while not self._stopped:
                for stream_id in self._discarded:
                    engine.discard(stream_id)
                self._discarded.clear()
                ready = [s for s in self._streams.values() if _has_frame(s, engine.frame_samples)]
                if ready and (self._stream_turn or not self._windows):
                    self._stream_turn = False
                    return [(stream, _take_frame(stream, engine.frame_samples)) for stream in ready]
                if self._windows:
                    self._stream_turn = True
                    return self._windows.popleft()
                self._lock.wait()
            return None

    def _step(self, engine: Engine, taken: list[tuple[Stream, Frame]]) -> None:
        try:
            texts = engine.step([frame for _, frame in taken])
        except Exception as error:
            logger.exception("realtime step failed")
            with self._lock:
                for stream, _ in taken:
                    if self._streams.pop(stream.id, None) is not None:
                        self._discarded.append(stream.id)
            for stream, _ in taken:
                stream.deliver(error)
            return
        for (stream, frame), text in zip(taken, texts, strict=True):
            with self._lock:
                stream.steps += 1
                submitted = stream.received_samples / SAMPLE_RATE
                start = (stream.steps - 2) * engine.frame_samples - engine.lead_in_samples
                end = (stream.steps - 1) * engine.frame_samples - engine.lead_in_samples
                delta = StreamDelta(
                    text,
                    min(submitted, max(0.0, round(start / SAMPLE_RATE, 3))),
                    submitted
                    if frame.is_last
                    else min(submitted, max(0.0, round(end / SAMPLE_RATE, 3))),
                )
                if frame.is_last:
                    self._streams.pop(stream.id, None)
            stream.deliver(delta)
            if frame.is_last:
                stream.deliver(None)

    def _decode(self, engine: Engine, window: _Window) -> None:
        if not window.future.set_running_or_notify_cancel():
            return
        try:
            transcript = engine.transcribe(window.audio)
        except Exception as error:
            window.future.set_exception(error)
        else:
            window.future.set_result(transcript)


def _has_frame(stream: Stream, frame_samples: int) -> bool:
    # Until the final commit a stream keeps at least one sample back, so the final
    # commit always has 1..frame_samples left for the padded last frame.
    return len(stream.audio) > frame_samples or (stream.final and len(stream.audio) > 0)


def _take_frame(stream: Stream, frame_samples: int) -> Frame:
    audio = stream.audio
    length = min(len(audio), frame_samples)
    samples = np.zeros(frame_samples, np.float32)
    samples[:length] = audio[:length]
    frame = Frame(
        stream_id=stream.id,
        samples=samples,
        length=length,
        is_first=not stream.started,
        is_last=stream.final and len(audio) <= frame_samples,
    )
    stream.audio = audio[length:]
    stream.started = True
    return frame
