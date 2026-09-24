"""A deterministic engine without torch, for tests and local development."""

import numpy as np

from vadsa.audio import SAMPLE_RATE
from vadsa.engine.base import Frame, Segment, Transcript, Word


def _is_speech(samples: np.ndarray) -> bool:
    return samples.size > 0 and float(np.max(np.abs(samples))) > 0.01


class FakeEngine:
    """Hears one word per second of non-silent audio (streaming: per non-silent frame).

    Word n is "ordn". Streaming keeps a frame count per open stream, like the state a
    real pipeline keeps, so tests can see it created, finished and discarded."""

    lead_in_samples = 0
    commit_delay_samples = 0

    def __init__(self, frame_samples: int = SAMPLE_RATE) -> None:
        self.frame_samples = frame_samples
        self.streams: dict[int, int] = {}

    def step(self, frames: list[Frame]) -> list[str]:
        texts = []
        for frame in frames:
            if frame.is_first:
                self.streams[frame.stream_id] = 0
            index = self.streams[frame.stream_id]
            self.streams[frame.stream_id] = index + 1
            texts.append(f" ord{index}" if _is_speech(frame.samples[: frame.length]) else "")
            if frame.is_last:
                del self.streams[frame.stream_id]
        return texts

    def discard(self, stream_id: int) -> None:
        self.streams.pop(stream_id, None)

    def transcribe(self, audio: np.ndarray) -> Transcript:
        duration = len(audio) / SAMPLE_RATE
        words = [
            Word(f"ord{second}", float(second), min(second + 1.0, duration))
            for second in range(-(-len(audio) // SAMPLE_RATE))
            if _is_speech(audio[second * SAMPLE_RATE : (second + 1) * SAMPLE_RATE])
        ]
        # consecutive words form a segment; a silent second ends it
        runs: list[list[Word]] = []
        for word in words:
            if runs and runs[-1][-1].end == word.start:
                runs[-1].append(word)
            else:
                runs.append([word])
        segments = [
            Segment(run[0].start, run[-1].end, " ".join(word.word for word in run)) for run in runs
        ]
        return Transcript(" ".join(word.word for word in words), words, segments)
