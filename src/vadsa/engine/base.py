"""What the scheduler needs from a speech model."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class Word:
    word: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    words: list[Word]
    segments: list[Segment]


@dataclass(frozen=True, slots=True)
class Frame:
    """One streaming step for one stream, shaped like NeMo's Frame.

    `samples` is always one full frame of 16 kHz mono float32; the last frame of a
    stream is zero-padded after `length` valid samples."""

    stream_id: int
    samples: np.ndarray
    length: int
    is_first: bool
    is_last: bool


class Engine(Protocol):
    """A loaded model. The scheduler calls it from its GPU thread only."""

    # samples per streaming frame, as the model decodes them
    frame_samples: int
    # silence every new stream starts with, for a model that needs some before speech
    lead_in_samples: int

    def step(self, frames: list[Frame]) -> list[str]:
        """Decode one frame for each of several streams; returns the text each stream
        committed in this step, in frame order."""
        ...

    def discard(self, stream_id: int) -> None:
        """Drop the state of a stream that ends before its last frame."""
        ...

    def transcribe(self, audio: np.ndarray) -> Transcript:
        """Decode one window of 16 kHz mono float32; times are seconds from its start."""
        ...
