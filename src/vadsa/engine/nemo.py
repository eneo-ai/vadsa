"""KlangAI/pianissimo-sv through NeMo 3.0.0: a buffered streaming pipeline for realtime
and a second model instance for batch transcription.

NeMo and torch are imported when the engine is built, so the rest of vadsa (and its
tests) runs without them."""

from pathlib import Path

import numpy as np

from vadsa.config import Settings
from vadsa.engine.base import Frame, Segment, Transcript, Word

CONFIG = Path(__file__).parent.parent / "conf" / "buffered_tdt.yaml"


class NemoEngine:
    def __init__(self, settings: Settings) -> None:
        import torch
        from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
        from nemo.collections.asr.inference.streaming.framing.request import Frame as NemoFrame
        from nemo.collections.asr.inference.streaming.framing.request_options import (
            ASRRequestOptions,
        )
        from nemo.collections.asr.models import ASRModel
        from omegaconf import OmegaConf

        self._torch = torch
        self._frame_class = NemoFrame
        self._options = ASRRequestOptions
        device = "cuda" if torch.cuda.is_available() else "cpu"

        cfg = OmegaConf.load(CONFIG)
        cfg.asr.model_name = settings.model
        cfg.asr.device = device
        cfg.streaming.chunk_size = settings.stream_chunk_seconds
        cfg.streaming.left_padding_size = settings.stream_left_padding_seconds
        cfg.streaming.right_padding_size = settings.stream_right_padding_seconds
        self._pipeline = PipelineBuilder.build_pipeline(cfg)
        self._pipeline.open_session()
        # A stateful pipeline rounds its sizes up to whole model frames; the frame it
        # decodes is its own chunk_size, never the requested one.
        self.frame_samples = int(self._pipeline.chunk_size * self._pipeline.sample_rate)

        load = (
            ASRModel.restore_from if settings.model.endswith(".nemo") else ASRModel.from_pretrained
        )
        self._model = load(settings.model, map_location=torch.device(device)).eval()

    def step(self, frames: list[Frame]) -> list[str]:
        requests = [
            self._frame_class(
                samples=self._torch.from_numpy(frame.samples),
                stream_id=frame.stream_id,
                is_first=frame.is_first,
                is_last=frame.is_last,
                length=frame.length,
                # the pipeline builds a stream's state from its first frame's options
                options=self._options() if frame.is_first else None,
            )
            for frame in frames
        ]
        texts: list[str] = []
        size = self._pipeline.batch_size
        with self._torch.inference_mode():
            for start in range(0, len(requests), size):
                outputs = self._pipeline.transcribe_step(requests[start : start + size])
                texts.extend(output.current_step_transcript for output in outputs)
        return texts

    def discard(self, stream_id: int) -> None:
        # a normal last frame clears both; any other end has to clear both here
        self._pipeline.delete_state(stream_id)
        self._pipeline.bufferer.rm_bufferer(stream_id)

    def transcribe(self, audio: np.ndarray) -> Transcript:
        # torch wants a writable array; the decoded upload is a read-only buffer
        (hypothesis,) = self._model.transcribe(
            audio=[audio.copy()],
            batch_size=1,
            return_hypotheses=True,
            timestamps=True,
            verbose=False,
        )
        timestamps = hypothesis.timestamp
        return Transcript(
            text=hypothesis.text,
            words=[Word(w["word"], w["start"], w["end"]) for w in timestamps.get("word", [])],
            segments=[
                Segment(s["start"], s["end"], s["segment"]) for s in timestamps.get("segment", [])
            ],
        )
