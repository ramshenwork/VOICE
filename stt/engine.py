"""
Speech to text.

Answers exactly one question: "what words are in this audio?" It is handed a
finished utterance by the VAD layer and never decides when the user stopped
talking.

Nothing outside this file imports faster_whisper. The rest of the system sees
only `transcribe(audio) -> Transcription`, so swapping in a different model,
or a cloud API, touches this file alone.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Transcription:
    text: str
    language: str = ""
    duration_s: float = 0.0
    latency_s: float = 0.0
    # avg_logprob from the decoder, when the model reports it. This is NOT a
    # probability and is deliberately left as the model's own number rather
    # than being rescaled into a fake "confidence" percentage.
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class STTError(RuntimeError):
    pass


class STTEngine:
    """faster-whisper implementation of the transcription interface."""

    def __init__(self, config):
        self.cfg = config
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # pragma: no cover
            raise STTError("faster-whisper is not installed.") from exc

        t0 = time.perf_counter()
        try:
            self._model = WhisperModel(
                self.cfg.model,
                device=self.cfg.device,
                compute_type=self.cfg.compute_type,
            )
        except Exception as exc:
            raise STTError(f"Could not load STT model '{self.cfg.model}': {exc}") from exc
        logger.info(
            "stt model '%s' loaded in %.2fs (device=%s compute=%s)",
            self.cfg.model, time.perf_counter() - t0, self.cfg.device, self.cfg.compute_type,
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcription:
        """
        audio: mono float32 in [-1, 1] at `sample_rate` (must be 16 kHz for Whisper).
        Returns a Transcription. Never raises for empty/short audio -- it
        returns an empty transcription instead, because a half-second of room
        noise is a normal event, not an error.
        """
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = audio.shape[0] / float(sample_rate)

        # Cheap rejections first. A 200 ms noise blip must not pay for a
        # model load or a decode.
        if duration * 1000.0 < self.cfg.min_audio_ms:
            logger.debug("utterance too short (%.0f ms), skipping STT", duration * 1000)
            return Transcription(text="", duration_s=duration)

        if sample_rate != 16000:
            raise STTError(f"STT expects 16 kHz audio, got {sample_rate} Hz")

        if self._model is None:
            self.load()

        t0 = time.perf_counter()
        try:
            segments, info = self._model.transcribe(
                audio,
                language=self.cfg.language or None,
                beam_size=self.cfg.beam_size,
                vad_filter=self.cfg.vad_filter,
                condition_on_previous_text=False,
                # Names Whisper has never seen ("Riya") get spelled as whatever
                # it knows ("Rhea", "Re-e"). A hint makes it spell them right.
                initial_prompt=self.cfg.prompt or None,
            )
            parts, logprobs, no_speech = [], [], []
            for seg in segments:
                parts.append(seg.text)
                if getattr(seg, "avg_logprob", None) is not None:
                    logprobs.append(seg.avg_logprob)
                if getattr(seg, "no_speech_prob", None) is not None:
                    no_speech.append(seg.no_speech_prob)
        except Exception as exc:
            raise STTError(f"Transcription failed: {exc}") from exc

        latency = time.perf_counter() - t0
        text = " ".join(p.strip() for p in parts).strip()
        logger.info("stt: %.2fs audio -> %.2fs decode -> %r", duration, latency, text)

        return Transcription(
            text=text,
            language=getattr(info, "language", self.cfg.language) or "",
            duration_s=duration,
            latency_s=latency,
            avg_logprob=(sum(logprobs) / len(logprobs)) if logprobs else None,
            no_speech_prob=(sum(no_speech) / len(no_speech)) if no_speech else None,
            metadata={"model": self.cfg.model, "segments": len(parts)},
        )
