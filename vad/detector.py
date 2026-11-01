"""
Voice activity detection.

This module answers exactly one question: "is a human speaking right now, and
has an utterance just begun or just ended?" It does not know what words are
being said and it never calls the transcriber.

Why this is more than `probability > threshold`
-----------------------------------------------
A raw threshold produces two failure modes that make a voice agent unusable:

  1. Chopping. Natural speech contains 200-400 ms gaps between words and
     clauses. A bare threshold treats each one as end-of-turn and fires the
     transcriber on fragments.
  2. Clipping. By the time the model is confident speech started, the first
     phoneme is already gone, so "what's the weather" is transcribed as
     "the weather".

The fix is a state machine with three properties:

  * Hysteresis. Entering speech needs a HIGH probability; leaving it needs a
    sustained LOW probability. Two thresholds, not one, so the detector does
    not oscillate around a single value.
  * Duration gates. min_speech_ms rejects coughs and door slams.
    min_silence_ms is the "have they actually finished?" timer -- it is the
    single knob that trades responsiveness against interrupting the user.
  * Pre-roll. A ring buffer continuously retains the last pre_speech_pad_ms of
    audio. When speech is confirmed, that history is prepended, recovering the
    onset the model needed time to notice.

State transitions
-----------------
    IDLE ---- p >= speech_threshold ----> MAYBE_SPEECH
    MAYBE_SPEECH -- held for min_speech_ms --> SPEAKING   (emit SPEECH_STARTED)
    MAYBE_SPEECH -- p drops --------------> IDLE          (rejected as noise)
    SPEAKING ---- p < silence_threshold ---> MAYBE_END
    MAYBE_END ---- p recovers ------------> SPEAKING      (it was just a pause)
    MAYBE_END -- silent for min_silence_ms -> IDLE        (emit SPEECH_ENDED)
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SILERO_FRAME_16K = 512
SILERO_FRAME_8K = 256


class VADState(str, Enum):
    IDLE = "idle"
    MAYBE_SPEECH = "maybe_speech"
    SPEAKING = "speaking"
    MAYBE_END = "maybe_end"


class VADEvent(str, Enum):
    NONE = "none"
    SPEECH_STARTED = "speech_started"
    SPEECH_CONTINUING = "speech_continuing"
    SPEECH_ENDED = "speech_ended"


@dataclass
class VADResult:
    event: VADEvent
    state: VADState
    speech_probability: float
    is_speech: bool
    # Populated only on SPEECH_ENDED: the complete utterance, padded.
    segment: Optional[np.ndarray] = None
    segment_duration_s: float = 0.0


class SileroVAD:
    """Thin wrapper over the Silero VAD neural model. Frame in, probability out."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.frame_size = SILERO_FRAME_16K if sample_rate == 16000 else SILERO_FRAME_8K
        self._model = None
        self._torch = None

    def load(self) -> None:
        import torch
        self._torch = torch
        torch.set_num_threads(1)  # VAD is tiny; extra threads add latency, not speed
        try:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad()
        except Exception:
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
        logger.info("silero vad loaded (frame=%d @ %d Hz)", self.frame_size, self.sample_rate)

    def reset(self) -> None:
        if self._model is not None and hasattr(self._model, "reset_states"):
            self._model.reset_states()

    def probability(self, frame: np.ndarray) -> float:
        if self._model is None:
            self.load()
        if frame.shape[0] != self.frame_size:
            raise ValueError(
                f"Silero VAD needs exactly {self.frame_size} samples, got {frame.shape[0]}"
            )
        tensor = self._torch.from_numpy(frame.astype(np.float32))
        with self._torch.no_grad():
            return float(self._model(tensor, self.sample_rate).item())


class VADController:
    """
    The state machine. Feed it audio chunks of any size; it re-frames them to
    the exact size Silero requires and emits utterance boundaries.
    """

    def __init__(self, config, sample_rate: int = 16000, model: Optional[SileroVAD] = None):
        self.cfg = config
        self.sample_rate = sample_rate
        self.model = model or SileroVAD(sample_rate)

        self.state = VADState.IDLE
        self._frame_size = self.model.frame_size
        self._frame_ms = 1000.0 * self._frame_size / sample_rate

        # Carry-over buffer so arbitrary chunk sizes work.
        self._pending = np.zeros(0, dtype=np.float32)

        # Pre-roll ring buffer, sized in whole frames.
        pre_frames = max(1, int(self.cfg.pre_speech_pad_ms / self._frame_ms))
        self._preroll: collections.deque = collections.deque(maxlen=pre_frames)

        self._utterance: list[np.ndarray] = []
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._utterance_ms = 0.0
        self._post_pad_ms = 0.0
        self._last_prob = 0.0

    def load(self) -> None:
        self.model.load()

    def reset(self) -> None:
        self.state = VADState.IDLE
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll.clear()
        self._utterance = []
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._utterance_ms = 0.0
        self._post_pad_ms = 0.0
        self.model.reset()

    # -- main entry point --------------------------------------------------
    def process(self, chunk: np.ndarray) -> list[VADResult]:
        """Consume a chunk, return zero or more results (one per VAD frame)."""
        self._pending = np.concatenate([self._pending, chunk.astype(np.float32).reshape(-1)])
        results: list[VADResult] = []
        while self._pending.shape[0] >= self._frame_size:
            frame = self._pending[: self._frame_size]
            self._pending = self._pending[self._frame_size :]
            results.append(self._process_frame(frame))
        return results

    def _process_frame(self, frame: np.ndarray) -> VADResult:
        prob = self.model.probability(frame)
        self._last_prob = prob
        event = VADEvent.NONE
        segment = None
        seg_dur = 0.0

        above = prob >= self.cfg.speech_threshold
        below = prob < self.cfg.silence_threshold

        if self.state is VADState.IDLE:
            self._preroll.append(frame)
            if above:
                self.state = VADState.MAYBE_SPEECH
                self._speech_ms = self._frame_ms

        elif self.state is VADState.MAYBE_SPEECH:
            self._preroll.append(frame)
            if above:
                self._speech_ms += self._frame_ms
                if self._speech_ms >= self.cfg.min_speech_ms:
                    # Confirmed. Seed the utterance with the pre-roll history.
                    self.state = VADState.SPEAKING
                    self._utterance = list(self._preroll)
                    self._utterance_ms = len(self._utterance) * self._frame_ms
                    self._silence_ms = 0.0
                    event = VADEvent.SPEECH_STARTED
            else:
                self.state = VADState.IDLE
                self._speech_ms = 0.0

        elif self.state is VADState.SPEAKING:
            self._utterance.append(frame)
            self._utterance_ms += self._frame_ms
            event = VADEvent.SPEECH_CONTINUING
            if below:
                self.state = VADState.MAYBE_END
                self._silence_ms = self._frame_ms
            if self._utterance_ms >= self.cfg.max_utterance_ms:
                segment, seg_dur = self._finish()
                event = VADEvent.SPEECH_ENDED

        elif self.state is VADState.MAYBE_END:
            self._utterance.append(frame)
            self._utterance_ms += self._frame_ms
            if above:
                # Just a pause mid-sentence. Keep going.
                self.state = VADState.SPEAKING
                self._silence_ms = 0.0
                event = VADEvent.SPEECH_CONTINUING
            else:
                self._silence_ms += self._frame_ms
                if self._silence_ms >= self.cfg.min_silence_ms:
                    segment, seg_dur = self._finish()
                    event = VADEvent.SPEECH_ENDED
                else:
                    event = VADEvent.SPEECH_CONTINUING

        return VADResult(
            event=event,
            state=self.state,
            speech_probability=prob,
            is_speech=self.state in (VADState.SPEAKING, VADState.MAYBE_END),
            segment=segment,
            segment_duration_s=seg_dur,
        )

    def _finish(self) -> tuple[np.ndarray, float]:
        segment = np.concatenate(self._utterance) if self._utterance else np.zeros(0, np.float32)
        pad = int(self.sample_rate * self.cfg.post_speech_pad_ms / 1000.0)
        if pad > 0:
            segment = np.concatenate([segment, np.zeros(pad, dtype=np.float32)])
        duration = segment.shape[0] / float(self.sample_rate)
        self.state = VADState.IDLE
        self._utterance = []
        self._utterance_ms = 0.0
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._preroll.clear()
        logger.info("speech ended: %.2fs segment", duration)
        return segment, duration

    @property
    def last_probability(self) -> float:
        return self._last_prob
