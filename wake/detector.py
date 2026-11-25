"""
Wake word.

While asleep, the only thing running on the microphone is this: a tiny
openWakeWord model (about 1 MB, a few percent of one CPU core) listening for
"hey jarvis". Nothing is transcribed and nothing leaves the machine until it
fires.

openWakeWord wants 16 kHz int16 in 80 ms frames (1280 samples). The mic hands
us 512-sample float32 chunks, so this buffers and converts.

`AwakeSession` is the small piece of logic deciding whether we are currently
listening for a command: woken by the wake word, kept awake while a
conversation is going, and put back to sleep after a quiet follow-up window.
It is pure logic with an injectable clock, so it is tested without audio.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

OWW_FRAME = 1280


class WakeWordError(RuntimeError):
    pass


class WakeWordDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self._model = None
        self._buf = np.zeros(0, dtype=np.int16)
        self.last_score = 0.0

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import openwakeword.utils as oww_utils
            from openwakeword.model import Model
        except Exception as exc:
            raise WakeWordError("pip install openwakeword") from exc
        try:
            oww_utils.download_models([self.cfg.model])
        except Exception as exc:  # already cached, or offline
            logger.debug("wake word model download skipped: %s", exc)
        try:
            self._model = Model(wakeword_models=[self.cfg.model], inference_framework="onnx")
        except Exception as exc:
            raise WakeWordError(f"could not load wake word '{self.cfg.model}': {exc}") from exc
        logger.info("wake word ready: %s (threshold %.2f)", self.cfg.model, self.cfg.threshold)

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.int16)
        self.last_score = 0.0
        if self._model is not None:
            try:
                self._model.reset()
            except Exception:
                pass

    def process(self, chunk: np.ndarray) -> bool:
        """Feed float32 16 kHz audio. True the moment the wake word is heard."""
        pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
        self._buf = np.concatenate([self._buf, pcm])
        fired = False
        while self._buf.size >= OWW_FRAME:
            frame, self._buf = self._buf[:OWW_FRAME], self._buf[OWW_FRAME:]
            scores = self._model.predict(frame)
            self.last_score = max(scores.values()) if scores else 0.0
            if self.last_score >= self.cfg.threshold:
                fired = True
        if fired:
            self.reset()  # so one "hey jarvis" fires once
        return fired


class AwakeSession:
    """
    asleep --wake()--> awake --(reply finished)--> follow-up window
       ^                                                  |
       +---------- window expires with no speech ---------+
    """

    def __init__(self, follow_up_s: float, clock: Callable[[], float] = time.monotonic):
        self.follow_up_s = follow_up_s
        self.clock = clock
        self.awake = False
        self._deadline: float | None = None
        self._window_end: float | None = None

    def wake(self) -> None:
        self.awake = True
        self._deadline = self._window_end = self.clock() + max(self.follow_up_s, 6.0)

    def user_speaking(self) -> None:
        """Speech started: never time out mid-sentence."""
        self._deadline = None

    def reply_finished(self) -> None:
        self._deadline = self._window_end = self.clock() + self.follow_up_s

    def resume(self) -> None:
        """Speech that wasn't for her: carry on the window that was already
        running instead of granting a fresh one, so background talk can't
        keep her awake forever."""
        self._deadline = self._window_end if self._window_end is not None else self.clock()

    def sleep(self) -> None:
        self.awake = False
        self._deadline = self._window_end = None

    def expired(self) -> bool:
        return (self.awake and self._deadline is not None
                and self.clock() >= self._deadline)


class TranscriptWake:
    """
    Wake on a *name* heard in a transcript, for names with no trained
    wake-word model. While asleep, the VAD cuts speech into utterances, short
    ones are transcribed, and this decides whether the name was in it.

        match("Hey Riya, what's the weather?") -> (True, "what's the weather?")
        match("Riya")                          -> (True, "")
        match("I'm listening to Rhea's song")  -> (False, "")   # not addressed

    Only an utterance that *starts* by addressing her counts ("Riya ...",
    "Hey Riya ...", "OK Riya ..."), so the name merely being mentioned, or
    sung in a track, doesn't wake her.
    """

    GREETING = r"(?:hey|hi|hello|ok|okay|oi|yo)"

    def __init__(self, phrases: str | list[str], max_utterance_s: float = 8.0):
        import re
        names = phrases.split(",") if isinstance(phrases, str) else list(phrases)
        names = sorted({n.strip().lower() for n in names if n.strip()}, key=len, reverse=True)
        if not names:
            raise WakeWordError("WAKE_PHRASES is empty")
        alt = "|".join(re.escape(n) for n in names)
        self._re = re.compile(rf"^\W*(?:{self.GREETING}\W+)?(?:{alt})\b[\W]*", re.IGNORECASE)
        self.max_utterance_s = max_utterance_s
        self.names = names

    def match(self, text: str) -> tuple[bool, str]:
        m = self._re.match(text or "")
        if not m:
            return False, ""
        return True, text[m.end():].strip()
