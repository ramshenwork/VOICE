"""
Text to speech.

Interface: `synthesize(text) -> np.ndarray` and `stream(text) -> iterator of
np.ndarray`. Nothing outside this file imports Kokoro.

Why streaming matters here more than anywhere else
--------------------------------------------------
Synthesising a three-sentence answer takes roughly three times as long as
synthesising one sentence. If the system waits for all of it, the user stares
at silence for the whole duration. If it emits sentence one while sentence two
is still being generated, perceived latency collapses to the cost of the first
clause -- and speech playback is slower than speech generation, so the buffer
never runs dry.

So the split is on sentence boundaries: small enough to start fast, large
enough that prosody does not sound chopped. Splitting on words would sound
robotic; splitting on paragraphs would defeat the purpose.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

# Split after . ! ? : ; when followed by whitespace. Avoids splitting on
# decimals and common abbreviations, which would produce audible stutter.
_SENTENCE = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9\"'(])")
_ABBREV = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e)\.$", re.IGNORECASE)


class TTSError(RuntimeError):
    pass


def split_for_speech(text: str, max_chars: int = 220) -> list[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    rough = _SENTENCE.split(text)

    merged: list[str] = []
    for part in rough:
        if merged and _ABBREV.search(merged[-1]):
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)

    chunks: list[str] = []
    for part in merged:
        while len(part) > max_chars:
            cut = part.rfind(", ", 0, max_chars)
            if cut < max_chars // 3:
                cut = part.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            chunks.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            chunks.append(part)
    return chunks


class TTSEngine:
    """Kokoro-backed synthesiser with a graceful system fallback."""

    def __init__(self, config):
        self.cfg = config
        self.sample_rate = config.sample_rate
        self._pipeline = None
        self._backend = ""

    def load(self) -> None:
        if self._pipeline is not None or self._backend == "system":
            return
        if self.cfg.engine.lower() == "system":
            self._backend = "system"
            self.sample_rate = 22050
            logger.info("tts backend: system (macOS say)")
            return
        try:
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code=self.cfg.lang_code)
            self._backend = "kokoro"
            self.sample_rate = 24000
            logger.info("tts backend: kokoro voice=%s", self.cfg.voice)
        except Exception as exc:
            logger.warning("kokoro unavailable (%s); falling back to system voice", exc)
            self._backend = "system"
            self.sample_rate = 22050

    @property
    def backend(self) -> str:
        return self._backend

    # -- public interface --------------------------------------------------
    def synthesize(self, text: str) -> np.ndarray:
        parts = list(self.stream(text))
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts)

    def stream(self, text: str) -> Iterator[np.ndarray]:
        if self._backend == "" and self._pipeline is None:
            self.load()
        for chunk in split_for_speech(text):
            audio = self._synth_one(chunk)
            if audio is not None and audio.size:
                yield audio

    # -- backends ----------------------------------------------------------
    def _synth_one(self, text: str) -> np.ndarray | None:
        if self._backend == "kokoro":
            try:
                out = []
                for _, _, audio in self._pipeline(text, voice=self.cfg.voice,
                                                  speed=self.cfg.speed):
                    arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") \
                        else np.asarray(audio)
                    out.append(arr.astype(np.float32).reshape(-1))
                return np.concatenate(out) if out else None
            except Exception as exc:
                logger.error("kokoro synthesis failed: %s", exc)
                return None
        return self._system_say(text)

    def _system_say(self, text: str) -> np.ndarray | None:
        """macOS `say`. Present so the pipeline is testable before Kokoro's
        weights are downloaded, and so a TTS failure degrades instead of
        killing the turn."""
        try:
            import soundfile as sf
        except Exception:
            logger.error("soundfile missing; cannot use system TTS")
            return None
        try:
            with tempfile.NamedTemporaryFile(suffix=".aiff", delete=True) as tmp:
                subprocess.run(["say", "-o", tmp.name, text], check=True,
                               capture_output=True, timeout=30)
                data, rate = sf.read(tmp.name, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            self.sample_rate = rate
            return data.astype(np.float32)
        except FileNotFoundError:
            logger.error("system TTS unavailable ('say' not found)")
        except Exception as exc:
            logger.error("system TTS failed: %s", exc)
        return None
