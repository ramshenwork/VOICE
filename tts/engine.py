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
import time
from pathlib import Path
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


class SentenceBuffer:
    """
    Turns a stream of LLM text deltas into speakable sentences as soon as each
    one is complete, so the first sentence can be synthesised while the model
    is still writing the second.

        buf = SentenceBuffer()
        for delta in deltas:
            for sentence in buf.feed(delta): speak(sentence)
        for sentence in buf.flush(): speak(sentence)
    """

    def __init__(self, max_chars: int = 220):
        self.max_chars = max_chars
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        self._buf += delta or ""
        parts = split_for_speech(self._buf, self.max_chars)
        if len(parts) <= 1:
            # One part may still be growing -- unless it is already too long
            # to wait for, in which case split_for_speech would have cut it.
            return []
        ready, last = parts[:-1], parts[-1]
        # Keep the unfinished tail (with its original trailing space, if any).
        self._buf = last + (" " if self._buf.endswith((" ", "\n")) else "")
        return ready

    def flush(self) -> list[str]:
        parts = split_for_speech(self._buf, self.max_chars)
        self._buf = ""
        return parts


class CloneVoice:
    """
    Client for tts/clone_server.py. Starts the server in its own virtualenv
    if it isn't already running, and stops it again on exit.
    """

    def __init__(self, cfg, root: Path):
        self.cfg = cfg
        self.root = root
        self.url = f"http://127.0.0.1:{cfg.clone_port}"
        self.sample_rate = 24000
        self._proc = None

    def _path(self, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else self.root / q

    def _health(self, timeout: float = 1.0) -> dict | None:
        import json
        import urllib.request
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=timeout) as r:
                return json.loads(r.read())
        except Exception:
            return None

    def start(self, wait_s: float = 600.0) -> None:
        info = self._health()
        if info is None:
            python = self._path(self.cfg.clone_python)
            if not python.exists():
                raise TTSError(f"{python} not found -- set up .venv-voice first (see README)")
            sample = self._path(self.cfg.clone_sample)
            cmd = [str(python), str(self.root / "tts" / "clone_server.py"),
                   "--port", str(self.cfg.clone_port),
                   "--model", self.cfg.clone_model,
                   "--voice", str(sample) if sample.exists() else "",
                   "--exaggeration", str(self.cfg.clone_exaggeration),
                   "--cfg-weight", str(self.cfg.clone_cfg_weight)]
            if not sample.exists():
                logger.warning("voice sample %s not found; clone server uses its default voice",
                               sample)
            log_path = self.root / "logs" / "clone_tts.log"
            log_path.parent.mkdir(exist_ok=True)
            self._proc = subprocess.Popen(cmd, stdout=open(log_path, "ab"),
                                          stderr=subprocess.STDOUT, cwd=self.root)
            import atexit
            atexit.register(self.stop)
            logger.info("starting cloned-voice server (first run downloads the model)...")
            deadline = time.monotonic() + wait_s
            while (info := self._health()) is None:
                if self._proc.poll() is not None:
                    raise TTSError(f"clone server exited; see {log_path}")
                if time.monotonic() > deadline:
                    self.stop()
                    raise TTSError("clone server did not start in time")
                time.sleep(1.0)
        self.sample_rate = int(info.get("sample_rate", 24000))
        logger.info("tts backend: cloned voice (%s on %s)", info.get("voice"), info.get("device"))

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self._proc = None

    def speak(self, text: str) -> np.ndarray | None:
        import json
        import urllib.request
        req = urllib.request.Request(self.url + "/speak", data=json.dumps({"text": text}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                rate = int(r.headers.get("X-Sample-Rate", self.sample_rate))
                pcm = np.frombuffer(r.read(), dtype="<f4").astype(np.float32)
        except Exception as exc:
            logger.error("cloned voice failed: %s", exc)
            return None
        if rate != self.sample_rate:
            logger.warning("clone server rate changed %s -> %s", self.sample_rate, rate)
        return pcm


class TTSEngine:
    """Kokoro-backed synthesiser with a cloned-voice option and a graceful
    system fallback."""

    def __init__(self, config, root: Path | None = None):
        self.cfg = config
        self.sample_rate = config.sample_rate
        self._pipeline = None
        self._backend = ""
        self._clone: CloneVoice | None = None
        self._root = root or Path(__file__).resolve().parent.parent

    def load(self) -> None:
        if self._pipeline is not None or self._backend in ("system", "clone"):
            return
        if self.cfg.engine.lower() == "clone":
            try:
                clone = CloneVoice(self.cfg, self._root)
                clone.start()
                self._clone, self._backend = clone, "clone"
                self.sample_rate = clone.sample_rate
                return
            except Exception as exc:
                logger.warning("cloned voice unavailable (%s); using kokoro", exc)
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
        if self._backend == "clone":
            return self._clone.speak(text)
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
