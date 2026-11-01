"""
Speaker output.

Single responsibility: play float32 audio buffers, without ever blocking the
thread that produced them, and stop instantly when told to.

The stop() path is what makes barge-in possible: it clears the pending queue
and sets a flag the playback callback checks, so audio ceases within one
buffer period (~20 ms) rather than at the end of the sentence.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AudioOutputError(RuntimeError):
    pass


class SpeakerStream:
    """Queued, interruptible playback."""

    def __init__(self, sample_rate: int = 24000, device: str = "", blocksize: int = 1024):
        self.sample_rate = sample_rate
        self.device = device
        self.blocksize = blocksize
        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self._stream = None
        self._residual = np.zeros(0, dtype=np.float32)
        self._stop_flag = threading.Event()
        self._idle = threading.Event()
        self._idle.set()

    def start(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover
            raise AudioOutputError("sounddevice unavailable for playback.") from exc

        device = self.device or None
        if device is not None and device.isdigit():
            device = int(device)

        def callback(outdata, frames, time_info, status):
            if status:
                logger.debug("output stream status: %s", status)

            if self._stop_flag.is_set():
                self._residual = np.zeros(0, dtype=np.float32)
                outdata[:] = 0.0
                return

            buf = self._residual
            while buf.shape[0] < frames:
                try:
                    nxt = self._queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    break
                buf = np.concatenate([buf, nxt.astype(np.float32)])

            if buf.shape[0] == 0:
                self._idle.set()
                outdata[:] = 0.0
                return

            self._idle.clear()
            take = min(frames, buf.shape[0])
            outdata[:take, 0] = buf[:take]
            if take < frames:
                outdata[take:, 0] = 0.0
            self._residual = buf[take:]

        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                channels=1,
                dtype="float32",
                device=device,
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:
            raise AudioOutputError(f"Speaker could not be opened: {exc}") from exc
        logger.info("speaker opened rate=%d", self.sample_rate)

    def play(self, audio: np.ndarray) -> None:
        """Enqueue audio. Returns immediately."""
        if audio is None or audio.size == 0:
            return
        self._stop_flag.clear()
        self._idle.clear()
        self._queue.put(np.asarray(audio, dtype=np.float32).reshape(-1))

    def stop(self) -> None:
        """Barge-in: discard everything pending and silence output now."""
        self._stop_flag.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._residual = np.zeros(0, dtype=np.float32)
        self._idle.set()

    @property
    def is_playing(self) -> bool:
        return not self._idle.is_set()

    def wait_until_done(self, timeout: Optional[float] = None) -> None:
        self._idle.wait(timeout=timeout)

    def close(self) -> None:
        self.stop()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # pragma: no cover
                logger.warning("error closing speaker: %s", exc)
            self._stream = None

    def __enter__(self) -> "SpeakerStream":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """
    Cheap linear resampler.

    Used only on the output side (Kokoro emits 24 kHz; browsers and some
    devices want something else). Linear interpolation is audibly fine for
    speech at these ratios and costs almost nothing.
    """
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(np.float32)
    duration = audio.shape[0] / float(src_rate)
    n_out = int(round(duration * dst_rate))
    src_idx = np.linspace(0.0, audio.shape[0] - 1, num=n_out, dtype=np.float64)
    return np.interp(src_idx, np.arange(audio.shape[0]), audio).astype(np.float32)


def float_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
