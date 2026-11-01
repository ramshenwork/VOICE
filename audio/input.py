"""
Microphone input.

Single responsibility: turn a physical microphone into a stream of small,
fixed-size float32 mono PCM chunks. It knows nothing about speech, words,
or meaning.

Design note -- the callback boundary:
    sounddevice invokes its callback on a high-priority audio thread owned by
    the OS. Doing anything slow there (VAD inference, disk, logging) causes
    dropouts. So the callback does exactly one thing: push a copy of the block
    into a bounded queue. All real work happens on the consumer thread that
    drains that queue via chunks().
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AudioInputError(RuntimeError):
    pass


class MicrophoneStream:
    """Live microphone as an iterator of float32 mono chunks in [-1.0, 1.0]."""

    def __init__(self, config, max_queue_chunks: int = 100):
        self.cfg = config
        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=max_queue_chunks)
        self._stream = None
        self._running = threading.Event()
        self._dropped = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover
            raise AudioInputError(
                "sounddevice is not available. Install PortAudio and 'sounddevice'."
            ) from exc

        device = self.cfg.input_device or None
        if device is not None and device.isdigit():
            device = int(device)

        def callback(indata, frames, time_info, status):
            if status:
                logger.debug("input stream status: %s", status)
            try:
                self._queue.put_nowait(indata[:, 0].copy())
            except queue.Full:
                self._dropped += 1

        try:
            self._stream = sd.InputStream(
                samplerate=self.cfg.sample_rate,
                blocksize=self.cfg.chunk_size,
                channels=self.cfg.channels,
                dtype="float32",
                device=device,
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:
            raise AudioInputError(
                f"Microphone could not be opened: {exc}. "
                "Check System Settings > Privacy & Security > Microphone."
            ) from exc

        self._running.set()
        logger.info(
            "microphone opened rate=%d chunk=%d (%.1f ms) channels=%d",
            self.cfg.sample_rate, self.cfg.chunk_size, self.cfg.chunk_ms, self.cfg.channels,
        )

    def stop(self) -> None:
        self._running.clear()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # pragma: no cover
                logger.warning("error closing microphone: %s", exc)
            self._stream = None
        try:
            self._queue.put_nowait(None)  # unblock a waiting consumer
        except queue.Full:
            pass
        if self._dropped:
            logger.warning("dropped %d audio chunks (consumer too slow)", self._dropped)

    # -- consumption -------------------------------------------------------
    def chunks(self) -> Iterator[np.ndarray]:
        """Blocking generator of audio chunks. Ends when stop() is called."""
        while self._running.is_set():
            try:
                chunk = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if chunk is None:
                break
            yield chunk

    @property
    def dropped_chunks(self) -> int:
        return self._dropped

    def __enter__(self) -> "MicrophoneStream":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def list_input_devices() -> list[dict]:
    """Diagnostic helper for Phase 0/1."""
    import sounddevice as sd
    out = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            out.append({
                "index": idx,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "default_samplerate": dev["default_samplerate"],
            })
    return out


def rms(chunk: np.ndarray) -> float:
    """Root-mean-square amplitude. Used for level meters and sanity checks."""
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
