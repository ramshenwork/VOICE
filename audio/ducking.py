"""
Music ducking: turn Spotify down while the assistant is listening or talking,
and put it back afterwards.

macOS only, via AppleScript. It checks Spotify is already running before
touching it -- `tell application "Spotify"` on its own would launch the app.
Calls run on one background worker thread -- in order, so a restore can never
overtake the duck it undoes -- and a slow osascript (~100 ms) never delays
the wake response. On any other platform, or if Spotify isn't open, every
method is a silent no-op.
"""

from __future__ import annotations

import logging
import queue
import subprocess
import sys
import threading
from typing import Callable

logger = logging.getLogger(__name__)

_GET = 'if application "Spotify" is running then tell application "Spotify" to get sound volume'
_SET = 'if application "Spotify" is running then tell application "Spotify" to set sound volume to {v}'


def _osascript(script: str) -> str:
    out = subprocess.run(["osascript", "-e", script], capture_output=True,
                         text=True, timeout=3)
    return out.stdout.strip()


class SpotifyDucker:
    def __init__(self, enabled: bool = True, duck_volume: int = 15,
                 runner: Callable[[str], str] = _osascript, background: bool = True):
        self.enabled = enabled and (sys.platform == "darwin" or runner is not _osascript)
        self.duck_volume = max(0, min(100, int(duck_volume)))
        self._run = runner
        self._background = background
        self._saved: int | None = None
        self._jobs: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self._worker: threading.Thread | None = None

    def duck(self) -> None:
        self._dispatch(self._duck)

    def restore(self) -> None:
        self._dispatch(self._restore)

    def _dispatch(self, fn) -> None:
        if not self.enabled:
            return
        if not self._background:
            fn()
            return
        if self._worker is None:
            self._worker = threading.Thread(target=self._work, daemon=True)
            self._worker.start()
        self._jobs.put(fn)

    def _work(self) -> None:
        while True:
            self._jobs.get()()

    def _duck(self) -> None:
        if self._saved is not None:
            return  # already ducked
        try:
            current = self._run(_GET)
            if not current.isdigit():
                return  # Spotify not running
            level = int(current)
            if level <= self.duck_volume:
                return
            self._saved = level
            self._run(_SET.format(v=self.duck_volume))
        except Exception as exc:
            logger.debug("spotify duck failed: %s", exc)

    def _restore(self) -> None:
        if self._saved is None:
            return
        try:
            self._run(_SET.format(v=self._saved))
        except Exception as exc:
            logger.debug("spotify restore failed: %s", exc)
        finally:
            self._saved = None
