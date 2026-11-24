"""
Your profile: a plain Markdown file (profile/profile.md) describing you --
routines, places, tastes, money rules, how you like to be spoken to.

It is read fresh whenever the file changes and placed in front of the model on
every turn, so editing the file changes Riya's behaviour immediately, with no
restart. The real file is git-ignored; profile/profile.example.md is the
template that is committed.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_CHARS = 12000


class Profile:
    def __init__(self, path: Path, max_chars: int = MAX_CHARS):
        self.path = Path(path)
        self.max_chars = max_chars
        self._mtime: float | None = None
        self._text = ""

    def text(self) -> str:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            self._mtime, self._text = None, ""
            return ""
        if mtime != self._mtime:
            raw = self.path.read_text(encoding="utf-8", errors="replace")
            # Drop HTML comments (template guidance) and blank-only sections.
            import re
            raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
            lines = [ln.rstrip() for ln in raw.splitlines()]
            text = "\n".join(ln for ln in lines if ln.strip()).strip()
            if len(text) > self.max_chars:
                logger.warning("profile is %d chars; only the first %d are used",
                               len(text), self.max_chars)
                text = text[: self.max_chars]
            self._mtime, self._text = mtime, text
            logger.info("profile loaded (%d chars)", len(text))
        return self._text
