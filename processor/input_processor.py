"""
Input processing.

A deliberately thin seam between the transcriber and the agent. Its job is
janitorial, not semantic: tidy the raw transcript and decide whether it is
worth waking the agent for at all.

This layer is NOT a second language model. Meaning is the agent's job. Keeping
this small is a design choice, not an oversight -- every rule added here is a
rule that can silently mangle a user's words before anyone reasons about them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

FILLERS = {"uh", "um", "erm", "uhh", "umm", "hmm", "mhm", "ah", "eh"}

# Things Whisper emits when handed silence or noise. These are hallucinations,
# not speech, and must not reach the agent.
HALLUCINATIONS = {
    "you", "thank you", "thanks for watching", "thank you for watching",
    "bye", "bye.", ".", "!", "?", "...", "[blank_audio]", "[silence]",
    "please subscribe", "subscribe to my channel", "you're welcome",
}

_WS = re.compile(r"\s+")
_BRACKETS = re.compile(r"[\[\(<][^\]\)>]{0,40}[\]\)>]")  # [MUSIC], (laughs)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])")
_REPEATED_WORD = re.compile(r"\b(\w+)(\s+\1\b)+", re.IGNORECASE)


@dataclass
class ProcessedInput:
    text: str
    raw_text: str
    should_respond: bool
    reason: str = ""
    metadata: dict = field(default_factory=dict)


class InputProcessor:
    def __init__(self, min_chars: int = 2, strip_fillers: bool = True,
                 wake_word: str = "", require_wake_word: bool = False):
        self.min_chars = min_chars
        self.strip_fillers = strip_fillers
        self.wake_word = wake_word.lower().strip()
        self.require_wake_word = require_wake_word and bool(self.wake_word)

    def process(self, transcription) -> ProcessedInput:
        raw = getattr(transcription, "text", transcription) or ""
        text = self._clean(raw)
        meta = {
            "duration_s": getattr(transcription, "duration_s", 0.0),
            "avg_logprob": getattr(transcription, "avg_logprob", None),
        }

        if not text or len(text) < self.min_chars:
            return ProcessedInput(text, raw, False, "empty_or_too_short", meta)

        stripped = text.lower().strip(" .!?,;:-'\"")
        if not stripped:
            return ProcessedInput(text, raw, False, "punctuation_only", meta)

        if stripped in HALLUCINATIONS or text.lower().strip() in HALLUCINATIONS:
            logger.debug("discarded likely STT hallucination: %r", text)
            return ProcessedInput(text, raw, False, "likely_hallucination", meta)

        if self.require_wake_word:
            lowered = text.lower()
            if self.wake_word not in lowered:
                return ProcessedInput(text, raw, False, "no_wake_word", meta)
            idx = lowered.index(self.wake_word) + len(self.wake_word)
            text = text[idx:].lstrip(" ,.").strip() or text
            meta["wake_word_stripped"] = True

        return ProcessedInput(self._capitalize(text), raw, True, "ok", meta)

    # -- cleaning steps ----------------------------------------------------
    def _clean(self, text: str) -> str:
        text = _BRACKETS.sub(" ", text)
        text = _WS.sub(" ", text).strip()
        if self.strip_fillers and text:
            words = [w for w in text.split(" ")
                     if w.lower().strip(",.!?") not in FILLERS]
            text = " ".join(words)
        text = _REPEATED_WORD.sub(r"\1", text)
        text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
        return text.strip()

    @staticmethod
    def _capitalize(text: str) -> str:
        if not text:
            return text
        return text[0].upper() + text[1:]
