"""
Attention: deciding whether something the microphone heard was meant for Riya.

Three failure modes this exists to stop:

1. Hearing herself. Without headphones her own voice comes back through the
   mic. `EchoGuard` remembers what she said recently and when she stopped
   talking; a transcript that matches her own words, or speech that starts in
   the instant after she stops, is dropped.

2. Answering conversations that aren't for her (you on a phone call, someone
   in the room). After a reply she keeps listening for a follow-up, but
   `AttentionGate` only lets an utterance through if it is plausibly addressed
   to her: it starts with her name, it's a short reply to a question she just
   asked, or a quick yes/no classification by the LLM says so. Two ignored
   utterances in a row and she goes back to sleep.

3. Being told to be quiet. `quiet_request()` spots "I'm on a call", "not now",
   "shut up" and the like, so she goes silent instantly without a word.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9']+")


def _words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


QUIET_PHRASES = re.compile(
    r"\b(i'?m on (a|the) (call|phone)|on a call|taking a call|be quiet|keep quiet|quiet please|shut up|"
    r"shush|stop talking|not now|not you|i'?m not talking to you|wasn'?t talking to you|"
    r"leave it|go to sleep|stop listening|that'?s all|that will be all|never ?mind|"
    r"good ?night|thanks,? that'?s it)\b",
    re.IGNORECASE)

STOP_PHRASES = re.compile(r"^\W*(stop|wait|hold on|hold|enough|okay stop|ok stop)\b",
                          re.IGNORECASE)


def quiet_request(text: str) -> bool:
    """True if this utterance tells her to stop/sleep. Kept to short
    utterances so 'how do I stop the server' isn't mistaken for one."""
    return bool(text) and len(_words(text)) <= 8 and bool(QUIET_PHRASES.search(text))


class EchoGuard:
    def __init__(self, memory_s: float = 30.0, tail_s: float = 0.8,
                 clock: Callable[[], float] = time.monotonic):
        self.memory_s = memory_s
        self.tail_s = tail_s
        self.clock = clock
        self._said: list[tuple[float, list[str]]] = []
        self._stopped_at: float | None = None

    def spoke(self, sentence: str) -> None:
        self._said.append((self.clock(), _words(sentence)))
        self._prune()

    def playback_ended(self) -> None:
        self._stopped_at = self.clock()

    def in_tail(self) -> bool:
        """Just stopped talking: the room is still ringing with her voice."""
        return self._stopped_at is not None and self.clock() - self._stopped_at < self.tail_s

    def _prune(self) -> None:
        cutoff = self.clock() - self.memory_s
        self._said = [(t, w) for t, w in self._said if t >= cutoff]

    def is_echo(self, text: str, threshold: float = 0.6) -> bool:
        """Does this transcript mostly consist of something she just said?"""
        heard = _words(text)
        if len(heard) < 2:
            return False
        self._prune()
        recent = [w for _, ws in self._said for w in ws]
        if not recent:
            return False
        pool = set(recent)
        overlap = sum(1 for w in heard if w in pool) / len(heard)
        if overlap >= threshold:
            return True
        # Ordered match: a 4-word run of her own words is almost certainly echo.
        joined = " " + " ".join(recent) + " "
        for i in range(len(heard) - 3):
            if " " + " ".join(heard[i:i + 4]) + " " in joined:
                return True
        return False


GATE_PROMPT = """You decide whether speech picked up by a voice assistant's microphone \
was said TO the assistant, whose name is {name}. The user may be on a phone call, \
talking to someone in the room, watching a video, or talking to {name}.

{name} last said: "{last}"
The microphone then heard: "{heard}"

Is that speech addressed to {name}? YES if it is a follow-up on her topic (even a \
fragment like "and tomorrow?" or "what about the weekend"), an answer to her, or a \
request an assistant could act on (notes, reminders, memory, maths, weather, plans, \
spending). Speech that sounds like one side of a phone call or a conversation \
with someone else, or that doesn't follow from what {name} said, is NOT addressed to her. \

Examples: "and tomorrow?" YES. "can you note that down" YES. "remind me what I spent" \
YES. "yeah I'll reach by seven, send me the pin" NO. "tell him the meeting moved" NO. \
"haan, I ate already" NO.

Reply with exactly one word: YES or NO."""


class AttentionGate:
    """
    classify(prompt) -> str is any cheap LLM call; it's injected so this is
    testable and so a failure can't take down the voice loop.
    """

    def __init__(self, name: str, aliases: list[str], classify: Optional[Callable[[str], str]],
                 max_ignored: int = 2):
        names = {n.strip().lower() for n in [name, *aliases] if n and n.strip()}
        alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
        self._named = re.compile(rf"\b(?:{alt})\b", re.IGNORECASE)
        self.name = name
        self.classify = classify
        self.max_ignored = max_ignored
        self.ignored_in_a_row = 0
        self.last_reply = ""

    def reset(self) -> None:
        self.ignored_in_a_row = 0

    def addressed(self, text: str) -> bool:
        """Is this follow-up utterance meant for her?"""
        verdict = self._decide(text)
        if verdict:
            self.ignored_in_a_row = 0
        else:
            self.ignored_in_a_row += 1
        logger.info("attention: %r -> %s", text, "for me" if verdict else "not for me")
        return verdict

    @property
    def should_sleep(self) -> bool:
        return self.ignored_in_a_row >= self.max_ignored

    def _decide(self, text: str) -> bool:
        if self._named.search(text or ""):
            return True
        words = _words(text)
        if not words:
            return False
        # A short answer straight after she asked something.
        if self.last_reply.rstrip().endswith("?") and len(words) <= 4:
            return True
        if self.classify is None:
            return len(words) <= 12
        try:
            out = self.classify(GATE_PROMPT.format(
                name=self.name, last=self.last_reply[-300:] or "(nothing yet)",
                heard=text[:400]))
        except Exception as exc:
            logger.warning("attention classifier failed (%s); assuming not for me", exc)
            return False
        return out.strip().upper().startswith("Y")


def llm_classifier(llm_cfg) -> Callable[[str], str]:
    """A tiny, fast yes/no call on the main model."""
    from dataclasses import replace

    from agent.llm import build_llm
    llm = build_llm(replace(llm_cfg, max_tokens=3, temperature=0.0,
                            timeout_s=min(llm_cfg.timeout_s, 6.0)))

    def classify(prompt: str) -> str:
        return llm.complete(system="Answer YES or NO only.",
                            messages=[{"role": "user", "content": prompt}]).text
    return classify
