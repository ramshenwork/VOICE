"""
The agent.

Input is text. Output is text. It has no idea a microphone exists, and that is
the point -- the same agent is driven by the local mic runner, the web server,
and the typed-input test harness without modification.

The tool loop
-------------
    user text
        -> model
        -> model asks for tool(s)?
             yes -> registry executes -> results appended -> model again
             no  -> that text is the answer

`max_tool_rounds` bounds the loop. A model that keeps asking for tools is
either confused or stuck, and in a voice interface a long silence is worse
than an imperfect answer, so the loop exits and asks the model for a final
spoken reply with tools disabled.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Union

from .llm import LLMError, LLMResponse, build_llm
from .prompts import FALLBACK_RESPONSES, SYSTEM_PROMPT
from .state import ConversationState

logger = logging.getLogger(__name__)


@dataclass
class AgentReply:
    text: str
    tools_used: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


class Agent:
    """
    `system_prompt` may be a string or a zero-argument callable. A callable is
    re-evaluated every turn, which is how the current time and freshly
    remembered facts reach the model without restarting anything.

    `on_turn(role, text)` is called for each finished user / assistant turn;
    the pipeline uses it to persist the conversation across restarts.
    """

    def __init__(self, config, registry, state: ConversationState | None = None,
                 system_prompt: Union[str, Callable[[], str]] = SYSTEM_PROMPT,
                 on_turn: Callable[[str, str], None] | None = None):
        self.cfg = config
        self.registry = registry
        self.state = state or ConversationState()
        self.system_prompt = system_prompt
        self.on_turn = on_turn
        self._llm = None

    def _system(self) -> str:
        sp = self.system_prompt
        return sp() if callable(sp) else sp

    def _log(self, role: str, text: str) -> None:
        if self.on_turn and text:
            try:
                self.on_turn(role, text)
            except Exception:  # persistence must never break a reply
                logger.exception("could not persist %s turn", role)

    def load(self) -> None:
        if self._llm is None:
            self._llm = build_llm(self.cfg)
            logger.info("llm ready: %s / %s", self.cfg.provider, self.cfg.model)

    def respond(self, user_text: str) -> AgentReply:
        """Blocking form: the whole reply at once."""
        for item in self.respond_stream(user_text):
            if isinstance(item, AgentReply):
                return item
        return AgentReply(FALLBACK_RESPONSES["llm_error"], error="no_reply")  # pragma: no cover

    def respond_stream(self, user_text: str) -> Iterator[Union[str, AgentReply]]:
        """
        Yields spoken text as it is produced -- whole sentences, ready for
        TTS -- and finally one AgentReply with the complete text.

        Text the model says *before* a tool call ("Let me check.") is yielded
        too, so the user hears something while the tool runs.
        """
        from tts.engine import SentenceBuffer

        if self._llm is None:
            self.load()

        t0 = time.perf_counter()
        self.state.add_user(user_text)
        self._log("user", user_text)
        specs = self.registry.specs()
        used: list[str] = []
        spoken: list[str] = []
        system = self._system()

        def finish(text: str, error: str = "") -> AgentReply:
            self._log("assistant", text)
            return AgentReply(text, used, time.perf_counter() - t0, error)

        for round_index in range(self.cfg.max_tool_rounds + 1):
            last_round = round_index == self.cfg.max_tool_rounds
            buf = SentenceBuffer()
            resp = None
            try:
                stream = getattr(self._llm, "stream", None)
                items = stream(system=system, messages=self.state.messages(),
                               tools=None if last_round else specs) if stream else \
                    iter([self._llm.complete(system=system, messages=self.state.messages(),
                                             tools=None if last_round else specs)])
                for item in items:
                    if isinstance(item, LLMResponse):
                        resp = item
                    elif isinstance(item, str) and item:
                        for sentence in buf.feed(item):
                            spoken.append(sentence)
                            yield sentence
            except LLMError as exc:
                logger.error("llm failure: %s", exc)
                fallback = FALLBACK_RESPONSES["llm_error"]
                self.state.add_assistant(fallback)
                yield fallback
                yield finish(fallback, str(exc))
                return

            if resp is None:
                resp = LLMResponse()
            if stream is None and resp.text:
                # Non-streaming client: speak the whole text now.
                for sentence in buf.feed(resp.text):
                    spoken.append(sentence)
                    yield sentence
            for sentence in buf.flush():
                spoken.append(sentence)
                yield sentence

            if not resp.wants_tools:
                text = resp.text
                if not text:
                    text = FALLBACK_RESPONSES["llm_error"]
                    yield text
                self.state.add_assistant(text)
                full = " ".join(spoken) if spoken else text
                yield finish(full)
                return

            self.state.add_assistant(resp.text, tool_calls=resp.tool_calls)
            for call in resp.tool_calls:
                logger.info("tool call: %s(%s)", call.name, call.arguments)
                result = self.registry.execute(call.name, call.arguments)
                used.append(call.name)
                self.state.add_tool_result(call.id, call.name, result.to_model_payload())

        # Unreachable in practice; the last round runs with tools disabled.
        text = FALLBACK_RESPONSES["llm_error"]
        self.state.add_assistant(text)
        yield text
        yield finish(text, "tool_loop_exhausted")

    def restore(self, turns: list[dict]) -> None:
        """Seed history with plain turns from a previous session."""
        for t in turns:
            if t["role"] == "user":
                self.state.add_user(t["content"])
            elif t["role"] == "assistant":
                self.state.add_assistant(t["content"])

    def reset(self) -> None:
        self.state.reset()
