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

from .llm import LLMError, build_llm
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
    def __init__(self, config, registry, state: ConversationState | None = None,
                 system_prompt: str = SYSTEM_PROMPT):
        self.cfg = config
        self.registry = registry
        self.state = state or ConversationState()
        self.system_prompt = system_prompt
        self._llm = None

    def load(self) -> None:
        if self._llm is None:
            self._llm = build_llm(self.cfg)
            logger.info("llm ready: %s / %s", self.cfg.provider, self.cfg.model)

    def respond(self, user_text: str) -> AgentReply:
        if self._llm is None:
            self.load()

        t0 = time.perf_counter()
        self.state.add_user(user_text)
        specs = self.registry.specs()
        used: list[str] = []

        for round_index in range(self.cfg.max_tool_rounds + 1):
            last_round = round_index == self.cfg.max_tool_rounds
            try:
                resp = self._llm.complete(
                    system=self.system_prompt,
                    messages=self.state.messages(),
                    tools=None if last_round else specs,
                )
            except LLMError as exc:
                logger.error("llm failure: %s", exc)
                fallback = FALLBACK_RESPONSES["llm_error"]
                self.state.add_assistant(fallback)
                return AgentReply(fallback, used, time.perf_counter() - t0, str(exc))

            if not resp.wants_tools:
                text = resp.text or FALLBACK_RESPONSES["llm_error"]
                self.state.add_assistant(text)
                return AgentReply(text, used, time.perf_counter() - t0)

            self.state.add_assistant(resp.text, tool_calls=resp.tool_calls)
            for call in resp.tool_calls:
                logger.info("tool call: %s(%s)", call.name, call.arguments)
                result = self.registry.execute(call.name, call.arguments)
                used.append(call.name)
                self.state.add_tool_result(call.id, call.name, result.to_model_payload())

        # Unreachable in practice; the last round runs with tools disabled.
        text = FALLBACK_RESPONSES["llm_error"]
        self.state.add_assistant(text)
        return AgentReply(text, used, time.perf_counter() - t0, "tool_loop_exhausted")

    def reset(self) -> None:
        self.state.reset()
