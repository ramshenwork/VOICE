"""
Conversation state.

Deliberately minimal for V1: an in-memory transcript with a turn cap. There is
no vector store, no summarisation, no persistence. Adding those before the
pipeline works would be optimising a problem the project does not have yet.

The turn cap matters for latency as much as for cost -- every retained turn is
tokens the model re-reads before it produces its first spoken word.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationState:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: float = field(default_factory=time.time)
    max_turns: int = 12

    history: list[dict] = field(default_factory=list)
    last_tool: str = ""
    last_tool_result: Any = None
    turn_count: int = 0

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})
        self.turn_count += 1
        self._trim()

    def add_assistant(self, text: str, tool_calls: list | None = None) -> None:
        entry: dict = {"role": "assistant", "content": text}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        self.history.append(entry)
        self._trim()

    def add_tool_result(self, call_id: str, name: str, payload: dict) -> None:
        self.history.append({
            "role": "tool", "tool_call_id": call_id,
            "name": name, "content": payload,
        })
        self.last_tool = name
        self.last_tool_result = payload

    def messages(self) -> list[dict]:
        return list(self.history)

    def _trim(self) -> None:
        """
        Drop oldest turns beyond the cap, but never leave a tool result whose
        originating tool call has been trimmed away -- providers reject that
        as a malformed conversation. So we cut forward to the next clean
        user turn.
        """
        limit = self.max_turns * 3
        if len(self.history) <= limit:
            return
        idx = len(self.history) - limit
        while idx < len(self.history) and self.history[idx]["role"] != "user":
            idx += 1
        self.history = self.history[idx:]

    def reset(self) -> None:
        self.history = []
        self.last_tool = ""
        self.last_tool_result = None
        self.turn_count = 0
