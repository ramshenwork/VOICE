"""
Tool contract.

Security stance: the model never executes code. It emits a tool *name* and a
JSON argument object. The registry looks that name up in a fixed dictionary
built at startup. An unknown name is an error, not an invitation. There is no
eval, no exec, no shell. The worst a confused model can do is call a
registered tool with bad arguments, which the tool itself validates.

Tool results are treated as untrusted input to the model, not as instructions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str = ""
    tool: str = ""
    metadata: dict = field(default_factory=dict)

    def to_model_payload(self) -> dict:
        """What the LLM actually sees. Errors are explicit so it cannot
        mistake a failure for a result and invent an answer."""
        if self.ok:
            return {"status": "ok", "result": self.data}
        return {"status": "error", "error": self.error or "The tool failed."}


class Tool(ABC):
    name: str = "tool"
    description: str = ""

    @property
    @abstractmethod
    def input_schema(self) -> dict:
        """JSON Schema for the arguments object."""

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        ...

    def spec(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def _ok(self, data, **meta) -> ToolResult:
        return ToolResult(True, data=data, tool=self.name, metadata=meta)

    def _fail(self, error: str, **meta) -> ToolResult:
        logger.warning("tool %s failed: %s", self.name, error)
        return ToolResult(False, error=error, tool=self.name, metadata=meta)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool
        logger.info("registered tool: %s", tool.name)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def execute(self, name: str, arguments: dict) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                False,
                error=f"No such tool: {name}. Available: {', '.join(self._tools)}",
                tool=name,
            )
        if not isinstance(arguments, dict):
            return ToolResult(False, error="Tool arguments must be an object.", tool=name)
        try:
            return tool.execute(**arguments)
        except TypeError as exc:
            return ToolResult(False, error=f"Invalid arguments for {name}: {exc}", tool=name)
        except Exception as exc:  # a tool must never take down the agent
            logger.exception("tool %s raised", name)
            return ToolResult(False, error=f"{name} failed: {exc}", tool=name)
