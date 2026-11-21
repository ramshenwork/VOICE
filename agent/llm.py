"""
LLM providers, normalised.

This is the hardest interface in the project, because the three realistic
back-ends disagree about everything: message shape, tool declaration format,
and how a tool call is represented in the reply.

The normalisation is:

    complete(system, messages, tools) -> LLMResponse(text, tool_calls)

`messages` uses one internal shape everywhere:

    {"role": "user"|"assistant", "content": "..."}
    {"role": "assistant", "tool_calls": [ToolCall...]}
    {"role": "tool", "tool_call_id": "...", "name": "...", "content": {...}}

Each provider translates that to and from its own wire format inside its own
class. Nothing above this file ever sees a provider-specific object.

Ollama note: small local models are unreliable at native tool calling, so the
Ollama client uses a JSON-contract prompt with a repair pass instead. That is
a real behavioural difference between providers and is documented rather than
hidden.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Iterator, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Optional[dict] = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMError(RuntimeError):
    pass


StreamItem = Union[str, LLMResponse]


class BaseLLM:
    def complete(self, system: str, messages: list[dict],
                 tools: list[dict] | None = None) -> LLMResponse:
        raise NotImplementedError

    def stream(self, system: str, messages: list[dict],
               tools: list[dict] | None = None) -> Iterator[StreamItem]:
        """Yield text deltas (str) as they arrive, then exactly one final
        LLMResponse holding the full text and any tool calls.

        Default: no real streaming, one delta then the response. Providers
        that can stream override this; the contract is the same either way."""
        resp = self.complete(system, messages, tools)
        if resp.text:
            yield resp.text
        yield resp


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------
class AnthropicLLM(BaseLLM):
    def __init__(self, cfg):
        self.cfg = cfg
        if not cfg.api_key:
            raise LLMError("LLM_API_KEY is required for LLM_PROVIDER=anthropic")
        try:
            import anthropic
        except Exception as exc:
            raise LLMError("pip install anthropic") from exc
        self._client = anthropic.Anthropic(api_key=cfg.api_key, timeout=cfg.timeout_s)

    def _kwargs(self, system, messages, tools) -> dict:
        kwargs = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            system=system,
            messages=[self._to_anthropic(m) for m in messages],
        )
        if tools:
            kwargs["tools"] = tools
        return kwargs

    def complete(self, system, messages, tools=None) -> LLMResponse:
        try:
            resp = self._client.messages.create(**self._kwargs(system, messages, tools))
        except Exception as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc
        return self._parse(resp)

    def stream(self, system, messages, tools=None):
        try:
            with self._client.messages.stream(**self._kwargs(system, messages, tools)) as s:
                for text in s.text_stream:
                    if text:
                        yield text
                final = s.get_final_message()
        except Exception as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc
        yield self._parse(final)

    @staticmethod
    def _parse(resp) -> LLMResponse:
        text_parts, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name,
                                      arguments=dict(block.input or {})))
        return LLMResponse(text="".join(text_parts).strip(), tool_calls=calls)

    @staticmethod
    def _to_anthropic(m: dict) -> dict:
        if m["role"] == "tool":
            return {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": json.dumps(m["content"]),
            }]}
        if m.get("tool_calls"):
            content = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                content.append({"type": "tool_use", "id": tc.id,
                                "name": tc.name, "input": tc.arguments})
            return {"role": "assistant", "content": content}
        return {"role": m["role"], "content": m["content"]}


# --------------------------------------------------------------------------
# OpenAI-compatible (OpenAI, Groq, Together, LM Studio, vLLM...)
# --------------------------------------------------------------------------
class OpenAILLM(BaseLLM):
    def __init__(self, cfg):
        self.cfg = cfg
        try:
            from openai import OpenAI
        except Exception as exc:
            raise LLMError("pip install openai") from exc
        self._client = OpenAI(
            api_key=cfg.api_key or "not-needed",
            base_url=cfg.base_url or None,
            timeout=cfg.timeout_s,
        )

    def _kwargs(self, system, messages, tools) -> dict:
        payload = [{"role": "system", "content": system}]
        for m in messages:
            payload.append(self._to_openai(m))
        kwargs = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            messages=payload,
        )
        if tools:
            kwargs["tools"] = [{
                "type": "function",
                "function": {"name": t["name"], "description": t["description"],
                             "parameters": t["input_schema"]},
            } for t in tools]
        return kwargs

    def stream(self, system, messages, tools=None):
        text_parts: list[str] = []
        pending: dict[int, dict] = {}
        try:
            for chunk in self._client.chat.completions.create(
                    stream=True, **self._kwargs(system, messages, tools)):
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    text_parts.append(delta.content)
                    yield delta.content
                for tc in (delta.tool_calls or []):
                    slot = pending.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["args"] += tc.function.arguments
        except Exception as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        calls = []
        for i in sorted(pending):
            slot = pending[i]
            try:
                args = json.loads(slot["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=slot["id"] or f"call_{uuid.uuid4().hex[:8]}",
                                  name=slot["name"], arguments=args))
        yield LLMResponse(text="".join(text_parts).strip(), tool_calls=calls)

    def complete(self, system, messages, tools=None) -> LLMResponse:
        try:
            resp = self._client.chat.completions.create(**self._kwargs(system, messages, tools))
        except Exception as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        msg = resp.choices[0].message
        calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return LLMResponse(text=(msg.content or "").strip(), tool_calls=calls)

    @staticmethod
    def _to_openai(m: dict) -> dict:
        if m["role"] == "tool":
            return {"role": "tool", "tool_call_id": m["tool_call_id"],
                    "content": json.dumps(m["content"])}
        if m.get("tool_calls"):
            return {"role": "assistant", "content": m.get("content") or None,
                    "tool_calls": [{
                        "id": tc.id, "type": "function",
                        "function": {"name": tc.name,
                                     "arguments": json.dumps(tc.arguments)},
                    } for tc in m["tool_calls"]]}
        return {"role": m["role"], "content": m["content"]}


# --------------------------------------------------------------------------
# Ollama (local) -- JSON contract instead of native tool calling
# --------------------------------------------------------------------------
JSON_CONTRACT = """
You must reply with a single JSON object and nothing else. No prose, no
markdown fences.

To call a tool:
  {"action": "tool", "tool": "<tool_name>", "arguments": {...}}

To answer the user:
  {"action": "answer", "text": "<what to say out loud>"}

Available tools:
%s
"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class OllamaLLM(BaseLLM):
    def __init__(self, cfg):
        self.cfg = cfg
        self.base_url = (cfg.base_url or "http://localhost:11434").rstrip("/")

    def complete(self, system, messages, tools=None) -> LLMResponse:
        import httpx
        sys_prompt = system
        if tools:
            listing = "\n".join(
                f"- {t['name']}: {t['description']}\n  arguments schema: "
                f"{json.dumps(t['input_schema'])}" for t in tools
            )
            sys_prompt = system + "\n" + (JSON_CONTRACT % listing)

        payload = [{"role": "system", "content": sys_prompt}]
        for m in messages:
            if m["role"] == "tool":
                payload.append({"role": "user", "content":
                                f"Result from {m['name']}: {json.dumps(m['content'])}"})
            elif m.get("tool_calls"):
                payload.append({"role": "assistant", "content": json.dumps({
                    "action": "tool", "tool": m["tool_calls"][0].name,
                    "arguments": m["tool_calls"][0].arguments})})
            else:
                payload.append({"role": m["role"], "content": m["content"]})

        try:
            with httpx.Client(timeout=self.cfg.timeout_s) as client:
                r = client.post(f"{self.base_url}/api/chat", json={
                    "model": self.cfg.model, "messages": payload, "stream": False,
                    "format": "json" if tools else None,
                    "options": {"temperature": self.cfg.temperature,
                                "num_predict": self.cfg.max_tokens},
                })
                r.raise_for_status()
                content = r.json()["message"]["content"]
        except Exception as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

        if not tools:
            return LLMResponse(text=content.strip())
        return self._parse(content)

    @staticmethod
    def _parse(content: str) -> LLMResponse:
        """Repair pass: small models fence their JSON or add a sentence
        around it. Extract the outermost object and fall back to treating the
        whole reply as speech."""
        text = content.strip()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            match = _JSON_BLOCK.search(text)
            if not match:
                return LLMResponse(text=text)
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError:
                return LLMResponse(text=text)

        if not isinstance(obj, dict):
            return LLMResponse(text=text)
        if obj.get("action") == "tool" and obj.get("tool"):
            return LLMResponse(tool_calls=[ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=str(obj["tool"]),
                arguments=obj.get("arguments") or {},
            )])
        return LLMResponse(text=str(obj.get("text") or obj.get("answer") or text).strip())


def build_llm(cfg) -> BaseLLM:
    provider = cfg.provider.lower()
    if provider == "anthropic":
        return AnthropicLLM(cfg)
    if provider in ("openai", "openai_compatible", "groq", "together"):
        return OpenAILLM(cfg)
    if provider in ("ollama", "local"):
        return OllamaLLM(cfg)
    raise LLMError(f"Unknown LLM_PROVIDER: {cfg.provider}")
