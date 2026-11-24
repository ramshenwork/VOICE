"""
The "superbrain": tools that make Riya reliably right instead of plausibly
right.

  calculate      exact arithmetic. Language models guess at maths; this doesn't.
  dates          calendar arithmetic (weekdays, days between, N days from now).
  knowledge      search your own documents in knowledge/.
  think          hand a hard, multi-step problem (plans, trade-offs, scenarios)
                 to a stronger reasoning model with your full context, and get
                 back a considered answer.

Security: `calculate` parses the expression into a syntax tree and evaluates
only numbers, arithmetic operators and a fixed whitelist of math functions.
There is no eval(), no attribute access, no names beyond the whitelist.
"""

from __future__ import annotations

import ast
import logging
import math
import operator
from dataclasses import replace
from datetime import date, datetime, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

from .base import Tool, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- calculate
_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable] = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "abs": abs, "round": round,
    "min": min, "max": max, "ceil": math.ceil, "floor": math.floor,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval(expr: str) -> float:
    expr = (expr or "").replace("^", "**").replace("×", "*").replace("÷", "/")
    expr = expr.replace(",", "") if "(" not in expr else expr
    if len(expr) > 300:
        raise ValueError("expression too long")
    tree = ast.parse(expr, mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FUNCS and not node.keywords):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"not allowed: {ast.dump(node)[:40]}")

    return ev(tree)


class CalculatorTool(Tool):
    name = "calculate"
    description = (
        "Exact arithmetic. ALWAYS use this for any sum, percentage, split, EMI, unit "
        "conversion or budget maths instead of working it out yourself. Takes one Python-"
        "style expression, e.g. '1840 * 0.18', '(52000 - 31500) / 30', "
        "'250000 * 0.0075 * (1.0075**36) / (1.0075**36 - 1)'. Functions: sqrt, log, exp, "
        "round, min, max, ceil, floor, abs. Several expressions can be separated by ';'."
    )

    @property
    def input_schema(self) -> dict:
        return {"type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]}

    def execute(self, expression: str = "", **_) -> ToolResult:
        parts = [p.strip() for p in (expression or "").split(";") if p.strip()]
        if not parts:
            return self._fail("No expression given.")
        results = []
        for p in parts[:10]:
            try:
                v = safe_eval(p)
            except ZeroDivisionError:
                return self._fail(f"Division by zero in {p!r}.")
            except Exception as exc:
                return self._fail(f"Couldn't evaluate {p!r}: {exc}")
            if isinstance(v, float):
                v = int(v) if v.is_integer() and abs(v) < 1e15 else round(v, 6)
            results.append({"expression": p, "value": v})
        return self._ok(results[0] if len(results) == 1 else results)


# -------------------------------------------------------------------- dates
class DatesTool(Tool):
    name = "dates"
    description = (
        "Calendar arithmetic -- use instead of guessing. action 'weekday': what day of "
        "the week a date is. 'add': a date plus/minus some days (days may be negative). "
        "'between': days from one date to another. 'next_weekday': the date of the next "
        "given weekday (e.g. 'friday'). Dates are YYYY-MM-DD; omit for today."
    )

    WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    def __init__(self, timezone: str = "Asia/Kolkata"):
        self.timezone = timezone

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string",
                           "enum": ["weekday", "add", "between", "next_weekday"]},
                "date": {"type": "string", "description": "YYYY-MM-DD; default today."},
                "other_date": {"type": "string", "description": "For 'between'."},
                "days": {"type": "integer", "description": "For 'add'."},
                "weekday": {"type": "string", "description": "For 'next_weekday'."},
            },
            "required": ["action"],
        }

    def _today(self) -> date:
        try:
            return datetime.now(ZoneInfo(self.timezone)).date()
        except Exception:
            return date.today()

    def _parse(self, s: str | None) -> date:
        return date.fromisoformat(s.strip()) if s and s.strip() else self._today()

    def _fmt(self, d: date) -> dict:
        return {"date": d.isoformat(), "weekday": d.strftime("%A"),
                "spoken": d.strftime("%A %d %B %Y")}

    def execute(self, action: str = "weekday", date: str = "", other_date: str = "",
                days: int = 0, weekday: str = "", **_) -> ToolResult:
        try:
            d = self._parse(date)
            if action == "add":
                return self._ok(self._fmt(d + timedelta(days=int(days))))
            if action == "between":
                o = self._parse(other_date)
                return self._ok({"from": d.isoformat(), "to": o.isoformat(),
                                 "days": (o - d).days})
            if action == "next_weekday":
                w = (weekday or "").strip().lower()
                if w not in self.WEEKDAYS:
                    return self._fail("weekday must be a day name like 'friday'.")
                ahead = (self.WEEKDAYS.index(w) - d.weekday()) % 7 or 7
                return self._ok(self._fmt(d + timedelta(days=ahead)))
            return self._ok(self._fmt(d))
        except ValueError as exc:
            return self._fail(f"Bad date: {exc}. Use YYYY-MM-DD.")


# ---------------------------------------------------------------- knowledge
class KnowledgeTool(Tool):
    name = "knowledge"
    description = (
        "Search the user's own documents (project notes, specs, plans) in their knowledge "
        "folder. Use it whenever they discuss one of their projects or ask about something "
        "they've written, before answering from general knowledge. action 'search' with a "
        "query, or 'list' to see which documents exist."
    )

    def __init__(self, kb):
        self.kb = kb

    @property
    def input_schema(self) -> dict:
        return {"type": "object",
                "properties": {"action": {"type": "string", "enum": ["search", "list"]},
                               "query": {"type": "string"}},
                "required": ["action"]}

    def execute(self, action: str = "search", query: str = "", **_) -> ToolResult:
        if action == "list":
            files = self.kb.files()
            return self._ok({"documents": files} if files else
                            {"documents": [], "note": "The knowledge folder is empty."})
        hits = self.kb.search(query)
        return self._ok({"matches": hits} if hits else
                        {"matches": [], "note": "Nothing in their documents matches that."})


# -------------------------------------------------------------------- think
THINK_SYSTEM = """You are the deep-reasoning engine behind a personal voice assistant. \
The assistant hands you a hard problem; you work it out properly and hand back a \
conclusion the assistant will put into a couple of spoken sentences.

How to work:
- Use everything in the user's context below. Their preferences, rules and routines \
outrank generic advice.
- For plans: lay out the steps and timings, check they actually fit together, and \
flag the one thing most likely to go wrong.
- For decisions: compare the real options on the criteria this user cares about \
(cost, time, comfort, their stated tastes), do the arithmetic, and pick one.
- For scenarios / what-ifs: consider the likely case, a bad case and a good case, \
with rough numbers, then say what to do.
- If a key fact is missing, make the most reasonable assumption, state it, and continue.
- Never invent live data (prices, traffic, showtimes, news). Say it's an estimate.

Reply in plain text, no markdown, at most about 180 words: first the conclusion / \
recommendation, then the key reasons and numbers, then any assumption you made.

{context}"""


class ThinkTool(Tool):
    name = "think"
    description = (
        "Your deep-reasoning mode. Call it for anything that needs real thought rather "
        "than a quick reply: planning a day, trip or project; weighing options or a "
        "decision; budgeting; 'what if' scenarios; debugging an idea; multi-step "
        "problems. Pass the full problem in 'problem' with every relevant detail from "
        "the conversation. It returns a considered answer for you to say briefly in your "
        "own voice. Don't use it for simple questions or chit-chat."
    )

    def __init__(self, llm_cfg, model: str, context: Callable[[], str],
                 max_tokens: int = 1200, llm_factory: Callable | None = None):
        self.llm_cfg = llm_cfg
        self.model = model or llm_cfg.model
        self.context = context
        self.max_tokens = max_tokens
        self._factory = llm_factory
        self._llm = None

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "problem": {"type": "string",
                            "description": "The full problem, with all details and "
                                           "constraints the user gave."},
                "kind": {"type": "string",
                         "enum": ["plan", "decision", "scenario", "analysis"]},
            },
            "required": ["problem"],
        }

    def _client(self):
        if self._llm is None:
            if self._factory is not None:
                self._llm = self._factory()
            else:
                from agent.llm import build_llm
                cfg = replace(self.llm_cfg, model=self.model,
                              max_tokens=self.max_tokens, temperature=0.3,
                              timeout_s=max(self.llm_cfg.timeout_s, 60.0))
                self._llm = build_llm(cfg)
            if hasattr(self._llm, "load"):
                self._llm.load()
        return self._llm

    def execute(self, problem: str = "", kind: str = "analysis", **_) -> ToolResult:
        if not (problem or "").strip():
            return self._fail("No problem given.")
        from agent.llm import LLMError
        system = THINK_SYSTEM.format(context=self.context())
        messages = [{"role": "user", "content": f"[{kind}] {problem.strip()}"}]
        try:
            resp = self._client().complete(system=system, messages=messages, tools=None)
        except LLMError as exc:
            return self._fail(f"Reasoning model unavailable: {exc}")
        text = (resp.text or "").strip()
        if not text:
            return self._fail("The reasoning model returned nothing.")
        return self._ok({"analysis": text, "model": self.model})
