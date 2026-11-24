"""Superbrain tools, profile, rules and knowledge base -- no network, no model."""
import tempfile
from datetime import date
from pathlib import Path

import pytest

from agent.llm import LLMError, LLMResponse
from agent.prompts import build_system_prompt
from brain.knowledge import KnowledgeBase
from brain.profile import Profile
from config import AssistantConfig, LLMConfig, ToolsConfig
from memory.store import MemoryStore
from tools.brain import CalculatorTool, DatesTool, KnowledgeTool, ThinkTool, safe_eval
from tools.memory import PreferencesTool


def tmp() -> Path:
    return Path(tempfile.mkdtemp())


# ------------------------------------------------------------------ calculate
def test_calculator_exact_maths():
    c = CalculatorTool()
    assert c.execute(expression="1840 * 0.18").data["value"] == 331.2
    assert c.execute(expression="2^10").data["value"] == 1024
    assert c.execute(expression="1,840 + 160").data["value"] == 2000
    multi = c.execute(expression="10/4; round(sqrt(2), 3)").data
    assert [m["value"] for m in multi] == [2.5, 1.414]


@pytest.mark.parametrize("bad", [
    "__import__('os').system('ls')", "open('x')", "(1).__class__", "9**999999",
    "a + 1", "[1,2]",
])
def test_calculator_rejects_anything_but_maths(bad):
    with pytest.raises(Exception):
        safe_eval(bad)


def test_calculator_reports_division_by_zero():
    r = CalculatorTool().execute(expression="5/0")
    assert not r.ok and "zero" in r.error


# ---------------------------------------------------------------------- dates
def test_dates():
    d = DatesTool()
    assert d.execute(action="weekday", date="2026-11-24").data["weekday"] == "Tuesday"
    assert d.execute(action="add", date="2026-11-24", days=10).data["date"] == "2026-12-04"
    assert d.execute(action="between", date="2026-11-24",
                     other_date="2026-12-25").data["days"] == 31
    assert d.execute(action="next_weekday", date="2026-11-24",
                     weekday="friday").data["date"] == "2026-11-27"
    # next tuesday from a tuesday is a week later, not today
    assert d.execute(action="next_weekday", date="2026-11-24",
                     weekday="Tuesday").data["date"] == "2026-12-01"
    assert not d.execute(action="add", date="24/11/2026", days=1).ok


# ---------------------------------------------------------------------- rules
def test_preferences_add_list_remove():
    store = MemoryStore(tmp() / "a.db")
    t = PreferencesTool(store)
    assert t.execute(action="add", rule="Keep replies short").ok
    assert t.execute(action="add", rule="keep replies short").data["duplicate"]
    assert [r["rule"] for r in store.rules()] == ["Keep replies short"]
    assert t.execute(action="remove", query="short").data["removed"] == 1
    assert store.rules() == []
    assert not t.execute(action="remove", query="nothing").ok


# -------------------------------------------------------------------- profile
def test_profile_reloads_on_change_and_strips_comments():
    path = tmp() / "profile.md"
    p = Profile(path)
    assert p.text() == ""
    path.write_text("# Me\n<!-- guidance -->\n- Loves biryani\n\n\n")
    assert p.text() == "# Me\n- Loves biryani"
    import os, time
    path.write_text("# Me\n- Vegetarian now\n")
    os.utime(path, (time.time() + 5, time.time() + 5))
    assert "Vegetarian" in p.text()


def test_prompt_includes_profile_and_rules():
    prompt = build_system_prompt(AssistantConfig(), ToolsConfig(), [],
                                 profile="- Loves biryani",
                                 rules=[{"id": 1, "rule": "Never suggest Uber"}])
    assert "Loves biryani" in prompt and "Never suggest Uber" in prompt


# ------------------------------------------------------------------ knowledge
def test_knowledge_search_ranks_relevant_chunk_first():
    folder = tmp()
    (folder / "helmet.md").write_text(
        "Smart helmet project.\n\nThe helmet uses a Raspberry Pi camera and a HUD.\n\n"
        "Battery target is six hours of riding.")
    (folder / "garden.txt").write_text("Tomatoes need sun and water every day.")
    (folder / "image.png").write_bytes(b"\x89PNG")
    kb = KnowledgeBase(folder)
    hits = kb.search("what battery life does the helmet need?")
    assert hits and hits[0]["source"] == "helmet.md"
    assert kb.files() == ["garden.txt", "helmet.md"]
    (folder / "garden.txt").unlink()
    assert kb.files() == ["helmet.md"]


def test_knowledge_tool_empty_folder():
    t = KnowledgeTool(KnowledgeBase(tmp() / "missing"))
    assert t.execute(action="list").data["documents"] == []
    assert t.execute(action="search", query="helmet").data["matches"] == []


# ---------------------------------------------------------------------- think
class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.seen = None

    def complete(self, system, messages, tools=None):
        self.seen = (system, messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return LLMResponse(text=self.reply)


def test_think_passes_user_context_and_returns_analysis():
    fake = FakeLLM("Leave at 6:10.")
    t = ThinkTool(LLMConfig(api_key="x"), "big-model",
                  context=lambda: "User profile:\n- Hates traffic", llm_factory=lambda: fake)
    r = t.execute(problem="When should I leave?", kind="plan")
    assert r.ok and r.data["analysis"] == "Leave at 6:10."
    assert "Hates traffic" in fake.seen[0]
    assert fake.seen[1][0]["content"].startswith("[plan]")


def test_think_failure_is_reported_not_invented():
    t = ThinkTool(LLMConfig(api_key="x"), "", context=lambda: "",
                  llm_factory=lambda: FakeLLM(LLMError("down")))
    assert not t.execute(problem="x").ok
    assert not t.execute(problem="").ok
