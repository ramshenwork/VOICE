"""Memory, notes and expense-logging tools, against a throwaway database."""
import tempfile
from datetime import date
from pathlib import Path

from config import AssistantConfig, ToolsConfig
from agent.prompts import build_system_prompt
from memory.store import MemoryStore
from tools.memory import MemoryTool, NotesTool
from tools.personal_data import LogExpenseTool, PersonalDataTool


def store():
    return MemoryStore(Path(tempfile.mkdtemp()) / "m.db")


def test_remember_and_recall():
    s = store()
    tool = MemoryTool(s)
    assert tool.execute(action="remember", fact="User's office is in Gachibowli",
                        category="place").ok
    found = tool.execute(action="recall", query="where is the office")
    assert found.ok and "Gachibowli" in found.data["matches"][0]["fact"]


def test_duplicate_memory_is_not_stored_twice():
    s = store()
    s.remember("User likes filter coffee")
    s.remember("user likes filter coffee")
    assert len(s.memories()) == 1


def test_forget():
    s = store()
    s.remember("User is building a smart helmet")
    assert MemoryTool(s).execute(action="forget", query="helmet").data["forgotten"] == 1
    assert s.memories() == []


def test_remember_rejects_empty_fact():
    assert not MemoryTool(store()).execute(action="remember", fact="  ").ok


def test_notes_add_search_list_delete():
    tool = NotesTool(store())
    saved = tool.execute(action="add", body="Use a bone conduction speaker in the helmet",
                         title="Helmet audio", tags="helmet, ideas")
    assert saved.ok
    nid = saved.data["saved"]["id"]
    assert tool.execute(action="search", query="what did I note about the helmet").data["notes"]
    assert tool.execute(action="search", query="groceries").data["notes"] == []
    assert len(tool.execute(action="list").data["notes"]) == 1
    assert tool.execute(action="delete", note_id=nid).ok
    assert tool.execute(action="list").data["notes"] == []


def test_note_search_is_injection_safe():
    tool = NotesTool(store())
    tool.execute(action="add", body="hello")
    r = tool.execute(action="search", query="'; DROP TABLE notes; --")
    assert r.ok
    assert len(tool.execute(action="list").data["notes"]) == 1


def test_conversation_log_restores_from_a_user_turn():
    s = store()
    s.log_turn("assistant", "orphan reply")
    s.log_turn("user", "hi")
    s.log_turn("assistant", "hello")
    turns = s.recent_turns(3)
    assert turns[0]["role"] == "user" and len(turns) == 2


def test_memories_reach_the_system_prompt():
    s = store()
    s.remember("User's sister is called Priya")
    prompt = build_system_prompt(AssistantConfig(name="Jarvis", user_name="Ram"),
                                 ToolsConfig(), s.memories())
    assert "Priya" in prompt and "Jarvis" in prompt and '"Ram"' in prompt


def test_address_defaults_to_sir():
    assert AssistantConfig(user_name="").address == "sir"


def _expenses():
    data = PersonalDataTool(ToolsConfig(), Path(tempfile.mkdtemp()) / "e.db")
    return data, LogExpenseTool(data)


def test_log_expense_defaults_to_today_and_is_queryable():
    data, log = _expenses()
    r = log.execute(amount=200, category="Transport", description="auto")
    assert r.ok and r.data["logged"]["date"] == date.today().isoformat()
    total = data.execute(query_kind="total", period="today", category="transport")
    assert total.data["total"] == 200


def test_log_expense_rejects_bad_input():
    _, log = _expenses()
    assert not log.execute(amount=-5, category="food").ok
    assert not log.execute(amount="lots", category="food").ok
    assert not log.execute(amount=10, category="food", date="tomorrow-ish").ok
