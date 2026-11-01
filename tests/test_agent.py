"""Agent tests with a scripted fake LLM: the tool loop is what's under test,
not the model."""
import tempfile
from pathlib import Path

from agent.agent import Agent
from agent.llm import LLMError, LLMResponse, ToolCall
from agent.state import ConversationState
from config import LLMConfig, ToolsConfig
from tools.base import ToolRegistry
from tools.personal_data import PersonalDataTool


class FakeLLM:
    """Returns a queued list of responses, one per call."""
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def complete(self, system, messages, tools=None):
        self.calls.append({"messages": list(messages), "tools_offered": bool(tools)})
        if not self.script:
            return LLMResponse(text="done")
        nxt = self.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def build(script):
    tmp = Path(tempfile.mkdtemp()) / "a.db"
    tool = PersonalDataTool(ToolsConfig(), tmp)
    tool.add_expense("2020-01-01", "food", "old", 10.0)
    agent = Agent(LLMConfig(provider="anthropic", api_key="x"),
                  ToolRegistry([tool]), ConversationState())
    agent._llm = FakeLLM(script)
    return agent, tool


def test_plain_answer_needs_no_tools():
    agent, _ = build([LLMResponse(text="A transistor amplifies current.")])
    reply = agent.respond("what is a transistor")
    assert reply.ok and reply.tools_used == []


def test_tool_call_then_answer():
    agent, _ = build([
        LLMResponse(tool_calls=[ToolCall("c1", "query_expenses",
                                         {"query_kind": "total", "period": "all_time"})]),
        LLMResponse(text="You've spent ten rupees."),
    ])
    reply = agent.respond("how much have I spent")
    assert reply.tools_used == ["query_expenses"]
    assert "ten rupees" in reply.text


def test_tool_result_reaches_the_model():
    agent, _ = build([
        LLMResponse(tool_calls=[ToolCall("c1", "query_expenses", {"query_kind": "total"})]),
        LLMResponse(text="ok"),
    ])
    agent.respond("spend?")
    second_call = agent._llm.calls[1]["messages"]
    assert any(m["role"] == "tool" for m in second_call)


def test_unknown_tool_returns_an_error_not_a_crash():
    agent, _ = build([
        LLMResponse(tool_calls=[ToolCall("c1", "delete_everything", {})]),
        LLMResponse(text="I couldn't do that."),
    ])
    reply = agent.respond("delete my files")
    assert reply.ok


def test_llm_failure_produces_a_spoken_fallback():
    agent, _ = build([LLMError("connection refused")])
    reply = agent.respond("hello")
    assert not reply.ok
    assert reply.text and len(reply.text) < 120


def test_tool_loop_is_bounded():
    """A model that keeps requesting tools must be cut off, and the final
    round must be made with tools withheld."""
    forever = [LLMResponse(tool_calls=[ToolCall(f"c{i}", "query_expenses",
                                                {"query_kind": "total"})])
               for i in range(10)]
    agent, _ = build(forever)
    agent.cfg = LLMConfig(provider="anthropic", api_key="x", max_tool_rounds=2)
    agent.respond("loop")
    assert len(agent._llm.calls) <= 4
    assert agent._llm.calls[-1]["tools_offered"] is False


def test_context_carries_across_turns():
    agent, _ = build([LLMResponse(text="one"), LLMResponse(text="two")])
    agent.respond("first")
    agent.respond("second")
    assert any(m.get("content") == "first" for m in agent.state.messages())


def test_reset_clears_history():
    agent, _ = build([LLMResponse(text="hi")])
    agent.respond("hello")
    agent.reset()
    assert agent.state.messages() == []
