"""Streaming replies: sentence buffering, and the agent yielding sentences
before the full reply is done."""
import tempfile
from pathlib import Path

from agent.agent import Agent, AgentReply
from agent.llm import BaseLLM, LLMError, LLMResponse, ToolCall
from agent.state import ConversationState
from config import LLMConfig, ToolsConfig
from tools.base import ToolRegistry
from tools.personal_data import PersonalDataTool
from tts.engine import SentenceBuffer


def test_sentence_buffer_emits_complete_sentences_early():
    buf = SentenceBuffer()
    assert buf.feed("Good evening") == []
    assert buf.feed(", sir. Traffic is") == ["Good evening, sir."]
    assert buf.feed(" light.") == []
    assert buf.flush() == ["Traffic is light."]


def test_sentence_buffer_does_not_split_on_decimals_or_abbreviations():
    buf = SentenceBuffer()
    out = buf.feed("It is 22.5 degrees. Dr. Rao called. ")
    out += buf.flush()
    assert out == ["It is 22.5 degrees.", "Dr. Rao called."]


class StreamingFake(BaseLLM):
    """Streams each scripted response word by word."""
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def stream(self, system, messages, tools=None):
        self.calls += 1
        nxt = self.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        for word in nxt.text.split(" "):
            if word:
                yield word + " "
        yield nxt


def agent_with(script, on_turn=None):
    tool = PersonalDataTool(ToolsConfig(), Path(tempfile.mkdtemp()) / "a.db")
    agent = Agent(LLMConfig(provider="openai", api_key="x"), ToolRegistry([tool]),
                  ConversationState(), system_prompt=lambda: "dynamic", on_turn=on_turn)
    agent._llm = StreamingFake(script)
    return agent


def test_agent_streams_sentences_then_reply():
    agent = agent_with([LLMResponse(text="Right away. It's done.")])
    items = list(agent.respond_stream("go"))
    assert items[:2] == ["Right away.", "It's done."]
    assert isinstance(items[-1], AgentReply) and items[-1].ok


def test_preamble_before_tool_call_is_spoken():
    agent = agent_with([
        LLMResponse(text="Let me check.",
                    tool_calls=[ToolCall("c1", "query_expenses", {"query_kind": "total"})]),
        LLMResponse(text="Nothing spent yet."),
    ])
    items = list(agent.respond_stream("spend?"))
    assert items[0] == "Let me check."
    assert "Nothing spent yet." in items
    assert items[-1].tools_used == ["query_expenses"]


def test_stream_failure_speaks_fallback():
    agent = agent_with([LLMError("down")])
    items = list(agent.respond_stream("hi"))
    assert isinstance(items[0], str) and not items[-1].ok


def test_turns_are_persisted():
    log = []
    agent = agent_with([LLMResponse(text="Hello.")], on_turn=lambda r, t: log.append(r))
    agent.respond("hi")
    assert log == ["user", "assistant"]


def test_restore_seeds_history():
    agent = agent_with([LLMResponse(text="Yes.")])
    agent.restore([{"role": "user", "content": "earlier"},
                   {"role": "assistant", "content": "noted"}])
    agent.respond("and now?")
    assert agent.state.messages()[0]["content"] == "earlier"
