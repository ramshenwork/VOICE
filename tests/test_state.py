from agent.state import ConversationState


def test_trimming_never_orphans_a_tool_result():
    """A tool message whose originating assistant turn was trimmed away makes
    providers reject the whole conversation, so trimming must cut to a clean
    user turn."""
    s = ConversationState(max_turns=2)
    for i in range(20):
        s.add_user(f"q{i}")
        s.add_assistant("", tool_calls=[type("T", (), {"id": "c", "name": "t",
                                                       "arguments": {}})()])
        s.add_tool_result("c", "t", {"status": "ok"})
        s.add_assistant("a")
    assert s.messages()[0]["role"] == "user"


def test_history_is_capped():
    s = ConversationState(max_turns=3)
    for i in range(50):
        s.add_user(f"u{i}")
        s.add_assistant(f"a{i}")
    assert len(s.messages()) <= 3 * 3 + 2
