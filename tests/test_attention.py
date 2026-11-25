"""Echo guard, attention gate and quiet requests -- no audio, no network."""
from wake.attention import STOP_PHRASES, AttentionGate, EchoGuard, quiet_request
from wake.detector import AwakeSession


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_echo_guard_drops_her_own_words():
    clock = Clock()
    g = EchoGuard(clock=clock)
    g.spoke("No worries, sir. Just let me know what you want to talk about or do next.")
    assert g.is_echo("No worries sir. Just let me know what you want to talk about or do next.")
    assert g.is_echo("let me know what you want")          # partial pickup
    assert not g.is_echo("What's the weather tomorrow?")
    clock.t = 60
    assert not g.is_echo("Just let me know what you want to talk about")  # long forgotten


def test_echo_tail():
    clock = Clock()
    g = EchoGuard(tail_s=0.8, clock=clock)
    g.playback_ended()
    clock.t = 0.5
    assert g.in_tail()
    clock.t = 1.0
    assert not g.in_tail()


def test_quiet_requests():
    for s in ["I'm on a call", "Riya, not now", "shut up", "That's all, thanks",
              "be quiet", "I wasn't talking to you"]:
        assert quiet_request(s), s
    for s in ["How do I stop the server from crashing every night?",
              "What's on my calendar", ""]:
        assert not quiet_request(s), s


def test_stop_phrases_need_to_lead():
    assert STOP_PHRASES.search("Stop.")
    assert STOP_PHRASES.search("wait, that's wrong")
    assert not STOP_PHRASES.search("the bus didn't stop")


def gate(answer="NO"):
    calls = []

    def classify(prompt):
        calls.append(prompt)
        return answer
    return AttentionGate("Riya", ["ria", "rhea"], classify), calls


def test_gate_name_always_counts_without_llm():
    g, calls = gate("NO")
    assert g.addressed("Rhea, what about tomorrow?")
    assert calls == []


def test_gate_short_answer_to_her_question():
    g, calls = gate("NO")
    g.last_reply = "Want me to note that down?"
    assert g.addressed("yes please")
    assert calls == []


def test_gate_phone_call_is_ignored_and_she_sleeps():
    g, calls = gate("NO")
    g.last_reply = "Twenty-six and clear."
    assert not g.addressed("Yeah I'll be there by six, just send me the location")
    assert not g.should_sleep
    assert not g.addressed("Okay okay, bye")
    assert g.should_sleep
    assert "Twenty-six and clear." in calls[0]


def test_gate_follow_up_allowed_by_classifier():
    g, _ = gate("YES")
    g.last_reply = "Twenty-six and clear."
    assert g.addressed("and what about tomorrow morning")
    assert g.ignored_in_a_row == 0


def test_gate_classifier_failure_means_silence():
    def broken(prompt):
        raise RuntimeError("network down")
    g = AttentionGate("Riya", [], broken)
    assert not g.addressed("so anyway I told him the meeting moved")


def test_ignored_speech_does_not_extend_the_window():
    clock = Clock()
    s = AwakeSession(follow_up_s=8, clock=clock)
    s.wake()
    s.reply_finished()            # window ends at 8
    clock.t = 5
    s.user_speaking()             # background talk starts
    clock.t = 7
    s.resume()                    # ...and wasn't for her
    clock.t = 8.1
    assert s.expired()
