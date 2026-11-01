from processor.input_processor import InputProcessor


def p(text):
    return InputProcessor().process(text)


def test_empty_transcript_is_rejected():
    assert p("").should_respond is False
    assert p("   ").reason == "empty_or_too_short"


def test_filler_words_are_removed():
    assert "uh" not in p("uh can you tell me the weather").text.lower().split()


def test_repeated_words_collapse():
    assert p("what what what is the time").text.lower().startswith("what is")


def test_bracketed_annotations_are_stripped():
    assert "MUSIC" not in p("[MUSIC] what's the weather").text


def test_whisper_hallucinations_on_silence_are_discarded():
    for junk in ["Thank you.", "you", "Thanks for watching!", "..."]:
        assert p(junk).should_respond is False, junk


def test_normal_speech_passes_through():
    out = p("how much did i spend this month")
    assert out.should_respond
    assert out.text == "How much did i spend this month"


def test_space_before_punctuation_is_fixed():
    assert p("hello , world .").text == "Hello, world."


def test_wake_word_gating():
    proc = InputProcessor(wake_word="atlas", require_wake_word=True)
    assert proc.process("what time is it").should_respond is False
    out = proc.process("atlas what time is it")
    assert out.should_respond and "atlas" not in out.text.lower()
