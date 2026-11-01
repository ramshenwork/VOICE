from tts.engine import split_for_speech


def test_empty_text_produces_nothing():
    assert split_for_speech("") == []
    assert split_for_speech("   ") == []


def test_single_sentence_stays_whole():
    assert split_for_speech("It's twenty nine degrees.") == ["It's twenty nine degrees."]


def test_sentences_split_so_playback_can_start_early():
    out = split_for_speech("It's warm today. Rain is likely tomorrow. Bring an umbrella.")
    assert len(out) == 3


def test_decimals_do_not_split():
    assert len(split_for_speech("You spent 8420.50 rupees today.")) == 1


def test_abbreviations_do_not_split():
    assert len(split_for_speech("Dr. Rao called about the invoice.")) == 1


def test_very_long_clause_is_broken_at_a_comma():
    long = "You spent money on " + ", ".join(f"item {i}" for i in range(60)) + "."
    out = split_for_speech(long, max_chars=120)
    assert len(out) > 1
    assert all(len(c) <= 140 for c in out)


def test_nothing_is_lost_in_splitting():
    text = "One. Two! Three? Four."
    joined = " ".join(split_for_speech(text))
    for word in ["One", "Two", "Three", "Four"]:
        assert word in joined
