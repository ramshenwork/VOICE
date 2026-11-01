"""STT tests. The model is not loaded unless STT_RUN_MODEL=1, so the suite
stays fast; the guard behaviours are what matter here."""
import os

import numpy as np
import pytest

from config import STTConfig
from stt.engine import STTEngine, Transcription


def test_short_audio_returns_empty_not_an_error():
    """Half a second of room noise is normal, not an exception."""
    eng = STTEngine(STTConfig())
    out = eng.transcribe(np.zeros(1000, dtype=np.float32))
    assert out.is_empty and out.text == ""


def test_wrong_sample_rate_is_rejected_loudly():
    from stt.engine import STTError
    eng = STTEngine(STTConfig())
    eng._model = object()          # skip the real load
    with pytest.raises(STTError):
        eng.transcribe(np.zeros(48000, dtype=np.float32), sample_rate=48000)


def test_transcription_reports_no_fake_confidence():
    t = Transcription(text="hello")
    assert t.avg_logprob is None


@pytest.mark.skipif(os.getenv("STT_RUN_MODEL") != "1",
                    reason="set STT_RUN_MODEL=1 to load the real model")
def test_real_model_transcribes_a_tone_without_crashing():
    eng = STTEngine(STTConfig())
    eng.load()
    audio = (0.05 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.float32)
    assert isinstance(eng.transcribe(audio).text, str)
