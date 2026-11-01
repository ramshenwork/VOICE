"""Audio tests. Device tests are skipped automatically in headless CI."""
import numpy as np
import pytest

from audio.output import float_to_pcm16, pcm16_to_float, resample_linear
from config import AudioConfig


def test_audio_config_frame_matches_silero():
    cfg = AudioConfig()
    assert cfg.sample_rate == 16000
    assert cfg.chunk_size == 512, "Silero VAD requires 512-sample frames at 16 kHz"
    assert abs(cfg.chunk_ms - 32.0) < 0.1


def test_pcm_roundtrip_is_lossless_enough():
    original = np.array([0.0, 0.5, -0.5, 0.999, -0.999], dtype=np.float32)
    restored = pcm16_to_float(float_to_pcm16(original))
    assert np.allclose(original, restored, atol=1e-3)


def test_pcm_clips_instead_of_wrapping():
    loud = np.array([2.0, -2.0], dtype=np.float32)
    restored = pcm16_to_float(float_to_pcm16(loud))
    assert restored[0] > 0.9 and restored[1] < -0.9


def test_resample_preserves_duration():
    src = np.sin(np.linspace(0, 40, 24000)).astype(np.float32)
    out = resample_linear(src, 24000, 16000)
    assert abs(len(out) - 16000) <= 1


def test_resample_noop_when_rates_match():
    src = np.zeros(100, dtype=np.float32)
    assert resample_linear(src, 16000, 16000).shape == (100,)


@pytest.mark.skipif(True, reason="requires a real microphone; run manually")
def test_microphone_produces_chunks():
    from audio.input import MicrophoneStream, rms
    import itertools
    with MicrophoneStream(AudioConfig()) as mic:
        for chunk in itertools.islice(mic.chunks(), 10):
            assert chunk.dtype == np.float32
            assert chunk.shape[0] == 512
            assert 0.0 <= rms(chunk) <= 1.0
