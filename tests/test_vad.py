"""VAD state machine tests. The neural model is replaced by a scripted stub so
the state machine itself is what gets tested, deterministically."""
import numpy as np

from config import VADConfig
from vad.detector import VADController, VADEvent, VADState


class ScriptedVAD:
    """Stands in for Silero: returns probabilities from a fixed list."""
    frame_size = 512

    def __init__(self, probs):
        self.probs = list(probs)
        self.i = 0

    def load(self): pass
    def reset(self): self.i = 0

    def probability(self, frame):
        p = self.probs[min(self.i, len(self.probs) - 1)]
        self.i += 1
        return p


def run(probs, cfg=None):
    cfg = cfg or VADConfig()
    ctrl = VADController(cfg, 16000, model=ScriptedVAD(probs))
    events = []
    audio = np.ones(512 * len(probs), dtype=np.float32) * 0.1
    for r in ctrl.process(audio):
        if r.event is not VADEvent.NONE:
            events.append(r)
    return ctrl, events


def test_silence_produces_no_events():
    ctrl, events = run([0.0] * 50)
    assert events == []
    assert ctrl.state is VADState.IDLE


def test_speech_then_silence_yields_one_utterance():
    # 32 ms/frame: 20 speech frames = 640 ms, 30 silent = 960 ms > min_silence
    ctrl, events = run([0.1] * 5 + [0.9] * 20 + [0.05] * 30)
    kinds = [e.event for e in events]
    assert kinds.count(VADEvent.SPEECH_STARTED) == 1
    assert kinds.count(VADEvent.SPEECH_ENDED) == 1
    ended = [e for e in events if e.event is VADEvent.SPEECH_ENDED][0]
    assert ended.segment is not None and ended.segment.size > 0


def test_short_blip_is_rejected_as_noise():
    """A 64 ms spike is below min_speech_ms and must not start an utterance."""
    ctrl, events = run([0.05] * 10 + [0.95] * 2 + [0.05] * 20)
    assert not any(e.event is VADEvent.SPEECH_STARTED for e in events)


def test_mid_sentence_pause_does_not_end_the_turn():
    """This is the whole reason min_silence_ms exists: a 320 ms gap between
    clauses must not be mistaken for end-of-turn."""
    probs = [0.9] * 15 + [0.05] * 10 + [0.9] * 15 + [0.05] * 30
    ctrl, events = run(probs)
    kinds = [e.event for e in events]
    assert kinds.count(VADEvent.SPEECH_STARTED) == 1
    assert kinds.count(VADEvent.SPEECH_ENDED) == 1


def test_preroll_is_prepended_to_segment():
    """The segment must be longer than the confirmed-speech frames alone,
    proving the pre-speech padding recovered the onset."""
    ctrl, events = run([0.0] * 20 + [0.9] * 20 + [0.0] * 30)
    ended = [e for e in events if e.event is VADEvent.SPEECH_ENDED][0]
    speech_only_samples = 20 * 512
    assert ended.segment.shape[0] > speech_only_samples


def test_hysteresis_does_not_oscillate():
    """Probabilities parked between the two thresholds must not flip state."""
    cfg = VADConfig()
    mid = (cfg.speech_threshold + cfg.silence_threshold) / 2
    ctrl, events = run([0.9] * 15 + [mid] * 25 + [0.0] * 30)
    kinds = [e.event for e in events]
    assert kinds.count(VADEvent.SPEECH_ENDED) == 1


def test_max_utterance_forces_a_cut():
    cfg = VADConfig(max_utterance_ms=500)
    ctrl, events = run([0.95] * 60, cfg)
    assert any(e.event is VADEvent.SPEECH_ENDED for e in events)


def test_arbitrary_chunk_sizes_are_reframed():
    ctrl = VADController(VADConfig(), 16000, model=ScriptedVAD([0.0] * 100))
    results = ctrl.process(np.zeros(700, dtype=np.float32))
    assert len(results) == 1          # 700 -> one 512 frame, 188 carried over
    results = ctrl.process(np.zeros(400, dtype=np.float32))
    assert len(results) == 1          # 188 + 400 = 588 -> one more frame
