"""Wake session timing, Spotify ducking, and wake-word stripping -- no audio,
no Spotify, no model."""
from audio.ducking import SpotifyDucker
from processor.input_processor import InputProcessor
from wake.detector import AwakeSession


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_session_sleeps_after_follow_up_window():
    clock = Clock()
    s = AwakeSession(follow_up_s=8, clock=clock)
    s.wake()
    s.reply_finished()
    clock.t = 7.9
    assert not s.expired()
    clock.t = 8.1
    assert s.expired()


def test_session_never_expires_while_user_is_speaking():
    clock = Clock()
    s = AwakeSession(follow_up_s=2, clock=clock)
    s.wake()
    s.user_speaking()
    clock.t = 100
    assert not s.expired()


def test_asleep_session_does_not_expire():
    s = AwakeSession(follow_up_s=1, clock=Clock())
    assert not s.awake and not s.expired()


class FakeSpotify:
    def __init__(self, volume="70"):
        self.volume = volume
        self.scripts = []

    def __call__(self, script):
        self.scripts.append(script)
        if "set sound volume to" in script:
            self.volume = script.rsplit(" ", 1)[1]
            return ""
        return self.volume


def test_duck_then_restore_returns_to_original_volume():
    sp = FakeSpotify("70")
    d = SpotifyDucker(duck_volume=15, runner=sp, background=False)
    d.duck()
    assert sp.volume == "15"
    d.duck()  # second duck must not overwrite the saved level
    d.restore()
    assert sp.volume == "70"


def test_duck_is_a_noop_when_spotify_is_not_running():
    sp = FakeSpotify("")
    d = SpotifyDucker(runner=sp, background=False)
    d.duck()
    d.restore()
    assert not any("set sound volume" in s for s in sp.scripts)


def test_background_duck_and_restore_stay_in_order():
    import time
    sp = FakeSpotify("60")
    d = SpotifyDucker(duck_volume=10, runner=sp, background=True)
    d.duck()
    d.restore()
    for _ in range(50):
        if sp.volume == "60" and len(sp.scripts) >= 3:
            break
        time.sleep(0.01)
    assert sp.volume == "60"


def test_wake_word_is_stripped_from_the_request():
    p = InputProcessor(wake_word="Jarvis")
    assert p.process("Hey Jarvis, what's the time?").text == "What's the time?"
    assert p.process("Tell Jarvis hello").text == "Tell Jarvis hello"
