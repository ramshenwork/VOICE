"""
Local voice agent -- microphone in, speaker out.

    python main.py              full voice loop ("Hey Jarvis" to wake)
    python main.py --text       typed input, spoken output (no mic needed)
    python main.py --devices    list audio devices
    python main.py --check      verify every component loads

Wake word
---------
While asleep only the tiny wake word model runs. "Hey Jarvis" wakes it: Spotify
is turned down, it says "Yes?", and it listens. After each reply it keeps
listening for FOLLOW_UP_WINDOW_S so you can carry on without the wake word,
then goes back to sleep and Spotify comes back up. "That's all" sends it to
sleep straight away. WAKE_WORD_ENABLED=false gives the old always-listening mode.

The main loop is single-threaded on purpose. Audio capture already runs on its
own OS thread and hands frames over a queue, so this thread is free to do VAD
and, when an utterance ends, the STT/agent/TTS work. During that work the
capture thread keeps buffering, so nothing is lost.

Barge-in
--------
The microphone is never closed while the agent speaks. While playback is
active the VAD keeps running with a raised threshold; if the user starts
talking, playback is cut immediately and the new utterance is captured.

Acoustic caveat, and it is a real one: on laptop speakers the microphone hears
the agent's own voice, and no threshold cleanly separates that from a human.
Use headphones, or set BARGE_IN=false. Solving it properly needs acoustic echo
cancellation, which is a project of its own.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time

import numpy as np

from agent.agent import AgentReply
from audio.ducking import SpotifyDucker
from audio.input import AudioInputError, MicrophoneStream, list_input_devices, rms
from audio.output import SpeakerStream
from config import load_config
from pipeline import Pipeline, setup_logging
from vad.detector import VADEvent
from wake.detector import AwakeSession, WakeWordDetector, WakeWordError

logger = logging.getLogger("main")

BARGE_IN = os.getenv("BARGE_IN", "true").lower() in ("1", "true", "yes")
# Raised threshold while the agent is speaking, to reduce self-triggering.
BARGE_IN_THRESHOLD = float(os.getenv("BARGE_IN_THRESHOLD", "0.85"))


def cmd_devices() -> int:
    print("\nInput devices:")
    for d in list_input_devices():
        print(f"  [{d['index']:2d}] {d['name']}  ({d['channels']}ch @ {d['default_samplerate']:.0f} Hz)")
    print("\nSet AUDIO_INPUT_DEVICE in .env to an index or name.\n")
    return 0


def cmd_check(pipeline: Pipeline) -> int:
    ok = True
    print("\nComponent check")
    print("-" * 46)
    for label, fn in [
        ("VAD (silero)", pipeline.vad.load),
        ("STT (faster-whisper)", pipeline.stt.load),
        ("TTS", pipeline.tts.load),
        ("LLM", pipeline.agent.load),
        ("Wake word", WakeWordDetector(pipeline.cfg.wake).load),
    ]:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"  ok    {label:<26} {time.perf_counter()-t0:5.2f}s")
        except Exception as exc:
            ok = False
            print(f"  FAIL  {label:<26} {exc}")

    try:
        devs = list_input_devices()
        print(f"  ok    {'Microphone':<26} {len(devs)} device(s)")
    except Exception as exc:
        ok = False
        print(f"  FAIL  {'Microphone':<26} {exc}")

    print(f"  ok    {'Tools':<26} {', '.join(pipeline.registry.names())}")
    print("-" * 46)
    print("Ready.\n" if ok else "Some components failed. See above.\n")
    return 0 if ok else 1


def cmd_text(pipeline: Pipeline, speak: bool = True) -> int:
    pipeline.agent.load()
    speaker = None
    if speak:
        pipeline.tts.load()
        try:
            speaker = SpeakerStream(pipeline.tts.sample_rate, pipeline.cfg.audio.output_device)
            speaker.start()
        except Exception as exc:
            logger.warning("no speaker (%s); text only", exc)
            speaker = None

    name = pipeline.cfg.assistant.name.lower()
    print(f"\nTyped mode with {pipeline.cfg.assistant.name}. Ctrl-C to quit.\n")
    try:
        while True:
            try:
                text = input("you  > ").strip()
            except EOFError:
                break
            if not text:
                continue
            if text.lower() in ("quit", "exit"):
                break
            print(f"{name}> ", end="", flush=True)
            reply = stream_reply(pipeline, speaker, text, echo=True)
            if reply and reply.tools_used:
                print(f"       (tools: {', '.join(reply.tools_used)})")
            if speaker:
                speaker.wait_until_done(timeout=60)
    except KeyboardInterrupt:
        pass
    finally:
        if speaker:
            speaker.close()
    return 0


def stream_reply(pipeline: Pipeline, speaker: SpeakerStream | None, text: str,
                 echo: bool = False) -> AgentReply | None:
    """Run one turn, speaking each sentence the moment the model finishes it."""
    reply = None
    t0 = time.perf_counter()
    first = True
    for item in pipeline.agent.respond_stream(text):
        if isinstance(item, AgentReply):
            reply = item
            continue
        if first:
            logger.info("latency: first sentence after %.0fms", (time.perf_counter() - t0) * 1000)
            first = False
        if echo:
            print(item, end=" ", flush=True)
        if speaker:
            for chunk in pipeline.tts.stream(item):
                speaker.play(chunk)
    if echo:
        print()
    if reply:
        logger.info("latency: agent=%.0fms tools=%s",
                    reply.latency_s * 1000, reply.tools_used or "none")
    return reply


SLEEP_PHRASES = re.compile(
    r"\b(that'?s all|that will be all|go to sleep|stop listening|never ?mind|good ?night)\b",
    re.IGNORECASE)


def _status(msg: str) -> None:
    print(f"\r  {msg:<44}", end="", flush=True)


def cmd_voice(pipeline: Pipeline) -> int:
    cfg = pipeline.cfg
    name = cfg.assistant.name
    pipeline.warmup()

    try:
        speaker = SpeakerStream(pipeline.tts.sample_rate, cfg.audio.output_device)
        speaker.start()
    except Exception as exc:
        logger.error("speaker unavailable: %s", exc)
        return 1

    # -- wake word: optional, and the loop degrades to always-listening ----
    wake = None
    if cfg.wake.enabled:
        wake = WakeWordDetector(cfg.wake)
        try:
            wake.load()
        except WakeWordError as exc:
            logger.warning("%s -- falling back to always-listening", exc)
            wake = None
    session = AwakeSession(cfg.wake.follow_up_s)
    if wake is None:
        session.wake()
        session.follow_up_s = float("inf")
    ducker = SpotifyDucker(enabled=bool(wake) and cfg.wake.duck_spotify,
                           duck_volume=cfg.wake.duck_volume)

    ack = None
    if wake and BARGE_IN and cfg.wake.acknowledgement:
        # Pre-rendered so answering the wake word costs nothing.
        ack = pipeline.tts.synthesize(cfg.wake.acknowledgement.replace("{address}",
                                                                       cfg.assistant.address))

    def go_to_sleep() -> None:
        session.sleep()
        pipeline.vad.reset()
        if wake:
            wake.reset()
        ducker.restore()
        _status(f"[ asleep -- say 'hey {name.lower()}' ]")

    awaiting_playback_end = False
    sleep_after_reply = False
    last_meter = 0.0

    print("\n" + "=" * 56)
    if wake:
        print(f"  {name} is asleep. Say 'Hey {name}' to wake it. Ctrl-C to quit.")
    else:
        print(f"  {name} is listening. Speak naturally. Ctrl-C to quit.")
    print(f"  barge-in: {'on' if BARGE_IN else 'off'}   voice: {cfg.tts.voice}"
          f"   spotify ducking: {'on' if ducker.enabled else 'off'}")
    print("=" * 56 + "\n")
    if wake:
        _status(f"[ asleep -- say 'hey {name.lower()}' ]")

    try:
        with MicrophoneStream(cfg.audio) as mic:
            for chunk in mic.chunks():
                playing = speaker.is_playing

                # ---- asleep: only the wake word model runs -------------
                if not session.awake:
                    if wake.process(chunk):
                        logger.info("wake word (score %.2f)", wake.last_score)
                        ducker.duck()
                        session.wake()
                        pipeline.vad.reset()
                        if ack is not None and ack.size:
                            speaker.play(ack)
                        _status("[ listening... ]")
                    continue

                # ---- awake: reply playback finished -> start window ----
                if awaiting_playback_end and not playing:
                    awaiting_playback_end = False
                    if sleep_after_reply:
                        sleep_after_reply = False
                        go_to_sleep()
                        continue
                    session.reply_finished()
                    _status("[ listening... ]" if wake else "")

                if wake and not playing and session.expired():
                    go_to_sleep()
                    continue

                if playing and not BARGE_IN:
                    continue  # deaf while speaking: no self-trigger, no barge-in

                for result in pipeline.vad.process(chunk):
                    if result.event is VADEvent.SPEECH_STARTED:
                        if playing:
                            if result.speech_probability < BARGE_IN_THRESHOLD:
                                continue
                            logger.info("barge-in: stopping playback")
                            speaker.stop()
                        session.user_speaking()
                        _status("[ listening... ]")

                    elif result.event is VADEvent.SPEECH_ENDED and result.segment is not None:
                        _status("[ thinking... ]")
                        text, reason = pipeline.transcribe_utterance(
                            result.segment, cfg.audio.sample_rate)
                        if text is None:
                            session.reply_finished()
                            _status("[ listening... ]" if wake else "")
                            continue
                        print(f"\r  you  : {text:<40}")
                        print(f"  {name.lower():<5}: ", end="", flush=True)
                        stream_reply(pipeline, speaker, text, echo=True)
                        print()
                        awaiting_playback_end = True
                        sleep_after_reply = bool(wake) and bool(SLEEP_PHRASES.search(text))

                # cheap level meter while idle in always-listening mode
                if not wake and not playing and time.time() - last_meter > 0.25:
                    level = min(20, int(rms(chunk) * 260))
                    if pipeline.vad.state.value == "idle":
                        bar = "#" * level + "-" * (20 - level)
                        _status(f"[{bar}] p={pipeline.vad.last_probability:.2f}")
                    last_meter = time.time()

    except AudioInputError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        ducker.restore()
        time.sleep(0.3)  # let the restore run before the process exits
        speaker.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Small local voice agent")
    parser.add_argument("--text", action="store_true", help="typed input instead of mic")
    parser.add_argument("--no-speak", action="store_true", help="with --text, skip TTS")
    parser.add_argument("--devices", action="store_true", help="list audio devices")
    parser.add_argument("--check", action="store_true", help="verify components load")
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg.log_level)

    if args.devices:
        return cmd_devices()

    pipeline = Pipeline(cfg)
    if args.check:
        return cmd_check(pipeline)
    if args.text:
        return cmd_text(pipeline, speak=not args.no_speak)
    return cmd_voice(pipeline)


if __name__ == "__main__":
    sys.exit(main())
