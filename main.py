"""
Local voice agent -- microphone in, speaker out.

    python main.py              full voice loop
    python main.py --text       typed input, spoken output (no mic needed)
    python main.py --devices    list audio devices
    python main.py --check      verify every component loads

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
import sys
import time

import numpy as np

from audio.input import AudioInputError, MicrophoneStream, list_input_devices, rms
from audio.output import SpeakerStream
from config import load_config
from pipeline import Pipeline, setup_logging
from vad.detector import VADEvent

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

    print("\nTyped mode. Ctrl-C to quit.\n")
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
            reply = pipeline.agent.respond(text)
            print(f"agent> {reply.text}")
            if reply.tools_used:
                print(f"       (tools: {', '.join(reply.tools_used)})")
            if speaker:
                for chunk in pipeline.tts.stream(reply.text):
                    speaker.play(chunk)
                speaker.wait_until_done(timeout=30)
    except KeyboardInterrupt:
        pass
    finally:
        if speaker:
            speaker.close()
    return 0


def speak(pipeline: Pipeline, speaker: SpeakerStream, text: str) -> None:
    """Stream TTS to the speaker sentence by sentence."""
    for chunk in pipeline.tts.stream(text):
        speaker.play(chunk)


def cmd_voice(pipeline: Pipeline) -> int:
    cfg = pipeline.cfg
    pipeline.warmup()

    try:
        speaker = SpeakerStream(pipeline.tts.sample_rate, cfg.audio.output_device)
        speaker.start()
    except Exception as exc:
        logger.error("speaker unavailable: %s", exc)
        return 1

    base_threshold = pipeline.vad.cfg.speech_threshold
    last_meter = 0.0

    print("\n" + "=" * 52)
    print("  Listening. Speak naturally. Ctrl-C to quit.")
    print(f"  barge-in: {'on' if BARGE_IN else 'off'}   tts: {pipeline.tts.backend}")
    print("=" * 52 + "\n")

    try:
        with MicrophoneStream(cfg.audio) as mic:
            for chunk in mic.chunks():
                playing = speaker.is_playing

                if playing and not BARGE_IN:
                    continue  # deaf while speaking: no self-trigger, no barge-in

                for result in pipeline.vad.process(chunk):
                    if result.event is VADEvent.SPEECH_STARTED:
                        if playing:
                            if result.speech_probability < BARGE_IN_THRESHOLD:
                                continue
                            logger.info("barge-in: stopping playback")
                            speaker.stop()
                        print("\r  [ listening... ]           ", end="", flush=True)

                    elif result.event is VADEvent.SPEECH_ENDED and result.segment is not None:
                        print("\r  [ thinking...  ]           ", end="", flush=True)
                        reply, transcript = pipeline.handle_utterance(
                            result.segment, cfg.audio.sample_rate)
                        if reply is None:
                            print("\r" + " " * 30 + "\r", end="", flush=True)
                            continue
                        print(f"\r  you  : {transcript}")
                        print(f"  agent: {reply.text}\n")
                        speak(pipeline, speaker, reply.text)

                # cheap level meter while idle
                if not playing and time.time() - last_meter > 0.25:
                    level = min(20, int(rms(chunk) * 260))
                    if pipeline.vad.state.value == "idle":
                        bar = "#" * level + "-" * (20 - level)
                        print(f"\r  [{bar}] p={pipeline.vad.last_probability:.2f}",
                              end="", flush=True)
                    last_meter = time.time()

    except AudioInputError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
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
