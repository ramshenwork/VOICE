# Riya's voice sample

Put a 10-30 second recording here as `sample.wav` (or set TTS_CLONE_SAMPLE):
one person speaking naturally, no music or background noise, in the tone you
want her to have. Use your own voice or someone who has agreed to it.

Then set `TTS_ENGINE=clone` in `.env`. Everything in this folder except this
README is git-ignored -- a voice sample is personal data.

## One-time setup

The cloner (Chatterbox) needs different library versions from the rest of
Riya, so it gets its own environment:

    /opt/homebrew/bin/python3.12 -m venv .venv-voice
    .venv-voice/bin/pip install chatterbox-tts soundfile "setuptools<81"

Riya starts and stops the voice server herself. The first start downloads the
model (a few minutes). If anything goes wrong she falls back to her normal
Kokoro voice. Server log: logs/clone_tts.log.

Speed on an M-series Mac: about 1.5-2 s per sentence with the default
`turbo` model, versus about 0.3 s for Kokoro.
