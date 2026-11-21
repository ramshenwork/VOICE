# R.I.Y.A. -- Ready If You Ask

A personal local voice assistant. You say "Hey Riya". She turns Spotify down,
listens, works out when you've finished, understands you, looks things up if
she needs to, and talks back -- with a personality, and a memory of who you
are. Runs on your laptop.

## What it can do now

| Ask it | What happens |
|---|---|
| "Hey Riya" | wakes up, ducks Spotify, says "Yes, sir?" |
| "Hey Riya, what's the weather?" | wakes and answers in one go |
| "Remember my office is in Gachibowli" | stored as a fact it knows from then on, across restarts |
| "Note this down: ..." / "What did I note about the helmet?" | its own notebook |
| "I just spent 180 on an auto" | logs the expense |
| "How much did I spend on food this week?" | reads your expenses |
| "What's the weather?" | live weather |
| anything else | conversation, opinions, discussing your projects |
| "That's all, Riya" | goes back to sleep, Spotify comes back up |

After each reply it keeps listening for `FOLLOW_UP_WINDOW_S` (8 s) so you can
carry on without saying the wake word again. Replies are streamed: it starts
speaking the first sentence while the model is still writing the rest.

Personalise her in `.env`: `ASSISTANT_NAME`, `USER_NAME` (empty = "sir"),
`TTS_VOICE`.

**About the wake word.** No ready-made model exists for "Riya", so by default
(`WAKE_WORD_ENGINE=transcript`) she wakes when the speech recogniser hears an
utterance that *starts* with her name. That works for any name with no
training, but uses more CPU while asleep than a dedicated model, and loud
music through laptop speakers can keep the recogniser busy -- headphones fix
both. A dedicated "hey riya" openWakeWord model can be trained later for
near-zero idle CPU. Memories, notes, expenses and the conversation log all live in
`data/agent.db` on your machine.

---

## 1. What this actually is, in plain terms

Think of it as an assembly line with eight stations. A sound goes in one end
and a sentence comes out the other, and each station does exactly one job and
hands its work to the next.

1. **The ear** (`audio/input.py`) holds a microphone open and hands over the
   sound in tiny 32-millisecond slices. It has no idea what sound *is*.
2. **The doorman** (`vad/detector.py`) watches those slices and answers one
   question: *is a person talking right now, and have they stopped?* It knows
   nothing about words. It just decides where a sentence starts and ends.
3. **The typist** (`stt/engine.py`) gets handed a complete sentence of audio
   and writes down the words. It never decides when you've finished — that's
   the doorman's job, and keeping them separate is the single most important
   structural decision in the project.
4. **The proofreader** (`processor/input_processor.py`) tidies the typist's
   work: drops the "um"s, throws away the nonsense the typist invents when
   handed silence.
5. **The thinker** (`agent/agent.py`) reads the sentence and decides what to
   do. It has no microphone, no speaker, no idea any of that exists. Text in,
   text out.
6. **The specialists** (`tools/`) are the only things allowed to fetch real
   facts. One phones the weather service. One reads your expense file. The
   thinker can *ask* for them by name but can never run anything else.
7. **The voice** (`tts/engine.py`) turns the answer into sound, one sentence
   at a time so you start hearing it before the whole thing is ready.
8. **The mouth** (`audio/output.py`) plays it, and — crucially — can be shut
   up mid-word the instant you start talking again.

The whole design rests on one rule: **no station knows how any other station
works.** Swap the typist for a different model and nobody else notices.

---

## 2. Get it running

```bash
cd voice_agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# open .env, set LLM_API_KEY

python data/seed.py        # sample expenses so it has something to talk about
                           # (--reset replaces them with fresh ones dated around today)
python main.py --check     # verify every model loads
python main.py             # talk to it
```

If PortAudio is missing: `brew install portaudio` then reinstall
`sounddevice`.

### The four ways to run it

| Command | What it does |
|---|---|
| `python main.py` | Full voice loop: "Hey Riya" to wake, mic in, speaker out |
| `python main.py --text` | Type, hear the answer spoken. No mic needed |
| `python main.py --check` | Load every component and report timings |
| `python main.py --devices` | List microphones, so you can pin one in `.env` |
| `uvicorn server.app:app --port 8000` | Web console at `http://localhost:8000` |

---

## 3. The two runtimes

The same pipeline, driven from two places.

**Local** (`main.py`) — the laptop microphone is the input device. Lowest
latency, fully private, supports true barge-in.

**Web** (`server/app.py` + `web/index.html`) — the *browser* is the audio
device. Your mic audio is resampled to 16 kHz in the browser, sent over a
WebSocket as PCM16, and the server runs VAD/STT/agent/TTS and sends WAV
frames back.

```
browser mic ─► downsample 48k→16k ─► PCM16 ─► WebSocket
                                                  │
              ┌───────────────────────────────────┘
              ▼
    VAD ─► STT ─► agent ─► tools ─► TTS
              │
              └──► WAV frames ─► WebSocket ─► browser playback
```

This exists because **a server has no microphone.** That's the reason the web
mode had to be built at all.

---

## 4. Full data flow

```
                microphone (or browser)
                          │  float32 mono @ 16 kHz
                          ▼
        ┌─────────────────────────────────┐
        │  audio/input.py                 │  OS audio thread → queue
        │  512-sample chunks (32 ms)      │  never blocks
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │  vad/detector.py                │  Silero + state machine
        │  IDLE → MAYBE_SPEECH → SPEAKING │  hysteresis + duration gates
        │       → MAYBE_END → IDLE        │  pre-roll ring buffer
        └────────────────┬────────────────┘
                         │  complete utterance, padded both ends
                         ▼
        ┌─────────────────────────────────┐
        │  stt/engine.py                  │  faster-whisper
        │  → Transcription(text, …)       │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │  processor/input_processor.py   │  clean, or discard as noise
        │  → ProcessedInput(should_respond)│
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │  agent/agent.py                 │  ◄──► agent/state.py
        │  loop, max_tool_rounds times:   │
        │    llm.complete(sys, msgs, tools)│
        │    if tool_calls → registry.execute
        │    else → done                  │
        └────────────────┬────────────────┘
                         │  agent/llm.py normalises Anthropic
                         │  / OpenAI / Ollama into one shape
                         ▼
        ┌─────────────────────────────────┐
        │  tools/base.py ToolRegistry     │  fixed dict lookup by name
        │    get_weather   → Open-Meteo   │  no eval, no shell
        │    query_expenses → SQLite      │  bound params only
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │  tts/engine.py                  │  Kokoro, per sentence
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │  audio/output.py                │  queued, interruptible
        └────────────────┬────────────────┘
                         ▼
                      speaker
```

---

## 5. Module contracts

Everything another developer (or model) needs to extend this without reading
the source.

| Module | Public interface | Returns |
|---|---|---|
| `audio/input.py` | `MicrophoneStream(cfg).chunks()` | iterator of `float32[512]` |
| `audio/output.py` | `SpeakerStream(rate).play(a)` / `.stop()` | – |
| `vad/detector.py` | `VADController(cfg, rate).process(chunk)` | `list[VADResult]` |
| `stt/engine.py` | `STTEngine(cfg).transcribe(audio, rate)` | `Transcription` |
| `processor/…` | `InputProcessor().process(t)` | `ProcessedInput` |
| `agent/agent.py` | `Agent(cfg, registry).respond(text)` | `AgentReply` |
| `agent/llm.py` | `llm.complete(system, messages, tools)` | `LLMResponse` |
| `tools/base.py` | `registry.execute(name, args)` | `ToolResult` |
| `tts/engine.py` | `TTSEngine(cfg).stream(text)` | iterator of `float32[]` |

Internal message shape used everywhere above the provider layer:

```python
{"role": "user"|"assistant", "content": str}
{"role": "assistant", "tool_calls": [ToolCall(id, name, arguments)]}
{"role": "tool", "tool_call_id": str, "name": str, "content": dict}
```

### Swapping a model

- **STT** → rewrite `STTEngine.transcribe` to return a `Transcription`. Done.
- **TTS** → rewrite `TTSEngine._synth_one` to return float32 at
  `self.sample_rate`. Done.
- **LLM** → add a class with `complete()` and register it in `build_llm`.
- **VAD** → any object with `.frame_size`, `.load()`, `.reset()`,
  `.probability(frame)` slots into `VADController`.

### Adding a tool

```python
class MyTool(Tool):
    name = "do_thing"
    description = "When the model should reach for this. Be specific."

    @property
    def input_schema(self):
        return {"type": "object", "properties": {...}, "required": [...]}

    def execute(self, **kwargs):
        return self._ok({...})   # or self._fail("why")
```

Then add it to the list in `pipeline.py`. That's the whole registration path.

---

## 6. The parts most likely to break

Read this section before you debug anything.

### 6.1 The VAD thresholds — highest risk, and it's a tuning problem

`VAD_MIN_SILENCE_MS` is the single most consequential number in the project.

- **Too low (< 400 ms)** — it cuts you off mid-sentence. You pause to think,
  it decides you're done, and transcribes half a question.
- **Too high (> 1200 ms)** — it feels sluggish. You finish, and nothing
  happens for over a second.

700 ms is the default. If it interrupts you, raise it. If it feels slow,
lower it. **This is the first thing to touch when the agent feels wrong**,
and no amount of better models compensates for a bad value here.

Related: `VAD_THRESHOLD` (0.5) is *entering* speech; `VAD_SILENCE_THRESHOLD`
(0.35) is *leaving* it. The gap between them is deliberate. Collapse them to
one value and the detector oscillates in noisy rooms.

### 6.2 Barge-in and the acoustic feedback loop

On laptop speakers, **the microphone hears the agent's own voice.** The VAD
cannot tell that from a human, so the agent interrupts itself, hears its own
interruption, and loops.

Mitigations in order of effectiveness:
1. **Headphones.** Solves it completely. Use these for demos.
2. Browser mode — `echoCancellation: true` is on and helps a lot.
3. `BARGE_IN_THRESHOLD=0.85` — a much higher bar to interrupt during
   playback.
4. `BARGE_IN=false` — deaf while speaking. No self-trigger, no barge-in.

A real fix needs acoustic echo cancellation. That is a separate project.

### 6.3 Small local models and tool calling

Anthropic and OpenAI have native tool calling and it's reliable. **Ollama does
not**, so `OllamaLLM` uses a JSON-contract prompt with a repair pass
(`OllamaLLM._parse`). Models below ~7B routinely emit prose around the JSON,
invent tool names, or ignore the schema.

If tool calls misbehave on a local model, that parser is where to look first.
Use a hosted provider to isolate whether a bug is yours or the model's.

### 6.4 Conversation trimming

`ConversationState._trim` must never leave a `tool` message whose originating
`assistant` tool call was trimmed away — providers reject that as malformed
and every subsequent turn fails. This is why trimming cuts forward to a clean
`user` turn rather than slicing at a fixed offset. Covered by
`tests/test_state.py`; don't "simplify" it.

### 6.5 STT hallucination on silence

Whisper handed near-silence emits confident nonsense — "Thank you.",
"Thanks for watching!", "you". Without filtering, the agent responds to a
cough. `HALLUCINATIONS` in `input_processor.py` catches the common ones. If
your agent replies to nothing, add what you see there.

### 6.6 The audio callback boundary

`sounddevice` runs its callback on a real-time OS thread. It does **one**
thing: copy the block into a queue. Put logging, inference, or disk I/O in
there and you get dropouts, which sound like the mic is broken. If
`dropped_chunks` is non-zero at shutdown, the consumer is too slow.

### 6.7 Sample rates

Three different rates live in this system and mixing them produces
chipmunk-or-slow-motion audio:

| Stage | Rate | Fixed by |
|---|---|---|
| Mic → VAD → STT | 16 kHz | Silero + Whisper both require it |
| VAD frame size | 512 samples | Silero requires exactly this |
| Kokoro output | 24 kHz | the model |

`SAMPLE_RATE` and `AUDIO_CHUNK_SIZE` in `.env` are **not** free parameters.

### 6.8 First-run downloads

`--check` looks like it hangs on first run. It's downloading Whisper (~75 MB
for `base.en`) and Kokoro weights. Subsequent starts take a few seconds.

---

## 7. Latency

Measured on Apple M4, `base.en`, hosted LLM:

| Stage | Typical |
|---|---|
| End of speech → VAD fires | = `VAD_MIN_SILENCE_MS` (700 ms) |
| STT decode (5 s utterance) | 300–600 ms |
| LLM, no tool | 500–900 ms |
| LLM + one tool | 1.2–2.0 s |
| TTS first sentence | 200–400 ms |
| **You stop → it starts talking** | **≈1.7–2.6 s** |

The biggest single win is lowering `VAD_MIN_SILENCE_MS`, since it's pure dead
time. Second is `STT_MODEL=tiny.en`. Every stage is logged, so watch the
`latency:` lines rather than guessing.

---

## 8. Deploying to Render

```bash
git init && git add . && git commit -m "voice agent"
# push to GitHub, then in Render: New → Blueprint → point at render.yaml
# set LLM_API_KEY in the dashboard
```

**Read this before you do:**

- **Free and Starter tiers will run out of memory.** torch + Whisper + Kokoro
  is roughly 1.5–2 GB resident. Use Standard or larger.
- **`render.yaml` sets `STT_MODEL=tiny.en`** on purpose. Shared vCPUs are far
  slower than your M4; `base.en` there feels sluggish.
- **Cold starts are brutal.** First request after a spin-down downloads and
  loads models — 60 s or more. Don't put a spin-down plan behind a demo.
- **The browser needs HTTPS for microphone access.** Render provides it;
  `localhost` is also exempt. Plain HTTP on a LAN IP will silently fail.
- **`requirements-server.txt`** omits `sounddevice`, because there's no
  PortAudio and no audio device on the box.
- WebSockets work on Render without configuration.

Honestly: **local is the better experience.** The server exists so you can
share a link, not because it's faster.

---

## 9. Security

- The model never executes code. It emits a tool *name*; the registry looks it
  up in a fixed dict built at startup. Unknown name → error. No `eval`, no
  `exec`, no shell.
- All SQL is fixed strings with bound parameters. The model chooses a query
  *kind* from an enum, never writes SQL. `tests/test_tools.py` includes an
  injection attempt that proves it's inert.
- Tool output is data in the prompt, never instruction.
- Audio is never written to disk and never leaves the machine. Only the
  finished text of your question reaches the LLM provider — and with
  `LLM_PROVIDER=ollama`, not even that.
- `.env` is gitignored. Model weights are gitignored.

---

## 10. Tests

```bash
pytest -m "not network"     # 79 tests, no network, no models, <1 s
pytest                      # includes the live Open-Meteo test
STT_RUN_MODEL=1 pytest tests/test_stt.py
```

The VAD tests replace Silero with a scripted probability list, so the state
machine is tested deterministically rather than against a neural model. The
agent tests use a scripted fake LLM for the same reason. Both are the right
place to add regression tests when something misbehaves.

---

## 11. Layout

```
voice_agent/
├── main.py                  local mic runtime + CLI
├── pipeline.py              wires every component together (shared)
├── config.py                the only place os.getenv is called
├── audio/input.py           mic → chunks
├── audio/output.py          chunks → speaker, interruptible
├── vad/detector.py          Silero + hysteresis state machine
├── stt/engine.py            faster-whisper behind a clean interface
├── processor/               transcript cleanup and noise rejection
├── agent/agent.py           the tool loop
├── agent/llm.py             Anthropic / OpenAI / Ollama, normalised
├── agent/prompts.py         system prompt (voice-optimised)
├── agent/state.py           conversation history with safe trimming
├── tools/base.py            Tool contract + registry
├── tools/weather.py         Open-Meteo
├── tools/personal_data.py   SQLite expenses (read + log_expense)
├── tools/memory.py          memory + notes tools
├── memory/store.py          memories, notes, conversation log (SQLite)
├── wake/detector.py         wake by name ("hey riya") or openWakeWord + awake window
├── audio/ducking.py         Spotify volume ducking (macOS)
├── tts/engine.py            Kokoro + sentence chunking
├── server/app.py            FastAPI + WebSocket (web runtime)
├── web/index.html           browser console
├── data/seed.py             sample expense data
└── tests/                   79 offline tests
```

---

## 12. What was deliberately left out

No wake word, no speaker identification, no long-term memory, no RAG, no
streaming STT, no Docker, no vector database, no agent framework. Each is a
reasonable addition *after* the pipeline is stable. Adding them first would
mean debugging four things at once when the mic cuts out.
