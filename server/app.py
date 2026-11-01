"""
Web runtime -- the same pipeline, driven by a browser instead of a local mic.

This exists because a server has no microphone and no speakers. The browser
becomes the audio device:

    browser mic -> resample to 16 kHz -> PCM16 -> WebSocket
        -> [ VAD -> STT -> agent -> tools -> TTS ] on the server
        -> WAV bytes -> WebSocket -> browser <audio>

Every connection gets its own VADController and ConversationState, because
both are stateful per speaker. The heavy models (Whisper, Kokoro, the tool
registry) are shared across connections -- they are stateless per call and
loading a copy per user would exhaust memory immediately.

Concurrency: STT and TTS are blocking CPU work. Running them directly in the
async handler would freeze every other connection, so they are dispatched to a
thread pool via run_in_executor.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import struct
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from agent.agent import Agent
from agent.state import ConversationState
from audio.output import pcm16_to_float
from config import load_config
from pipeline import Pipeline, setup_logging
from vad.detector import VADController, VADEvent

logger = logging.getLogger("server")

cfg = load_config()
setup_logging(cfg.log_level)

app = FastAPI(title="Local Voice Agent", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cfg.server.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
pipeline: Pipeline | None = None


@app.on_event("startup")
async def startup() -> None:
    global pipeline
    pipeline = Pipeline(cfg)
    await asyncio.get_running_loop().run_in_executor(None, pipeline.warmup)
    logger.info("server ready on %s:%d", cfg.server.host, cfg.server.port)


# --------------------------------------------------------------------------
# WAV encoding -- browsers will not play raw PCM
# --------------------------------------------------------------------------
def to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(pcm)))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(pcm)))
    buf.write(pcm)
    return buf.getvalue()


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok" if pipeline else "starting",
        "stt": pipeline.cfg.stt.model if pipeline else None,
        "tts": pipeline.tts.backend if pipeline else None,
        "llm": f"{cfg.llm.provider}/{cfg.llm.model}",
        "tools": pipeline.registry.names() if pipeline else [],
    }


class ChatRequest(BaseModel):
    text: str
    speak: bool = False


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Typed input. Useful for testing the agent without a microphone."""
    if pipeline is None:
        return JSONResponse({"error": "still starting"}, status_code=503)
    loop = asyncio.get_running_loop()
    reply = await loop.run_in_executor(None, pipeline.agent.respond, req.text)
    out = {"text": reply.text, "tools_used": reply.tools_used,
           "latency_s": round(reply.latency_s, 2), "error": reply.error}
    if req.speak:
        audio = await loop.run_in_executor(None, pipeline.tts.synthesize, reply.text)
        import base64
        out["audio_wav_b64"] = base64.b64encode(
            to_wav(audio, pipeline.tts.sample_rate)).decode()
    return out


@app.post("/api/reset")
async def reset():
    if pipeline:
        pipeline.agent.reset()
    return {"status": "ok"}


# --------------------------------------------------------------------------
# WebSocket voice loop
# --------------------------------------------------------------------------
@app.websocket("/ws")
async def ws_voice(ws: WebSocket):
    await ws.accept()
    if pipeline is None:
        await ws.close(code=1013)
        return

    loop = asyncio.get_running_loop()
    # Per-connection state. Models are shared; state is not.
    vad = VADController(cfg.vad, cfg.audio.sample_rate, model=pipeline.vad.model)
    state = ConversationState()
    agent = Agent(cfg.llm, pipeline.registry, state)
    agent.load()

    await ws.send_json({"type": "ready", "tts_backend": pipeline.tts.backend})
    logger.info("ws connected")

    try:
        while True:
            message = await ws.receive()

            if "text" in message and message["text"]:
                cmd = json.loads(message["text"])
                if cmd.get("type") == "reset":
                    state.reset()
                    vad.reset()
                    await ws.send_json({"type": "reset_ok"})
                continue

            if "bytes" not in message or message["bytes"] is None:
                continue

            audio = pcm16_to_float(message["bytes"])

            for result in vad.process(audio):
                if result.event is VADEvent.SPEECH_STARTED:
                    await ws.send_json({"type": "speech_started"})

                elif result.event is VADEvent.SPEECH_ENDED and result.segment is not None:
                    await ws.send_json({"type": "speech_ended",
                                        "duration_s": round(result.segment_duration_s, 2)})

                    transcription = await loop.run_in_executor(
                        None, pipeline.stt.transcribe, result.segment, cfg.audio.sample_rate)
                    processed = pipeline.processor.process(transcription)

                    if not processed.should_respond:
                        await ws.send_json({"type": "discarded", "reason": processed.reason})
                        continue

                    await ws.send_json({"type": "transcript", "text": processed.text})

                    reply = await loop.run_in_executor(None, agent.respond, processed.text)
                    await ws.send_json({"type": "reply", "text": reply.text,
                                        "tools_used": reply.tools_used,
                                        "latency_s": round(reply.latency_s, 2)})

                    # Stream audio per sentence so playback starts early.
                    for chunk_text in _sentences(pipeline, reply.text):
                        audio_out = await loop.run_in_executor(
                            None, pipeline.tts.synthesize, chunk_text)
                        if audio_out.size:
                            await ws.send_bytes(to_wav(audio_out, pipeline.tts.sample_rate))
                    await ws.send_json({"type": "audio_done"})

    except WebSocketDisconnect:
        logger.info("ws disconnected")
    except Exception as exc:
        logger.exception("ws error")
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


def _sentences(pl: Pipeline, text: str) -> list[str]:
    from tts.engine import split_for_speech
    return split_for_speech(text)


def run() -> None:
    import uvicorn
    uvicorn.run("server.app:app", host=cfg.server.host, port=cfg.server.port, log_level="info")


if __name__ == "__main__":
    run()
