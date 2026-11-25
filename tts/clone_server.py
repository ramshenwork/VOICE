"""
Cloned-voice server (Chatterbox), run in its own virtualenv.

Chatterbox pins older torch/numpy/transformers than the rest of Riya, so it
lives in .venv-voice and runs as a tiny local HTTP service. Riya's TTSEngine
starts it automatically when TTS_ENGINE=clone; you never run this by hand.

    .venv-voice/bin/python tts/clone_server.py --port 5175 --voice voice/sample.wav

Listens on 127.0.0.1 only.
  GET  /health          -> {"ok": true, "sample_rate": 24000}
  POST /speak {"text"}  -> raw little-endian float32 mono PCM
Uses only the standard library besides chatterbox itself.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s clone-tts %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger("clone")

MAX_TEXT = 600


def pick_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(device: str, variant: str = "turbo"):
    import torch
    if variant == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS as Model
    else:
        from chatterbox.tts import ChatterboxTTS as Model
    if device != "cuda":
        # Checkpoints were saved on CUDA; map them to wherever we are.
        _load = torch.load

        def patched(*a, **kw):
            kw.setdefault("map_location", torch.device(device))
            return _load(*a, **kw)
        torch.load = patched
    return Model.from_pretrained(device=device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5175)
    ap.add_argument("--voice", default="", help="reference sample (wav/m4a); empty = default voice")
    ap.add_argument("--exaggeration", type=float, default=0.5)
    ap.add_argument("--cfg-weight", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--model", default="turbo", choices=["turbo", "standard"],
                    help="turbo: fast enough for conversation; standard: more expressive, slower")
    args = ap.parse_args()

    t0 = time.time()
    device = pick_device(args.device)
    try:
        model = load_model(device, args.model)
    except Exception as exc:
        if device == "cpu":
            raise
        log.warning("could not load on %s (%s); using cpu", device, exc)
        device = "cpu"
        model = load_model(device, args.model)

    voice = args.voice if args.voice and os.path.isfile(args.voice) else ""
    if args.voice and not voice:
        log.warning("voice sample %s not found; using the default voice", args.voice)
    if voice:
        model.prepare_conditionals(voice, exaggeration=args.exaggeration)
    try:  # first generation compiles GPU kernels; pay for it now, not mid-reply
        model.generate("Ready.", temperature=args.temperature) if args.model == "turbo" \
            else model.generate("Ready.", exaggeration=args.exaggeration)
    except Exception as exc:
        log.warning("warm-up failed: %s", exc)
    log.info("ready on %s in %.1fs, model=%s voice=%s", device, time.time() - t0, args.model,
             voice or "default")

    lock = threading.Lock()  # one synthesis at a time on the GPU

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"ok": True, "sample_rate": model.sr, "device": device,
                                 "voice": voice or "default"})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/speak":
                return self._json(404, {"error": "not found"})
            try:
                n = min(int(self.headers.get("Content-Length", 0)), 20000)
                text = str(json.loads(self.rfile.read(n) or b"{}").get("text", "")).strip()
            except Exception:
                return self._json(400, {"error": "bad request"})
            if not text:
                return self._json(400, {"error": "empty text"})
            t = time.time()
            try:
                with lock:
                    if args.model == "turbo":
                        wav = model.generate(text[:MAX_TEXT], temperature=args.temperature)
                    else:
                        wav = model.generate(text[:MAX_TEXT], exaggeration=args.exaggeration,
                                             cfg_weight=args.cfg_weight,
                                             temperature=args.temperature)
            except Exception as exc:
                log.error("synthesis failed: %s", exc)
                return self._json(500, {"error": str(exc)[:200]})
            pcm = wav.detach().cpu().numpy().reshape(-1).astype("<f4").tobytes()
            log.info("%.2fs for %d chars", time.time() - t, len(text))
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-Sample-Rate", str(model.sr))
            self.send_header("Content-Length", str(len(pcm)))
            self.end_headers()
            self.wfile.write(pcm)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
