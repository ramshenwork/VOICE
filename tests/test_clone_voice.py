"""Cloned-voice client against a fake local server -- no model, no venv."""
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from config import TTSConfig
from tts.engine import TTSEngine


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"ok": True, "sample_rate": 24000, "voice": "x", "device": "cpu"})
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body.encode())

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        text = json.loads(self.rfile.read(n))["text"]
        pcm = np.full(len(text), 0.25, dtype="<f4").tobytes()
        self.send_response(200)
        self.send_header("X-Sample-Rate", "24000")
        self.send_header("Content-Length", str(len(pcm)))
        self.end_headers()
        self.wfile.write(pcm)


def test_clone_backend_uses_running_server():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cfg = replace(TTSConfig(), engine="clone", clone_port=srv.server_port,
                      clone_python="/nonexistent/python")
        tts = TTSEngine(cfg)
        audio = tts.synthesize("Hello there.")
        assert tts.backend == "clone" and tts.sample_rate == 24000
        assert audio.dtype == np.float32 and audio.size == len("Hello there.")
    finally:
        srv.shutdown()


def test_clone_backend_falls_back_when_unavailable():
    """No server, no venv: she keeps talking in her normal voice."""
    cfg = replace(TTSConfig(), engine="clone", clone_port=1, clone_python="/nonexistent/python")
    tts = TTSEngine(cfg)
    tts.load()
    assert tts.backend in ("kokoro", "system")
