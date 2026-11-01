"""
Pipeline assembly.

One place where every component is constructed and wired, so the local mic
runner and the web server share identical behaviour and there is exactly one
definition of "the system".

Model loading is eager and up front. Lazy loading looks tidy but pushes a
two-second model load into the middle of the user's first sentence, which is
precisely the moment latency is most noticeable.
"""

from __future__ import annotations

import logging
import sys
import time

from agent.agent import Agent
from agent.state import ConversationState
from config import Config, load_config
from processor.input_processor import InputProcessor
from stt.engine import STTEngine
from tools.base import ToolRegistry
from tools.personal_data import PersonalDataTool
from tools.weather import WeatherTool
from tts.engine import TTSEngine
from vad.detector import VADController

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("httpx", "httpcore", "urllib3", "faster_whisper", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Pipeline:
    """Holds every long-lived component. Construct once, reuse per turn."""

    def __init__(self, config: Config | None = None):
        self.cfg = config or load_config()
        self.vad = VADController(self.cfg.vad, self.cfg.audio.sample_rate)
        self.stt = STTEngine(self.cfg.stt)
        self.processor = InputProcessor()
        self.tts = TTSEngine(self.cfg.tts)
        self.registry = ToolRegistry([
            WeatherTool(self.cfg.tools),
            PersonalDataTool(self.cfg.tools, self.cfg.db_path()),
        ])
        self.state = ConversationState()
        self.agent = Agent(self.cfg.llm, self.registry, self.state)

    def warmup(self) -> None:
        t0 = time.perf_counter()
        logger.info("loading models...")
        self.vad.load()
        self.stt.load()
        self.tts.load()
        try:
            self.agent.load()
        except Exception as exc:
            logger.error("LLM not ready: %s", exc)
        logger.info("all models ready in %.1fs", time.perf_counter() - t0)

    def handle_utterance(self, audio, sample_rate: int = 16000):
        """
        audio (float32 16 kHz) -> (spoken_text, transcript) or (None, reason).

        The single place the whole chain runs, so latency instrumentation
        lives here rather than being scattered.
        """
        t_stt = time.perf_counter()
        try:
            transcription = self.stt.transcribe(audio, sample_rate)
        except Exception as exc:
            logger.error("stt error: %s", exc)
            return None, "stt_error"
        stt_ms = (time.perf_counter() - t_stt) * 1000

        processed = self.processor.process(transcription)
        if not processed.should_respond:
            logger.info("input discarded: %s (%r)", processed.reason, processed.raw_text)
            return None, processed.reason

        t_agent = time.perf_counter()
        reply = self.agent.respond(processed.text)
        agent_ms = (time.perf_counter() - t_agent) * 1000

        logger.info("latency: stt=%.0fms agent=%.0fms tools=%s",
                    stt_ms, agent_ms, reply.tools_used or "none")
        return reply, processed.text
