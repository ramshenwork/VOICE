"""
Central configuration.

Every tunable value in the system is read here, once, from environment
variables. No other module calls os.getenv(). Modules receive the config
object (or the sub-config they need) at construction time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _s(key: str, default: str) -> str:
    v = os.getenv(key)
    return default if v is None or v.strip() == "" else v.strip()


def _i(key: str, default: int) -> int:
    try:
        return int(_s(key, str(default)))
    except ValueError:
        return default


def _f(key: str, default: float) -> float:
    try:
        return float(_s(key, str(default)))
    except ValueError:
        return default


def _b(key: str, default: bool) -> bool:
    return _s(key, "true" if default else "false").lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AudioConfig:
    """
    SAMPLE_RATE and CHUNK_SIZE are NOT free parameters.

    Silero VAD requires exactly 512 samples per frame at 16 kHz.
    faster-whisper expects 16 kHz mono float32.
    Change these only if you also change the VAD frame handling.
    """
    sample_rate: int = field(default_factory=lambda: _i("SAMPLE_RATE", 16000))
    channels: int = field(default_factory=lambda: _i("CHANNELS", 1))
    chunk_size: int = field(default_factory=lambda: _i("AUDIO_CHUNK_SIZE", 512))
    input_device: str = field(default_factory=lambda: _s("AUDIO_INPUT_DEVICE", ""))
    output_device: str = field(default_factory=lambda: _s("AUDIO_OUTPUT_DEVICE", ""))

    @property
    def chunk_ms(self) -> float:
        return 1000.0 * self.chunk_size / self.sample_rate


# --------------------------------------------------------------------------
# VAD
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class VADConfig:
    speech_threshold: float = field(default_factory=lambda: _f("VAD_THRESHOLD", 0.5))
    silence_threshold: float = field(default_factory=lambda: _f("VAD_SILENCE_THRESHOLD", 0.35))
    min_speech_ms: int = field(default_factory=lambda: _i("VAD_MIN_SPEECH_MS", 250))
    min_silence_ms: int = field(default_factory=lambda: _i("VAD_MIN_SILENCE_MS", 700))
    pre_speech_pad_ms: int = field(default_factory=lambda: _i("VAD_PRE_SPEECH_PAD_MS", 300))
    post_speech_pad_ms: int = field(default_factory=lambda: _i("VAD_POST_SPEECH_PAD_MS", 200))
    max_utterance_ms: int = field(default_factory=lambda: _i("VAD_MAX_UTTERANCE_MS", 20000))


# --------------------------------------------------------------------------
# STT
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class STTConfig:
    model: str = field(default_factory=lambda: _s("STT_MODEL", "base.en"))
    device: str = field(default_factory=lambda: _s("STT_DEVICE", "cpu"))
    compute_type: str = field(default_factory=lambda: _s("STT_COMPUTE_TYPE", "int8"))
    language: str = field(default_factory=lambda: _s("STT_LANGUAGE", "en"))
    beam_size: int = field(default_factory=lambda: _i("STT_BEAM_SIZE", 1))
    vad_filter: bool = field(default_factory=lambda: _b("STT_VAD_FILTER", False))
    min_audio_ms: int = field(default_factory=lambda: _i("STT_MIN_AUDIO_MS", 300))
    # Vocabulary hint for the recogniser: names and words it should expect.
    prompt: str = field(default_factory=lambda: _s("STT_PROMPT", "Hey Riya."))


# --------------------------------------------------------------------------
# TTS
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TTSConfig:
    engine: str = field(default_factory=lambda: _s("TTS_ENGINE", "kokoro"))
    voice: str = field(default_factory=lambda: _s("TTS_VOICE", "af_heart"))
    lang_code: str = field(default_factory=lambda: _s("TTS_LANG_CODE", "a"))
    speed: float = field(default_factory=lambda: _f("TTS_SPEED", 1.0))
    sample_rate: int = field(default_factory=lambda: _i("TTS_SAMPLE_RATE", 24000))


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LLMConfig:
    provider: str = field(default_factory=lambda: _s("LLM_PROVIDER", "anthropic"))
    model: str = field(default_factory=lambda: _s("LLM_MODEL", "claude-sonnet-4-5"))
    api_key: str = field(default_factory=lambda: _s("LLM_API_KEY", ""))
    base_url: str = field(default_factory=lambda: _s("LLM_BASE_URL", ""))
    max_tokens: int = field(default_factory=lambda: _i("LLM_MAX_TOKENS", 400))
    temperature: float = field(default_factory=lambda: _f("LLM_TEMPERATURE", 0.4))
    max_tool_rounds: int = field(default_factory=lambda: _i("LLM_MAX_TOOL_ROUNDS", 3))
    timeout_s: float = field(default_factory=lambda: _f("LLM_TIMEOUT_S", 30.0))


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolsConfig:
    weather_provider: str = field(default_factory=lambda: _s("WEATHER_PROVIDER", "open_meteo"))
    default_location: str = field(default_factory=lambda: _s("DEFAULT_LOCATION", "Hyderabad"))
    database_path: str = field(default_factory=lambda: _s("DATABASE_PATH", "data/agent.db"))
    currency_symbol: str = field(default_factory=lambda: _s("CURRENCY_SYMBOL", "Rs."))
    http_timeout_s: float = field(default_factory=lambda: _f("HTTP_TIMEOUT_S", 8.0))


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ServerConfig:
    host: str = field(default_factory=lambda: _s("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _i("PORT", 8000))
    cors_origins: str = field(default_factory=lambda: _s("CORS_ORIGINS", "*"))


# --------------------------------------------------------------------------
# Assistant persona
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AssistantConfig:
    name: str = field(default_factory=lambda: _s("ASSISTANT_NAME", "Riya"))
    # How the assistant addresses you. Empty -> "sir".
    user_name: str = field(default_factory=lambda: _s("USER_NAME", ""))
    timezone: str = field(default_factory=lambda: _s("TIMEZONE", "Asia/Kolkata"))
    # How many remembered facts are put in front of the model every turn.
    memory_in_prompt: int = field(default_factory=lambda: _i("MEMORY_IN_PROMPT", 40))
    # Plain user/assistant turns restored from the last session on startup.
    restore_turns: int = field(default_factory=lambda: _i("RESTORE_TURNS", 6))

    @property
    def address(self) -> str:
        return self.user_name or "sir"


# --------------------------------------------------------------------------
# Wake word + "conversation window"
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class WakeConfig:
    """
    Two wake engines:
      transcript   -- the speech recogniser listens for the name ("Riya") in
                      anything said. Works for any name with no training, at
                      the cost of more CPU while asleep.
      openwakeword -- a tiny dedicated model; near-zero CPU, but only for names
                      that have a trained model (e.g. hey_jarvis).
    """
    enabled: bool = field(default_factory=lambda: _b("WAKE_WORD_ENABLED", True))
    engine: str = field(default_factory=lambda: _s("WAKE_WORD_ENGINE", "transcript"))
    # transcript engine: spellings the recogniser may produce for the name.
    phrases: str = field(default_factory=lambda: _s("WAKE_PHRASES", "riya,ria,rhea,reya,riyah,reah"))
    model: str = field(default_factory=lambda: _s("WAKE_WORD_MODEL", "hey_jarvis"))
    threshold: float = field(default_factory=lambda: _f("WAKE_WORD_THRESHOLD", 0.5))
    # After a reply, stay listening this long for a follow-up without the wake word.
    follow_up_s: float = field(default_factory=lambda: _f("FOLLOW_UP_WINDOW_S", 8.0))
    acknowledgement: str = field(default_factory=lambda: _s("WAKE_ACK", "Yes, {address}?"))
    duck_spotify: bool = field(default_factory=lambda: _b("DUCK_SPOTIFY", True))
    duck_volume: int = field(default_factory=lambda: _i("DUCK_VOLUME", 15))


@dataclass(frozen=True)
class Config:
    assistant: AssistantConfig = field(default_factory=AssistantConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    log_level: str = field(default_factory=lambda: _s("LOG_LEVEL", "INFO"))
    root: Path = ROOT

    def db_path(self) -> Path:
        p = Path(self.tools.database_path)
        return p if p.is_absolute() else self.root / p


def load_config() -> Config:
    return Config()
