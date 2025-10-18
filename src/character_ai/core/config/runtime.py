"""
Runtime configuration for Character AI Platform.

Contains runtime, system, and feature configuration classes.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration."""

    enabled: bool = True
    metrics_port: int = 8001
    prometheus_enabled: bool = True
    log_level: str = "INFO"
    structured_logging: bool = True

    # Performance tracking
    track_latency: bool = True
    track_memory: bool = True
    track_quality: bool = True
    track_errors: bool = True


@dataclass
class RuntimeConfig:
    """Runtime behavior configuration."""

    target_latency_s: float = 0.5
    streaming_enabled: bool = True
    predictive_loading: bool = True
    idle_timeout_s: int = 300


@dataclass
class InteractionConfig:
    """Interaction and conversation configuration."""

    stt_language: str = "en"
    min_audio_s: float = 0.2
    max_audio_s: float = 30.0
    max_new_tokens: int = 64
    temperature: float = 0.6
    sample_rate: int = 16000
    channels: int = 1


@dataclass
class TTSStreamingConfig:
    """TTS streaming configuration for reduced latency."""

    enabled: bool = False
    method: str = "sentence"  # sentence | full
    sentence_chunking: bool = True
    fallback_to_blocking: bool = True


@dataclass
class TTSConfig:
    """Text-to-speech configuration."""

    language: str = "en"
    default_voice_style: str = "neutral"
    compute_voice_embedding_on_ingest: bool = False
    streaming: TTSStreamingConfig = field(default_factory=TTSStreamingConfig)


@dataclass
class SafetyConfig:
    """Child-safety configuration."""

    banned_terms: List[str] = field(
        default_factory=lambda: ["kill", "hurt", "die", "blood"]
    )
    max_output_tokens: int = 64

    # Enhanced safety classifier settings
    enable_toxicity_classifier: bool = True
    enable_pii_detection: bool = True
    safety_confidence_threshold: float = 0.7
    log_safety_violations: bool = True


@dataclass
class MemoryConfig:
    """Session memory configuration."""

    max_turns: int = 10
    max_tokens: int = 1000
    max_age_seconds: int = 3600
    enable_memory: bool = True


@dataclass
class SecurityConfig:
    """Security configuration."""

    jwt_secret: str = field(
        default_factory=lambda: os.environ.get("CAI_JWT_SECRET", "")
    )
    jwt_algorithm: str = "HS256"
    jwt_expiry_seconds: int = 3600  # 1 hour
    device_id_file: Path = field(default_factory=lambda: Path("configs/device_id.json"))

    private_key_file: Path = field(
        default_factory=lambda: Path("configs/device_private.pem")
    )
    public_key_file: Path = field(
        default_factory=lambda: Path("configs/device_public.pem")
    )
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst: int = 10
    enable_device_registration: bool = True
    require_https: bool = False  # Set to True in production


@dataclass
class PathsConfig:
    """Filesystem paths for models and assets."""

    models_dir: Path = field(default_factory=lambda: Path.cwd() / "models")
    voices_dir: Path = field(default_factory=lambda: Path.cwd() / "catalog/voices")
    characters_dir: Path = field(
        default_factory=lambda: Path.cwd() / "configs/characters"
    )
    characters_index: Path = field(
        default_factory=lambda: Path.cwd() / "configs/characters/index.yaml"
    )


@dataclass
class APIConfig:
    """API server configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    rate_limit: int = 1000  # requests per minute
    max_request_size: int = 100 * 1024 * 1024  # 100MB
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    websocket_timeout: float = 300.0  # 5 minutes


@dataclass
class LanguageSupportConfig:
    """Multi-language support configuration."""

    default_language: str = "en"
    auto_detection_enabled: bool = True
    supported_languages: List[str] = field(
        default_factory=lambda: ["en", "es", "fr", "de", "zh", "ja", "ko", "ar"]
    )
    fallback_language: str = "en"
    cultural_adaptation: bool = True
    language_pack_dir: Path = field(
        default_factory=lambda: Path.cwd() / "configs/language_packs"
    )


@dataclass
class MultilingualAudioConfig:
    """Multi-language audio configuration."""

    tts_languages: List[str] = field(
        default_factory=lambda: ["en", "es", "fr", "de", "zh", "ja", "ko", "ar"]
    )
    stt_languages: List[str] = field(
        default_factory=lambda: ["en", "es", "fr", "de", "zh", "ja", "ko", "ar"]
    )
    voice_adaptation: bool = True
    cultural_voice_characteristics: bool = True
    auto_language_detection: bool = True
    language_confidence_threshold: float = 0.7


@dataclass
class PersonalizationConfig:
    """User personalization configuration."""

    enabled: bool = True
    learning_rate: float = 0.1
    max_preferences: int = 50
    privacy_mode: str = "strict"  # strict, moderate, open
    data_retention_days: int = 30
    adaptive_conversation: bool = True
    character_recommendations: bool = True
    preference_learning_enabled: bool = True
    style_adaptation: bool = True


@dataclass
class ParentalControlsConfig:
    """Parental controls configuration."""

    enabled: bool = True
    default_safety_level: str = "moderate"  # strict, moderate, lenient
    alert_threshold: float = 0.7
    time_limit_default: int = 60  # minutes
    content_filtering: bool = True
    usage_monitoring: bool = True
    parental_dashboard: bool = True
    safety_alerts_enabled: bool = True
    data_retention_days: int = 30


@dataclass
class StreamingConfig:
    """Streaming and performance configuration."""

    token_generation_delay: float = 0.05  # seconds between tokens
    placeholder_delay: float = 0.1  # seconds for placeholder responses
    connection_timeout_ms: int = 30000  # milliseconds
    max_response_time_ms: int = 30000  # milliseconds
    simulation_delays: bool = False  # enable simulation delays in performance API
