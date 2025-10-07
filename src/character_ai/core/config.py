"""
Configuration management for Character AI Platform.

Handles environment-specific configurations, GPU settings, and model parameters.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)

# Default model constants
DEFAULT_COQUI_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


class Environment(Enum):
    """Environment types for configuration."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    DEMO = "demo"
    TESTING = "testing"


@dataclass
class GPUConfig:
    """GPU configuration settings."""

    device: str = "cuda:0"
    memory_limit: str = "2GB"  # Edge default; adjusted per environment
    precision: str = "mixed"  # mixed, fp16, int8, int4
    batch_size: int = 8
    quantization: str = "fp16"  # 8bit, 4bit, fp16
    gradient_checkpointing: bool = True
    torch_compile: bool = True
    cuda_graphs: bool = True
    max_gpu_memory_gb: float = 20.0  # Leave 4GB for system
    cache_size: int = 1000  # Embedding cache size
    enable_memory_optimization: bool = True

    def __post_init__(self) -> None:
        """Validate GPU configuration."""
        if self.precision not in ["mixed", "fp16", "int8", "int4"]:
            raise ValueError(f"Invalid precision: {self.precision}")
        if self.quantization not in ["8bit", "4bit", "fp16"]:
            raise ValueError(f"Invalid quantization: {self.quantization}")


@dataclass
class RepresentationLearningConfig:
    """Representation learning model configuration."""

    wav2vec2_model: str = "facebook/wav2vec2-base-960h"
    clap_model: str = "laion/larger_clap_music_and_speech"
    emotion_model: str = "facebook/wav2vec2-base-960h"
    max_audio_length: float = 30.0
    sample_rate: int = 16000


@dataclass
class VoiceCloningConfig:
    """Voice cloning model configuration."""

    coqui_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    bark_model: str = "suno/bark"
    sovits_model: str = "so-vits-svc"
    max_audio_length: float = 30.0
    sample_rate: int = 22050


@dataclass
class MultimodalFusionConfig:
    """Multimodal fusion model configuration."""

    # Model selection with fallbacks
    mistral_model: str = "mistralai/Mistral-7B-v0.1"  # Gated model (requires HF token)
    phi_model: str = (
        "microsoft/Phi-3-mini-4k-instruct"  # Open-source model (no auth required)
    )
    fallback_model: str = "gpt2"  # Dev fallback
    clap_model: str = "laion/larger_clap_music_and_speech"
    max_context_length: int = 2048
    temperature: float = 0.7

    # Authentication
    huggingface_token: Optional[str] = None
    use_gated_models: bool = True


@dataclass
class ConversationalAIConfig:
    """Conversational AI model configuration."""

    wav2vec2_model: str = "facebook/wav2vec2-base"
    coqui_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    llama_model: str = "meta-llama/Llama-2-7b-hf"
    max_audio_length: float = 30.0
    sample_rate: int = 16000


@dataclass
class ModelConfig:
    """Model configuration settings."""

    # Conversational AI
    wav2vec2_model: str = "facebook/wav2vec2-base"  # Wav2Vec2 model for STT
    coqui_model: str = "tts_models/en/ljspeech/tacotron2-DDC"  # Coqui TTS model
    llama_model: str = "llama-2-7b-chat"
    llama_quantization: str = "4bit"  # 4bit, 8bit, fp16
    llama_backend: str = "transformers"  # transformers | llama_cpp
    llama_gguf_path: str = "models/llm/tinyllama-1.1b-q4_k_m.gguf"

    # Representation Learning
    clap_model: str = "laion/larger_clap_music_and_speech"
    bark_model: str = "suno/bark"
    so_vits_model: str = "so-vits-svc"

    # Performance
    max_audio_length: float = 30.0  # seconds
    sample_rate: int = 16000
    max_concurrent_requests: int = 10
    request_timeout: float = 30.0

    # Nested configurations for specific components
    representation_learning: Optional[RepresentationLearningConfig] = None
    voice_cloning: Optional[VoiceCloningConfig] = None
    multimodal_fusion: Optional[MultimodalFusionConfig] = None
    conversational_ai: Optional[ConversationalAIConfig] = None

    def __post_init__(self) -> None:
        """Validate model configuration."""
        if self.llama_quantization not in ["4bit", "8bit", "fp16"]:
            raise ValueError(f"Invalid Llama quantization: {self.llama_quantization}")


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
class TTSConfig:
    """Text-to-speech configuration."""

    language: str = "en"
    default_voice_style: str = "neutral"
    compute_voice_embedding_on_ingest: bool = False


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


@dataclass
class Config:
    """Main configuration class for Character AI Platform."""

    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False

    # GPU configuration
    gpu: GPUConfig = field(default_factory=GPUConfig)

    # Model configuration
    models: ModelConfig = field(default_factory=ModelConfig)

    # Multimodal fusion configuration
    multimodal_fusion: MultimodalFusionConfig = field(
        default_factory=MultimodalFusionConfig
    )

    # Monitoring configuration
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Runtime and behavior
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    # API configuration
    api: APIConfig = field(default_factory=APIConfig)

    # New feature configurations
    language_support: LanguageSupportConfig = field(
        default_factory=LanguageSupportConfig
    )
    multilingual_audio: MultilingualAudioConfig = field(
        default_factory=MultilingualAudioConfig
    )
    personalization: PersonalizationConfig = field(
        default_factory=PersonalizationConfig
    )
    parental_controls: ParentalControlsConfig = field(
        default_factory=ParentalControlsConfig
    )
    streaming: StreamingConfig = field(default_factory=StreamingConfig)

    # Paths - use absolute paths to avoid creating directories in project root
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "data")
    models_dir: Path = field(default_factory=lambda: Path.cwd() / "models")
    cache_dir: Path = field(default_factory=lambda: Path.cwd() / "cache")
    logs_dir: Path = field(default_factory=lambda: Path.cwd() / "logs")

    # Free model settings
    use_free_models_only: bool = True
    fallback_to_api: bool = False

    # CPU limiting settings (for development/testing)
    max_cpu_threads: Optional[int] = None  # None = use all cores, 2 = limit for testing

    enable_cpu_limiting: bool = False  # Enable CPU limiting for development
    blas_threads: Optional[int] = None  # BLAS/LAPACK thread limit
    torch_threads: Optional[int] = None  # PyTorch thread limit
    omp_threads: Optional[int] = None  # OpenMP thread limit

    def __post_init__(self) -> None:
        """Initialize configuration and create directories."""
        # Load from runtime.yaml if available
        self._load_from_runtime_yaml()

        # Don't create directories during instantiation - create when actually needed

        # Set environment-specific defaults
        if self.environment == Environment.PRODUCTION:
            self.gpu.precision = "fp16"
            self.gpu.quantization = "fp16"
            self.gpu.gradient_checkpointing = False
            self.monitoring.enabled = True
            self.debug = False
        elif self.environment == Environment.DEMO:
            self.gpu.batch_size = 4
            self.models.max_concurrent_requests = 5
            self.monitoring.enabled = False
        elif self.environment == Environment.TESTING:
            self.gpu.batch_size = 2
            self.models.max_concurrent_requests = 2
            self.monitoring.enabled = False
            self.debug = True
            # Enable CPU limiting for testing by default
            if self.max_cpu_threads is None:
                self.max_cpu_threads = 2
            if not self.enable_cpu_limiting:
                self.enable_cpu_limiting = True

        # Apply CPU limiting if enabled
        if self.enable_cpu_limiting and self.max_cpu_threads is not None:
            self._apply_cpu_limiting()

    def _load_from_runtime_yaml(self) -> None:
        """Load configuration from runtime.yaml if available."""
        try:
            from pathlib import Path

            import yaml

            runtime_path = Path("configs/runtime.yaml")
            if not runtime_path.exists():
                return

            with open(runtime_path, "r") as f:
                data = yaml.safe_load(f) or {}

            # Load TTS model from runtime.yaml
            tts_section = data.get("tts", {}) or {}
            if tts_section.get("model_name"):
                self.models.coqui_model = str(tts_section["model_name"])

        except Exception as e:
            # Non-fatal - just log and continue with defaults
            logger.debug(f"Could not load runtime.yaml: {e}")

    def _apply_cpu_limiting(self) -> None:
        """Apply CPU limiting for development/testing."""
        import os

        # Set environment variables for BLAS/LAPACK threading
        if self.blas_threads is not None:
            os.environ["OPENBLAS_NUM_THREADS"] = str(self.blas_threads)
            os.environ["MKL_NUM_THREADS"] = str(self.blas_threads)
            os.environ["VECLIB_MAXIMUM_THREADS"] = str(self.blas_threads)
        else:
            os.environ["OPENBLAS_NUM_THREADS"] = str(self.max_cpu_threads)
            os.environ["MKL_NUM_THREADS"] = str(self.max_cpu_threads)
            os.environ["VECLIB_MAXIMUM_THREADS"] = str(self.max_cpu_threads)

        # Set OpenMP threading
        if self.omp_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(self.omp_threads)
            os.environ["NUMEXPR_NUM_THREADS"] = str(self.omp_threads)
        else:
            os.environ["OMP_NUM_THREADS"] = str(self.max_cpu_threads)
            os.environ["NUMEXPR_NUM_THREADS"] = str(self.max_cpu_threads)

        # Configure PyTorch threading
        try:
            import torch

            if self.torch_threads is not None:
                if hasattr(torch, "set_num_threads"):
                    torch.set_num_threads(self.torch_threads)
                # Only set interop threads if PyTorch hasn't been used yet
                try:
                    if hasattr(torch, "set_num_interop_threads"):
                        torch.set_num_interop_threads(self.torch_threads)
                except RuntimeError:
                    # PyTorch already initialized, skip interop threads
                    pass
            else:
                if self.max_cpu_threads is not None:
                    if hasattr(torch, "set_num_threads"):
                        torch.set_num_threads(self.max_cpu_threads)
                # Only set interop threads if PyTorch hasn't been used yet
                try:
                    if self.max_cpu_threads is not None and hasattr(
                        torch, "set_num_interop_threads"
                    ):
                        torch.set_num_interop_threads(self.max_cpu_threads)
                except RuntimeError:
                    # PyTorch already initialized, skip interop threads
                    pass
        except (ImportError, AttributeError):
            pass  # PyTorch not available or missing methods

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        # Convert environment string to enum
        if "environment" in data:
            data["environment"] = Environment(data["environment"])

        # Create nested configs
        gpu_data = data.get("gpu", {})
        models_data = data.get("models", {})
        monitoring_data = data.get("monitoring", {})
        api_data = data.get("api", {})
        runtime_data = data.get("runtime", {})
        interaction_data = data.get("interaction", {})
        tts_data = data.get("tts", {})
        safety_data = data.get("safety", {})
        paths_data = data.get("paths", {})

        return cls(
            environment=data.get("environment", Environment.DEVELOPMENT),
            debug=data.get("debug", False),
            gpu=GPUConfig(**gpu_data),
            models=ModelConfig(**models_data),
            monitoring=MonitoringConfig(**monitoring_data),
            api=APIConfig(**api_data),
            runtime=RuntimeConfig(**runtime_data),
            interaction=InteractionConfig(**interaction_data),
            tts=TTSConfig(**tts_data),
            safety=SafetyConfig(**safety_data),
            paths=PathsConfig(**paths_data),
            data_dir=Path(data.get("data_dir", "data")),
            models_dir=Path(data.get("models_dir", "models")),
            cache_dir=Path(data.get("cache_dir", "cache")),
            logs_dir=Path(data.get("logs_dir", "logs")),
            use_free_models_only=data.get("use_free_models_only", True),
            fallback_to_api=data.get("fallback_to_api", False),
        )

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""

        def getenv_bool(name: str, default: bool) -> bool:
            v = os.getenv(name)
            return default if v is None else v.lower() in {"1", "true", "yes", "on"}

        def getenv_int(name: str, default: int) -> int:
            v = os.getenv(name)
            return default if v is None else int(v)

        def getenv_float(name: str, default: float) -> float:
            v = os.getenv(name)
            return default if v is None else float(v)

        def getenv_str(name: str, default: str) -> str:
            return os.getenv(name, default)

        # CAI_* env overrides only
        env = Environment(getenv_str("CAI_ENV", "development"))
        debug = getenv_bool("CAI_DEBUG", False)

        gpu = GPUConfig(
            device=getenv_str("CAI_GPU__DEVICE", "cpu"),
            memory_limit=getenv_str("CAI_GPU__MEMORY_LIMIT", "2GB"),
            precision=getenv_str("CAI_GPU__PRECISION", "fp16"),
            batch_size=getenv_int("CAI_GPU__BATCH_SIZE", 2),
            quantization=getenv_str("CAI_GPU__QUANTIZATION", "4bit"),
        )

        models = ModelConfig(
            wav2vec2_model=getenv_str(
                "CAI_MODELS__WAV2VEC2_MODEL", "facebook/wav2vec2-base"
            ),
            coqui_model=getenv_str(
                "CAI_MODELS__COQUI_MODEL",
                DEFAULT_COQUI_MODEL,
            ),
            llama_model=getenv_str(
                "CAI_MODELS__LLAMA_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            ),
            llama_quantization=getenv_str("CAI_MODELS__LLAMA_QUANTIZATION", "4bit"),
            llama_backend=getenv_str("CAI_MODELS__LLAMA_BACKEND", "llama_cpp"),
            llama_gguf_path=getenv_str(
                "CAI_MODELS__LLAMA_GGUF_PATH", "models/llm/tinyllama-1.1b-q4_k_m.gguf"
            ),
        )

        monitoring = MonitoringConfig()
        api = APIConfig()

        runtime = RuntimeConfig(
            target_latency_s=getenv_float("CAI_RUNTIME__TARGET_LATENCY_S", 0.5),
            streaming_enabled=getenv_bool("CAI_RUNTIME__STREAMING_ENABLED", True),
            predictive_loading=getenv_bool("CAI_RUNTIME__PREDICTIVE_LOADING", True),
            idle_timeout_s=getenv_int("CAI_RUNTIME__IDLE_TIMEOUT_S", 300),
        )

        interaction = InteractionConfig(
            stt_language=getenv_str("CAI_INTERACTION__STT_LANGUAGE", "en"),
            min_audio_s=getenv_float("CAI_INTERACTION__MIN_AUDIO_S", 0.2),
            max_audio_s=getenv_float("CAI_INTERACTION__MAX_AUDIO_S", 30.0),
            max_new_tokens=getenv_int("CAI_INTERACTION__MAX_NEW_TOKENS", 64),
            temperature=getenv_float("CAI_INTERACTION__TEMPERATURE", 0.6),
        )

        tts = TTSConfig(
            language=getenv_str("CAI_TTS__LANGUAGE", "en"),
            default_voice_style=getenv_str("CAI_TTS__DEFAULT_VOICE_STYLE", "neutral"),
        )

        # For lists, support comma-separated env var CAI_SAFETY__BANNED_TERMS
        banned_terms_env = os.getenv("CAI_SAFETY__BANNED_TERMS")
        banned_terms = (
            [t.strip() for t in banned_terms_env.split(",")]
            if banned_terms_env
            else ["kill", "hurt", "die", "blood"]
        )
        safety = SafetyConfig(
            banned_terms=banned_terms,
            max_output_tokens=getenv_int("CAI_SAFETY__MAX_OUTPUT_TOKENS", 64),
        )

        paths = PathsConfig(
            models_dir=Path(getenv_str("CAI_PATHS__MODELS_DIR", "models")),
            voices_dir=Path(getenv_str("CAI_PATHS__VOICES_DIR", "catalog/voices")),
            characters_dir=Path(
                getenv_str("CAI_PATHS__CHARACTERS_DIR", "configs/characters")
            ),
            characters_index=Path(
                getenv_str(
                    "CAI_PATHS__CHARACTERS_INDEX", "configs/characters/index.yaml"
                )
            ),
        )

        # New feature configurations
        supported_languages_env = os.getenv("CAI_LANGUAGE_SUPPORT__SUPPORTED_LANGUAGES")

        supported_languages = (
            [lang.strip() for lang in supported_languages_env.split(",")]
            if supported_languages_env
            else ["en", "es", "fr", "de", "zh", "ja", "ko", "ar"]
        )
        language_support = LanguageSupportConfig(
            default_language=getenv_str("CAI_LANGUAGE_SUPPORT__DEFAULT_LANGUAGE", "en"),
            auto_detection_enabled=getenv_bool(
                "CAI_LANGUAGE_SUPPORT__AUTO_DETECTION_ENABLED", True
            ),
            supported_languages=supported_languages,
            fallback_language=getenv_str(
                "CAI_LANGUAGE_SUPPORT__FALLBACK_LANGUAGE", "en"
            ),
            cultural_adaptation=getenv_bool(
                "CAI_LANGUAGE_SUPPORT__CULTURAL_ADAPTATION", True
            ),
            language_pack_dir=Path(
                getenv_str(
                    "CAI_LANGUAGE_SUPPORT__LANGUAGE_PACK_DIR", "configs/language_packs"
                )
            ),
        )

        tts_languages_env = os.getenv("CAI_MULTILINGUAL_AUDIO__TTS_LANGUAGES")
        tts_languages = (
            [lang.strip() for lang in tts_languages_env.split(",")]
            if tts_languages_env
            else ["en", "es", "fr", "de", "zh", "ja", "ko", "ar"]
        )
        stt_languages_env = os.getenv("CAI_MULTILINGUAL_AUDIO__STT_LANGUAGES")
        stt_languages = (
            [lang.strip() for lang in stt_languages_env.split(",")]
            if stt_languages_env
            else ["en", "es", "fr", "de", "zh", "ja", "ko", "ar"]
        )
        multilingual_audio = MultilingualAudioConfig(
            tts_languages=tts_languages,
            stt_languages=stt_languages,
            voice_adaptation=getenv_bool(
                "CAI_MULTILINGUAL_AUDIO__VOICE_ADAPTATION", True
            ),
            cultural_voice_characteristics=getenv_bool(
                "CAI_MULTILINGUAL_AUDIO__CULTURAL_VOICE_CHARACTERISTICS", True
            ),
            auto_language_detection=getenv_bool(
                "CAI_MULTILINGUAL_AUDIO__AUTO_LANGUAGE_DETECTION", True
            ),
            language_confidence_threshold=getenv_float(
                "CAI_MULTILINGUAL_AUDIO__LANGUAGE_CONFIDENCE_THRESHOLD", 0.7
            ),
        )

        personalization = PersonalizationConfig(
            enabled=getenv_bool("CAI_PERSONALIZATION__ENABLED", True),
            learning_rate=getenv_float("CAI_PERSONALIZATION__LEARNING_RATE", 0.1),
            max_preferences=getenv_int("CAI_PERSONALIZATION__MAX_PREFERENCES", 50),
            privacy_mode=getenv_str("CAI_PERSONALIZATION__PRIVACY_MODE", "strict"),
            data_retention_days=getenv_int(
                "CAI_PERSONALIZATION__DATA_RETENTION_DAYS", 30
            ),
            adaptive_conversation=getenv_bool(
                "CAI_PERSONALIZATION__ADAPTIVE_CONVERSATION", True
            ),
            character_recommendations=getenv_bool(
                "CAI_PERSONALIZATION__CHARACTER_RECOMMENDATIONS", True
            ),
            preference_learning_enabled=getenv_bool(
                "CAI_PERSONALIZATION__PREFERENCE_LEARNING_ENABLED", True
            ),
            style_adaptation=getenv_bool("CAI_PERSONALIZATION__STYLE_ADAPTATION", True),
        )

        parental_controls = ParentalControlsConfig(
            enabled=getenv_bool("CAI_PARENTAL_CONTROLS__ENABLED", True),
            default_safety_level=getenv_str(
                "CAI_PARENTAL_CONTROLS__DEFAULT_SAFETY_LEVEL", "moderate"
            ),
            alert_threshold=getenv_float("CAI_PARENTAL_CONTROLS__ALERT_THRESHOLD", 0.7),
            time_limit_default=getenv_int(
                "CAI_PARENTAL_CONTROLS__TIME_LIMIT_DEFAULT", 60
            ),
            content_filtering=getenv_bool(
                "CAI_PARENTAL_CONTROLS__CONTENT_FILTERING", True
            ),
            usage_monitoring=getenv_bool(
                "CAI_PARENTAL_CONTROLS__USAGE_MONITORING", True
            ),
            parental_dashboard=getenv_bool(
                "CAI_PARENTAL_CONTROLS__PARENTAL_DASHBOARD", True
            ),
            safety_alerts_enabled=getenv_bool(
                "CAI_PARENTAL_CONTROLS__SAFETY_ALERTS_ENABLED", True
            ),
            data_retention_days=getenv_int(
                "CAI_PARENTAL_CONTROLS__DATA_RETENTION_DAYS", 30
            ),
        )

        streaming = StreamingConfig(
            token_generation_delay=getenv_float(
                "CAI_STREAMING__TOKEN_GENERATION_DELAY", 0.05
            ),
            placeholder_delay=getenv_float("CAI_STREAMING__PLACEHOLDER_DELAY", 0.1),
            connection_timeout_ms=getenv_int(
                "CAI_STREAMING__CONNECTION_TIMEOUT_MS", 30000
            ),
            max_response_time_ms=getenv_int(
                "CAI_STREAMING__MAX_RESPONSE_TIME_MS", 30000
            ),
            simulation_delays=getenv_bool("CAI_STREAMING__SIMULATION_DELAYS", False),
        )

        return cls(
            environment=env,
            debug=debug,
            gpu=gpu,
            models=models,
            monitoring=monitoring,
            api=api,
            runtime=runtime,
            interaction=interaction,
            tts=tts,
            safety=safety,
            paths=paths,
            language_support=language_support,
            multilingual_audio=multilingual_audio,
            personalization=personalization,
            parental_controls=parental_controls,
            streaming=streaming,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "gpu": {
                "device": self.gpu.device,
                "memory_limit": self.gpu.memory_limit,
                "precision": self.gpu.precision,
                "batch_size": self.gpu.batch_size,
                "quantization": self.gpu.quantization,
                "gradient_checkpointing": self.gpu.gradient_checkpointing,
                "torch_compile": self.gpu.torch_compile,
                "cuda_graphs": self.gpu.cuda_graphs,
            },
            "models": {
                "wav2vec2_model": self.models.wav2vec2_model,
                "coqui_model": self.models.coqui_model,
                "llama_model": self.models.llama_model,
                "llama_quantization": self.models.llama_quantization,
                "clap_model": self.models.clap_model,
                "bark_model": self.models.bark_model,
                "so_vits_model": self.models.so_vits_model,
                "max_audio_length": self.models.max_audio_length,
                "sample_rate": self.models.sample_rate,
                "max_concurrent_requests": self.models.max_concurrent_requests,
                "request_timeout": self.models.request_timeout,
            },
            "monitoring": {
                "enabled": self.monitoring.enabled,
                "metrics_port": self.monitoring.metrics_port,
                "prometheus_enabled": self.monitoring.prometheus_enabled,
                "log_level": self.monitoring.log_level,
                "structured_logging": self.monitoring.structured_logging,
                "track_latency": self.monitoring.track_latency,
                "track_memory": self.monitoring.track_memory,
                "track_quality": self.monitoring.track_quality,
                "track_errors": self.monitoring.track_errors,
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "rate_limit": self.api.rate_limit,
                "max_request_size": self.api.max_request_size,
                "cors_origins": self.api.cors_origins,
                "websocket_timeout": self.api.websocket_timeout,
            },
            "data_dir": str(self.data_dir),
            "models_dir": str(self.models_dir),
            "cache_dir": str(self.cache_dir),
            "logs_dir": str(self.logs_dir),
            "use_free_models_only": self.use_free_models_only,
            "fallback_to_api": self.fallback_to_api,
        }

    def save(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
