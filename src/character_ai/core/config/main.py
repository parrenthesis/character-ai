"""
Main configuration class for Character AI Platform.

Contains the main Config class that orchestrates all configuration components.
"""

# CRITICAL: Import torch_init FIRST to set environment variables before any torch imports
# isort: off
from .. import torch_init  # noqa: F401

# isort: on

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from ..llm.config import LLMConfigService
from .base import DEFAULT_COQUI_MODEL, Environment
from .hardware import GPUConfig
from .models import ModelConfig
from .runtime import (
    APIConfig,
    InteractionConfig,
    LanguageSupportConfig,
    MemoryConfig,
    MemorySystemConfig,
    MonitoringConfig,
    MultilingualAudioConfig,
    ParentalControlsConfig,
    PathsConfig,
    PersonalizationConfig,
    RuntimeConfig,
    SafetyConfig,
    SecurityConfig,
    StreamingConfig,
    TTSConfig,
)
from .yaml_loader import YAMLConfigLoader

logger = logging.getLogger(__name__)


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

    # Monitoring configuration
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Runtime and behavior
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    memory_system: MemorySystemConfig = field(default_factory=MemorySystemConfig)
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

    # Internal: store runtime YAML data for LLM configuration
    _runtime_yaml_data: Optional[Dict[str, Any]] = field(default=None, init=False)

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
            runtime_path = Path("configs/runtime.yaml")
            if not runtime_path.exists():
                return

            data = YAMLConfigLoader.load_yaml(runtime_path)

            # Load model_registry from runtime.yaml
            if "model_registry" in data:
                self.runtime.model_registry = data["model_registry"]

            # Load TTS model from runtime.yaml
            tts_section = data.get("tts", {}) or {}
            if tts_section.get("model_name"):
                self.models.coqui_model = str(tts_section["model_name"])

            # Load TTS streaming configuration
            streaming_section = tts_section.get("streaming", {}) or {}
            if streaming_section:
                if "enabled" in streaming_section:
                    self.tts.streaming.enabled = bool(streaming_section["enabled"])
                if "method" in streaming_section:
                    self.tts.streaming.method = str(streaming_section["method"])
                if "sentence_chunking" in streaming_section:
                    self.tts.streaming.sentence_chunking = bool(
                        streaming_section["sentence_chunking"]
                    )
                if "fallback_to_blocking" in streaming_section:
                    self.tts.streaming.fallback_to_blocking = bool(
                        streaming_section["fallback_to_blocking"]
                    )
                logger.debug(
                    f"Loaded TTS streaming config: enabled={self.tts.streaming.enabled}, method={self.tts.streaming.method}"
                )

            # Load LLM model from runtime.yaml
            models_section = data.get("models", {}) or {}
            if models_section.get("llama_model_name"):
                # Auto-discover the GGUF file based on model name
                model_name = models_section["llama_model_name"]
                gguf_path = self._find_gguf_file(model_name)
                if gguf_path:
                    self.models.llama_gguf_path = str(gguf_path)
                    logger.info(f"Auto-discovered GGUF file: {gguf_path}")
                else:
                    logger.warning(f"Could not find GGUF file for model: {model_name}")

            # Load LLM context window size
            if models_section.get("llama_n_ctx"):
                self.models.llama_n_ctx = int(models_section["llama_n_ctx"])
                logger.info(
                    f"Loaded llama_n_ctx from runtime.yaml: {self.models.llama_n_ctx}"
                )

            # Store the raw YAML data for LLM configuration
            self._runtime_yaml_data = data

        except Exception as e:
            # Non-fatal - just log and continue with defaults
            logger.debug(f"Could not load runtime.yaml: {e}")
            self._runtime_yaml_data = {}

    def create_llm_config_manager(self) -> LLMConfigService:
        """Create LLMConfigManager from this Config instance, avoiding duplicate YAML loading."""

        if self._runtime_yaml_data:
            # Use the already-loaded YAML data
            llm_manager = LLMConfigService()
            llm_manager._load_llm_from_yaml_data(self._runtime_yaml_data)
            return llm_manager
        else:
            # Fallback to the class method if no YAML data available
            return LLMConfigService.from_config(self)

    def _find_gguf_file(self, model_name: str) -> Optional[Path]:
        """Find the GGUF file for a given model name."""
        try:
            # Look in the models directory
            models_dir = Path("models/llm")
            if not models_dir.exists():
                return None

            # Try different naming patterns
            patterns = [
                f"{model_name}-Q4_K_M.gguf",
                f"{model_name}-q4_k_m.gguf",
                f"{model_name}.gguf",
                f"{model_name.lower()}-q4_k_m.gguf",
                f"{model_name.lower()}-Q4_K_M.gguf",
            ]

            for pattern in patterns:
                gguf_file = models_dir / pattern
                if gguf_file.exists():
                    return gguf_file

            # If no exact match, look for files containing the model name
            for gguf_file in models_dir.glob("*.gguf"):
                if model_name.lower() in gguf_file.name.lower():
                    return gguf_file

            return None
        except Exception as e:
            logger.debug(f"Error finding GGUF file for {model_name}: {e}")
            return None

    def _apply_cpu_limiting(self) -> None:
        """Apply CPU limiting for development/testing."""
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
        else:
            os.environ["OMP_NUM_THREADS"] = str(self.max_cpu_threads)

        # Set NumExpr threading
        os.environ["NUMEXPR_NUM_THREADS"] = str(self.max_cpu_threads)

        # Set PyTorch threading
        if self.torch_threads is not None:
            os.environ["TORCH_NUM_THREADS"] = str(self.torch_threads)
        else:
            os.environ["TORCH_NUM_THREADS"] = str(self.max_cpu_threads)

        # Also set PyTorch threads at runtime if available
        try:
            import torch

            torch.set_num_threads(self.torch_threads or self.max_cpu_threads or 1)
        except (ImportError, AttributeError):
            pass

        logger.info(f"Applied CPU limiting: {self.max_cpu_threads} threads")

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        data = YAMLConfigLoader.load_yaml(config_path)

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
        memory_system_data = data.get("memory_system", {})
        paths_data = data.get("paths", {})

        # Add model_registry to runtime_data
        if "model_registry" in data:
            runtime_data["model_registry"] = data["model_registry"]

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
            memory_system=MemorySystemConfig(**memory_system_data),
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
            llama_n_ctx=getenv_int("CAI_MODELS__LLAMA_N_CTX", 2048),
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
            sample_rate=getenv_int("CAI_INTERACTION__SAMPLE_RATE", 16000),
            channels=getenv_int("CAI_INTERACTION__CHANNELS", 1),
        )

        tts = TTSConfig(
            language=getenv_str("CAI_TTS__LANGUAGE", "en"),
            default_voice_style=getenv_str("CAI_TTS__DEFAULT_VOICE_STYLE", "neutral"),
            compute_voice_embedding_on_ingest=getenv_bool(
                "CAI_TTS__COMPUTE_VOICE_EMBEDDING_ON_INGEST", False
            ),
        )

        safety = SafetyConfig(
            max_output_tokens=getenv_int("CAI_SAFETY__MAX_OUTPUT_TOKENS", 64),
            enable_toxicity_classifier=getenv_bool(
                "CAI_SAFETY__ENABLE_TOXICITY_CLASSIFIER", True
            ),
            enable_pii_detection=getenv_bool("CAI_SAFETY__ENABLE_PII_DETECTION", True),
            safety_confidence_threshold=getenv_float(
                "CAI_SAFETY__SAFETY_CONFIDENCE_THRESHOLD", 0.7
            ),
            log_safety_violations=getenv_bool(
                "CAI_SAFETY__LOG_SAFETY_VIOLATIONS", True
            ),
        )

        paths = PathsConfig()

        language_support = LanguageSupportConfig(
            default_language=getenv_str("CAI_LANGUAGE_SUPPORT__DEFAULT_LANGUAGE", "en"),
            auto_detection_enabled=getenv_bool(
                "CAI_LANGUAGE_SUPPORT__AUTO_DETECTION_ENABLED", True
            ),
            fallback_language=getenv_str(
                "CAI_LANGUAGE_SUPPORT__FALLBACK_LANGUAGE", "en"
            ),
            cultural_adaptation=getenv_bool(
                "CAI_LANGUAGE_SUPPORT__CULTURAL_ADAPTATION", True
            ),
        )

        multilingual_audio = MultilingualAudioConfig(
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
