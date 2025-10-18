"""
Configuration management for Character AI Platform.

Provides a clean public API for all configuration components.
"""

# Base infrastructure
from .base import DEFAULT_COQUI_MODEL, Environment

# Hardware configuration
from .hardware import GPUConfig

# Main configuration class
from .main import Config

# Model configuration
from .models import (
    ConversationalAIConfig,
    ModelConfig,
    MultimodalFusionConfig,
    RepresentationLearningConfig,
    VoiceCloningConfig,
)

# Runtime configuration
from .runtime import (
    APIConfig,
    InteractionConfig,
    LanguageSupportConfig,
    MemoryConfig,
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
    TTSStreamingConfig,
)

# Public API
__all__ = [
    # Main class
    "Config",
    # Base
    "Environment",
    "DEFAULT_COQUI_MODEL",
    # Hardware
    "GPUConfig",
    # Models
    "ModelConfig",
    "RepresentationLearningConfig",
    "VoiceCloningConfig",
    "MultimodalFusionConfig",
    "ConversationalAIConfig",
    # Runtime
    "APIConfig",
    "InteractionConfig",
    "LanguageSupportConfig",
    "MemoryConfig",
    "MonitoringConfig",
    "MultilingualAudioConfig",
    "ParentalControlsConfig",
    "PathsConfig",
    "PersonalizationConfig",
    "RuntimeConfig",
    "SafetyConfig",
    "SecurityConfig",
    "StreamingConfig",
    "TTSConfig",
    "TTSStreamingConfig",
]
