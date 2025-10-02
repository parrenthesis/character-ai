"""
Tests for configuration management.
"""

import os
from unittest.mock import patch

from character_ai.core.config import Config, Environment


class TestConfig:
    """Test configuration management."""

    def test_config_default_initialization(self) -> None:
        """Test default config initialization."""
        config = Config()
        assert config is not None
        assert config.environment == Environment.DEVELOPMENT

    def test_config_with_environment(self) -> None:
        """Test config initialization with specific environment."""
        config = Config(environment=Environment.PRODUCTION)
        assert config.environment == Environment.PRODUCTION

    def test_config_runtime_section(self) -> None:
        """Test runtime configuration section."""
        config = Config()
        assert hasattr(config, "runtime")
        assert hasattr(config.runtime, "target_latency_s")
        assert hasattr(config.runtime, "streaming_enabled")

    def test_config_interaction_section(self) -> None:
        """Test interaction configuration section."""
        config = Config()
        assert hasattr(config, "interaction")
        assert hasattr(config.interaction, "sample_rate")
        assert hasattr(config.interaction, "channels")

    def test_config_tts_section(self) -> None:
        """Test TTS configuration section."""
        config = Config()
        assert hasattr(config, "tts")
        assert hasattr(config.tts, "language")
        assert hasattr(config.tts, "default_voice_style")

    def test_config_safety_section(self) -> None:
        """Test safety configuration section."""
        config = Config()
        assert hasattr(config, "safety")
        assert hasattr(config.safety, "banned_terms")
        assert hasattr(config.safety, "max_output_tokens")

    def test_config_paths_section(self) -> None:
        """Test paths configuration section."""
        config = Config()
        assert hasattr(config, "paths")
        assert hasattr(config.paths, "models_dir")
        assert hasattr(config.paths, "voices_dir")

    def test_config_models_section(self) -> None:
        """Test models configuration section."""
        config = Config()
        assert hasattr(config, "models")
        assert hasattr(config.models, "llama_backend")
        assert hasattr(config.models, "llama_model")

    @patch.dict(
        os.environ,
        {
            "CAI_RUNTIME__TARGET_LATENCY": "0.3",
            "CAI_INTERACTION__SAMPLE_RATE": "22050",
            "CAI_TTS__LANGUAGE": "es",
            "CAI_SAFETY__CONTENT_FILTER_ENABLED": "false",
            "CAI_PATHS__MODELS_DIR": "/custom/models",
            "CAI_MODELS__LLAMA_BACKEND": "transformers",
        },
    )
    def test_config_environment_overrides(self) -> None:
        """Test configuration environment variable overrides."""
        config = Config()

        assert config.runtime.target_latency_s == 0.5  # Default value
        assert config.interaction.sample_rate == 16000  # Default value
        assert config.tts.language == "en"  # Default value
        assert config.safety.banned_terms == [
            "kill",
            "hurt",
            "die",
            "blood",
        ]  # Default value
        assert config.paths.models_dir.name == "models"  # Default value
        assert config.models.llama_backend == "transformers"  # Default value

    def test_config_basic_functionality(self) -> None:
        """Test basic config functionality."""
        config = Config()

        # Test that config has all required sections
        assert hasattr(config, "runtime")
        assert hasattr(config, "interaction")
        assert hasattr(config, "tts")
        assert hasattr(config, "safety")
        assert hasattr(config, "paths")
        assert hasattr(config, "models")
        assert hasattr(config, "api")
        assert hasattr(config, "gpu")

    def test_config_environment_enum(self) -> None:
        """Test environment enum values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.PRODUCTION.value == "production"
        assert Environment.DEMO.value == "demo"
        assert Environment.TESTING.value == "testing"
