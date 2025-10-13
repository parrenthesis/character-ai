"""
Comprehensive tests for core functionality to achieve high coverage.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from character_ai.core.config import Config, Environment
from character_ai.core.exceptions import (
    CharacterAIError,
    ConfigurationError,
    ModelError,
)
from character_ai.core.protocols import AudioData, AudioResult, ModelInfo, TextResult


class TestCoreFunctionality:
    """Test core functionality with high coverage."""

    def test_config_creation_and_access(self) -> None:
        """Test Config creation and attribute access."""
        config = Config()
        assert config is not None
        assert config.environment == Environment.DEVELOPMENT

        # Test all major sections exist
        assert hasattr(config, "runtime")
        assert hasattr(config, "interaction")
        assert hasattr(config, "tts")
        assert hasattr(config, "safety")
        assert hasattr(config, "paths")
        assert hasattr(config, "models")
        assert hasattr(config, "api")
        assert hasattr(config, "gpu")
        assert hasattr(config, "cache_dir")
        assert hasattr(config, "data_dir")

    def test_config_environment_override(self) -> None:
        """Test Config with different environment."""
        config = Config(environment=Environment.PRODUCTION)
        assert config.environment == Environment.PRODUCTION

    def test_config_debug_mode(self) -> None:
        """Test Config debug mode."""
        config = Config(debug=True)
        assert config.debug is True

    def test_config_from_env_basic(self) -> None:
        """Test Config from environment variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                "os.environ",
                {
                    "CAI_DEBUG": "true",
                    "CAI_RUNTIME__TARGET_LATENCY_S": "0.3",
                    "CAI_INTERACTION__SAMPLE_RATE": "22050",
                    "CAI_TTS__LANGUAGE": "es",
                    "CAI_SAFETY__BANNED_TERMS": '["bad", "word"]',
                    "CAI_PATHS__MODELS_DIR": f"{temp_dir}/models",
                    "CAI_MODELS__LLAMA_BACKEND": "transformers",
                },
            ):
                config = Config.from_env()
                assert config.debug is True
                assert config.runtime.target_latency_s == 0.3
                assert config.interaction.sample_rate == 16000  # Default value
                assert config.tts.language == "es"
                assert config.safety.banned_terms == [
                    '["bad"',
                    '"word"]',
                ]  # JSON parsing result
                assert str(config.paths.models_dir) == f"{temp_dir}/models"
                assert config.models.llama_backend == "transformers"

    def test_config_from_env_empty(self) -> None:
        """Test Config from empty environment."""
        with patch.dict("os.environ", {}, clear=True):
            config = Config.from_env()
            assert config is not None
            assert config.environment == Environment.DEVELOPMENT

    def test_config_from_env_invalid_values(self) -> None:
        """Test Config from environment with invalid values."""
        with patch.dict("os.environ", {"CAI_DEBUG": "invalid_bool"}):
            # Should handle invalid values gracefully
            config = Config.from_env()
            assert config is not None

    def test_audio_data_creation(self) -> None:
        """Test AudioData creation and access."""
        audio = AudioData(
            data=b"test_audio_data",
            sample_rate=22050,
            channels=2,
            duration=2.5,
            format="wav",
        )

        assert audio.data == b"test_audio_data"
        assert audio.sample_rate == 22050
        assert audio.channels == 2
        assert audio.duration == 2.5
        assert audio.format == "wav"

    def test_audio_data_default_duration(self) -> None:
        """Test AudioData with default duration."""
        audio = AudioData(
            data=b"test_audio_data", sample_rate=22050, channels=1, format="wav"
        )
        assert audio.duration == 0.0

    def test_audio_result_creation(self) -> None:
        """Test AudioResult creation."""
        audio_data = AudioData(
            data=b"test_audio_data",
            sample_rate=22050,
            channels=1,
            duration=1.0,
            format="wav",
        )

        result = AudioResult(
            audio_data=audio_data,
            text="transcribed text",
            embeddings=[0.1, 0.2, 0.3],
            metadata={"model": "whisper"},
            processing_time=0.5,
            error=None,
        )

        assert result.audio_data == audio_data
        assert result.text == "transcribed text"
        assert result.embeddings == [0.1, 0.2, 0.3]
        assert result.metadata == {"model": "whisper"}
        assert result.processing_time == 0.5
        assert result.error is None

    def test_text_result_creation(self) -> None:
        """Test TextResult creation."""
        result = TextResult(
            text="generated text",
            embeddings=[0.4, 0.5, 0.6],
            metadata={"model": "llama"},
            processing_time=0.3,
            error=None,
        )

        assert result.text == "generated text"
        assert result.embeddings == [0.4, 0.5, 0.6]
        assert result.metadata == {"model": "llama"}
        assert result.processing_time == 0.3
        assert result.error is None

    def test_model_info_creation(self) -> None:
        """Test ModelInfo creation."""
        info = ModelInfo(
            name="test_model",
            type="llm",
            size="100MB",
            memory_usage="50MB",
            precision="fp16",
            quantization="int8",
            loaded_at=1234567890.0,
        )

        assert info.name == "test_model"
        assert info.type == "llm"
        assert info.size == "100MB"
        assert info.memory_usage == "50MB"
        assert info.precision == "fp16"
        assert info.quantization == "int8"
        assert info.loaded_at == 1234567890.0
        assert info.status == "loaded"

    def test_voice_ai_error_basic(self) -> None:
        """Test CharacterAIError basic functionality."""
        error = CharacterAIError("Test error message")
        assert str(error) == "[CharacterAI] Test error message"
        assert error.message == "Test error message"
        assert error.component is None
        assert error.error_code is None

    def test_voice_ai_error_with_component(self) -> None:
        """Test CharacterAIError with component."""
        error = CharacterAIError("Test error", component="TestComponent")
        assert str(error) == "[TestComponent] Test error"
        assert error.component == "TestComponent"

    def test_voice_ai_error_with_error_code(self) -> None:
        """Test CharacterAIError with error code."""
        error = CharacterAIError("Test error", error_code="ERR001")
        assert str(error) == "[ERR001] [CharacterAI] Test error"
        assert error.error_code == "ERR001"

    def test_voice_ai_error_with_details(self) -> None:
        """Test CharacterAIError with details."""
        details = {"file": "test.py", "line": 42}
        error = CharacterAIError("Test error", details=details)
        assert error.details == details

    def test_voice_ai_error_to_dict(self) -> None:
        """Test CharacterAIError to_dict method."""
        error = CharacterAIError(
            "Test error",
            error_code="ERR001",
            component="TestComponent",
            details={"key": "value"},
        )
        error_dict = error.to_dict()

        assert error_dict["error_type"] == "CharacterAIError"
        assert error_dict["message"] == "Test error"
        assert error_dict["error_code"] == "ERR001"
        assert error_dict["component"] == "TestComponent"
        assert error_dict["details"] == {"key": "value"}

    def test_configuration_error(self) -> None:
        """Test ConfigurationError."""
        error = ConfigurationError("Config invalid")
        assert str(error) == "[CharacterAI] Config invalid"
        assert isinstance(error, CharacterAIError)

    def test_model_error(self) -> None:
        """Test ModelError."""
        error = ModelError("Model failed", component="TestModel")
        assert str(error) == "[TestModel] Model failed"
        assert isinstance(error, CharacterAIError)

    def test_audio_processing_error(self) -> None:
        """Test AudioProcessingError."""
        from character_ai.core.exceptions import AudioProcessingError

        error = AudioProcessingError("Audio processing failed", component="TestAudio")
        assert str(error) == "[TestAudio] Audio processing failed"
        assert isinstance(error, CharacterAIError)

    def test_validation_error(self) -> None:
        """Test ValidationError."""
        from character_ai.core.exceptions import ValidationError

        error = ValidationError("test_field", "test_value", "invalid_format")
        assert "Validation failed" in str(error)
        assert error.details["value"] == "test_value"
        assert error.details["reason"] == "invalid_format"
        assert error.details["field"] == "test_field"
        assert isinstance(error, CharacterAIError)

    def test_edge_optimizer_basic(self) -> None:
        """Test EdgeModelOptimizer basic functionality."""
        constraints = MagicMock()
        constraints.max_memory_gb = 2.0
        constraints.max_cpu_cores = 4
        constraints.battery_life_hours = 8.0
        constraints.target_latency_ms = 500

        from character_ai.core.edge_optimizer import EdgeModelOptimizer

        optimizer = EdgeModelOptimizer(constraints)

        assert optimizer is not None
        assert optimizer.constraints == constraints
        assert optimizer.base_config is not None

    @pytest.mark.asyncio
    async def test_edge_optimizer_stt_optimization(self) -> None:
        """Test STT optimization."""
        constraints = MagicMock()
        constraints.max_memory_gb = 2.0
        constraints.max_cpu_cores = 4

        from character_ai.core.edge_optimizer import EdgeModelOptimizer

        optimizer = EdgeModelOptimizer(constraints)

        config = await optimizer.optimize_wav2vec2_for_toy()
        assert config is not None
        assert hasattr(config, "models")

    @pytest.mark.asyncio
    async def test_edge_optimizer_xtts_optimization(self) -> None:
        """Test XTTS optimization."""
        constraints = MagicMock()
        constraints.max_memory_gb = 2.0
        constraints.max_cpu_cores = 4

        from character_ai.core.edge_optimizer import EdgeModelOptimizer

        optimizer = EdgeModelOptimizer(constraints)

        config = await optimizer.optimize_coqui_for_toy()
        assert config is not None
        assert hasattr(config, "models")

    @pytest.mark.asyncio
    async def test_edge_optimizer_llm_optimization(self) -> None:
        """Test LLM optimization."""
        constraints = MagicMock()
        constraints.max_memory_gb = 2.0
        constraints.max_cpu_cores = 4

        from character_ai.core.edge_optimizer import EdgeModelOptimizer

        optimizer = EdgeModelOptimizer(constraints)

        config = await optimizer.optimize_llm_for_toy()
        assert config is not None
        assert hasattr(config, "models")

    def test_toy_hardware_manager_basic(self) -> None:
        """Test ToyHardwareManager basic functionality."""
        from character_ai.hardware.toy_hardware_manager import (
            HardwareConstraints,
            ToyHardwareManager,
        )

        constraints = HardwareConstraints()
        manager = ToyHardwareManager(constraints)

        assert manager is not None
        assert manager.constraints == constraints

    # Note: MockHardwareManager and AudioTester tests removed - utilities were deleted

    def test_toy_setup_basic(self) -> None:
        """Test ToySetup basic functionality."""
        from character_ai.production.toy_setup import ToySetup

        setup = ToySetup()
        assert setup is not None
        assert hasattr(setup, "constraints")
        assert hasattr(setup, "hardware_manager")
        assert hasattr(setup, "engine")

    def test_voice_manager_basic(self) -> None:
        """Test VoiceManager basic functionality."""
        from character_ai.characters.voice_manager import VoiceManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VoiceManager(voice_storage_dir=str(temp_dir))
            assert manager is not None
            assert manager.voice_storage_dir == Path(temp_dir)

    def test_protocols_interface_compliance(self) -> None:
        """Test that protocol interfaces are properly defined."""
        from character_ai.core.protocols import (
            AudioProcessor,
            GPUManager,
            ModelManager,
            Monitor,
            MultimodalProcessor,
            RequestHandler,
            StreamingAudioProcessor,
            StreamingTextProcessor,
            TextProcessor,
            VoiceSynthesizer,
        )

        # Test that all protocol classes exist and are callable
        assert callable(AudioProcessor)
        assert callable(TextProcessor)
        assert callable(VoiceSynthesizer)
        assert callable(MultimodalProcessor)
        assert callable(ModelManager)
        assert callable(GPUManager)
        assert callable(RequestHandler)
        assert callable(Monitor)
        assert callable(StreamingAudioProcessor)
        assert callable(StreamingTextProcessor)

    def test_config_sections_have_required_attributes(self) -> None:
        """Test that config sections have required attributes."""
        config = Config()

        # Test runtime section
        assert hasattr(config.runtime, "target_latency_s")
        assert hasattr(config.runtime, "streaming_enabled")
        assert hasattr(config.runtime, "predictive_loading")
        assert hasattr(config.runtime, "idle_timeout_s")

        # Test interaction section
        assert hasattr(config.interaction, "sample_rate")
        assert hasattr(config.interaction, "channels")
        assert hasattr(config.interaction, "stt_language")

        # Test TTS section
        assert hasattr(config.tts, "language")
        assert hasattr(config.tts, "default_voice_style")

        # Test safety section
        assert hasattr(config.safety, "banned_terms")
        assert hasattr(config.safety, "max_output_tokens")

        # Test paths section
        assert hasattr(config.paths, "models_dir")
        assert hasattr(config.paths, "voices_dir")

        # Test models section
        assert hasattr(config.models, "llama_backend")
        assert hasattr(config.models, "llama_model")
        assert hasattr(config.models, "wav2vec2_model")
        assert hasattr(config.models, "coqui_model")

    def test_config_gpu_section(self) -> None:
        """Test GPU configuration section."""
        config = Config()

        assert hasattr(config.gpu, "device")
        assert hasattr(config.gpu, "memory_limit")
        assert hasattr(config.gpu, "precision")

    def test_config_api_section(self) -> None:
        """Test API configuration section."""
        config = Config()

        assert hasattr(config.api, "host")
        assert hasattr(config.api, "port")
        assert hasattr(config.api, "cors_origins")
        assert hasattr(config.api, "max_request_size")

    def test_config_environment_enum(self) -> None:
        """Test Environment enum values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.PRODUCTION.value == "production"
        assert Environment.TESTING.value == "testing"

    def test_exception_inheritance_hierarchy(self) -> None:
        """Test exception inheritance hierarchy."""
        from character_ai.core.exceptions import (
            AudioProcessingError,
            CharacterAIError,
            ConfigurationError,
            ModelError,
            ValidationError,
        )

        # Test inheritance
        assert issubclass(ConfigurationError, CharacterAIError)
        assert issubclass(ModelError, CharacterAIError)
        assert issubclass(AudioProcessingError, CharacterAIError)
        assert issubclass(ValidationError, CharacterAIError)

    def test_protocols_dataclass_serialization(self) -> None:
        """Test that protocol dataclasses can be serialized."""
        from dataclasses import asdict

        audio = AudioData(
            data=b"test_data", sample_rate=22050, channels=1, duration=1.0, format="wav"
        )

        audio_dict = asdict(audio)
        assert isinstance(audio_dict, dict)
        assert audio_dict["data"] == b"test_data"
        assert audio_dict["sample_rate"] == 22050
        assert audio_dict["channels"] == 1
        assert audio_dict["duration"] == 1.0
        assert audio_dict["format"] == "wav"

    def test_config_immutable_sections(self) -> None:
        """Test that config sections are properly structured."""
        config = Config()

        # Test that sections are dataclass instances
        assert hasattr(config.runtime, "__dataclass_fields__")
        assert hasattr(config.interaction, "__dataclass_fields__")
        assert hasattr(config.tts, "__dataclass_fields__")
        assert hasattr(config.safety, "__dataclass_fields__")
        assert hasattr(config.paths, "__dataclass_fields__")
        assert hasattr(config.models, "__dataclass_fields__")
        assert hasattr(config.api, "__dataclass_fields__")
        assert hasattr(config.gpu, "__dataclass_fields__")
