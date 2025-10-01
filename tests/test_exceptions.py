"""
Tests for custom exceptions.
"""

from character_ai.core.exceptions import (
    AudioProcessingError,
    ConfigurationError,
    ModelError,
    ValidationError,
    VoiceAIError,
)


class TestExceptions:
    """Test custom exception classes."""

    def test_voice_ai_error(self):
        """Test VoiceAIError."""
        error = VoiceAIError("VoiceAI failed")
        assert str(error) == "[VoiceAI] VoiceAI failed"

    def test_voice_ai_error_with_component(self):
        """Test VoiceAIError with component."""
        error = VoiceAIError("Test error", component="TestComponent")
        assert str(error) == "[TestComponent] Test error"

    def test_voice_ai_error_with_error_code(self):
        """Test VoiceAIError with error code."""
        error = VoiceAIError("Test error", error_code="ERR001")
        assert str(error) == "[ERR001] [VoiceAI] Test error"

    def test_voice_ai_error_to_dict(self):
        """Test VoiceAIError to_dict method."""
        error = VoiceAIError(
            "Test error", error_code="ERR001", component="TestComponent"
        )
        error_dict = error.to_dict()

        assert error_dict["error_type"] == "VoiceAIError"
        assert error_dict["message"] == "Test error"
        assert error_dict["error_code"] == "ERR001"
        assert error_dict["component"] == "TestComponent"

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Config invalid")
        assert str(error) == "[VoiceAI] Config invalid"

    def test_model_error(self):
        """Test ModelError."""
        error = ModelError("Model failed", component="TestModel")
        assert str(error) == "[TestModel] Model failed"

    def test_audio_processing_error(self):
        """Test AudioProcessingError."""
        error = AudioProcessingError("Audio processing failed", component="TestAudio")
        assert str(error) == "[TestAudio] Audio processing failed"

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("test_field", "test_value", "invalid_format")
        assert "Validation failed" in str(error)
        assert error.details["value"] == "test_value"
        assert error.details["reason"] == "invalid_format"
        assert error.details["field"] == "test_field"

    def test_exception_inheritance(self):
        """Test exception inheritance hierarchy."""
        # Test that specific exceptions inherit from base exceptions
        assert issubclass(ConfigurationError, VoiceAIError)
        assert issubclass(ModelError, VoiceAIError)
        assert issubclass(AudioProcessingError, VoiceAIError)
        assert issubclass(ValidationError, VoiceAIError)

    def test_exception_with_cause(self):
        """Test exceptions with cause."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            error = AudioProcessingError("Wrapped error", component="TestComponent")
            error.__cause__ = e

            assert str(error) == "[TestComponent] Wrapped error"
            assert error.__cause__ is e
