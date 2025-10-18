"""
Centralized error messages and fallback responses for services.

Provides consistent error handling and user-friendly fallback messages
across STT, LLM, and TTS services.
"""


class ServiceErrorMessages:
    """Centralized error messages for services."""

    # STT Service Messages
    STT_NO_AUDIO = "I didn't catch that. Could you please repeat?"
    STT_TRANSCRIPTION_FAILED = "I'm having trouble hearing you right now."
    STT_PROCESSOR_ERROR = "Audio processing is temporarily unavailable."

    # LLM Service Messages
    LLM_NO_RESPONSE = "Hi! I'm {character_name}..."
    LLM_GENERATION_FAILED = "Hello! I'm {character_name}..."
    LLM_PROCESSOR_ERROR = "I'm having trouble thinking right now."

    # TTS Service Messages
    TTS_SYNTHESIS_FAILED = "I'm having trouble speaking right now."
    TTS_NO_AUDIO = "No audio was generated."
    TTS_PROCESSOR_ERROR = "Voice synthesis is temporarily unavailable."

    # Generic Service Messages
    SERVICE_INITIALIZATION_FAILED = "Service initialization failed"
    SERVICE_HEALTH_CHECK_FAILED = "Service health check failed"
    PROCESSOR_CREATION_FAILED = "Failed to initialize {service_type} processor"

    @classmethod
    def get_stt_fallback(cls, has_audio: bool = True) -> str:
        """Get appropriate STT fallback message."""
        return cls.STT_NO_AUDIO if has_audio else cls.STT_TRANSCRIPTION_FAILED

    @classmethod
    def get_llm_fallback(
        cls, character_name: str, is_generation_error: bool = False
    ) -> str:
        """Get appropriate LLM fallback message."""
        template = (
            cls.LLM_GENERATION_FAILED if is_generation_error else cls.LLM_NO_RESPONSE
        )
        return template.format(character_name=character_name)

    @classmethod
    def get_tts_fallback(cls) -> str:
        """Get appropriate TTS fallback message."""
        return cls.TTS_SYNTHESIS_FAILED

    @classmethod
    def get_processor_error(cls, service_type: str) -> str:
        """Get processor initialization error message."""
        return cls.PROCESSOR_CREATION_FAILED.format(service_type=service_type.upper())


# Convenience constants for direct import
STT_NO_AUDIO = ServiceErrorMessages.STT_NO_AUDIO
STT_TRANSCRIPTION_FAILED = ServiceErrorMessages.STT_TRANSCRIPTION_FAILED
LLM_NO_RESPONSE = ServiceErrorMessages.LLM_NO_RESPONSE
LLM_GENERATION_FAILED = ServiceErrorMessages.LLM_GENERATION_FAILED
TTS_SYNTHESIS_FAILED = ServiceErrorMessages.TTS_SYNTHESIS_FAILED
PROCESSOR_CREATION_FAILED = ServiceErrorMessages.PROCESSOR_CREATION_FAILED
