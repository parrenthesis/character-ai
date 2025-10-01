"""
Multi-language audio processing API endpoints.

Provides endpoints for language-aware TTS and STT processing with automatic
language detection, cultural adaptations, and voice characteristics.
"""

import io
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..core.language_support import LanguageCode
from ..core.multilingual_audio import (
    MultiLanguageAudioManager,
    get_multilingual_audio_manager,
)
from ..core.protocols import AudioData
from .security_deps import get_security_middleware

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/audio", tags=["Multi-language Audio"])


# Pydantic models for API
class TTSRequest(BaseModel):
    """Request model for TTS synthesis."""

    text: str = Field(..., description="Text to synthesize")
    language_code: Optional[str] = Field(
        None, description="Specific language code (auto-detect if not provided)"
    )
    voice_style: Optional[str] = Field(None, description="Voice style preference")
    cultural_adaptations: Optional[Dict[str, Any]] = Field(
        None, description="Custom cultural adaptations"
    )


class TTSResponse(BaseModel):
    """Response model for TTS synthesis."""

    audio_data: str = Field(..., description="Base64 encoded audio data")
    detected_language: str = Field(..., description="Detected language code")
    confidence: float = Field(..., description="Language detection confidence")
    cultural_adaptations: Dict[str, Any] = Field(
        ..., description="Applied cultural adaptations"
    )
    voice_characteristics: Dict[str, Any] = Field(
        ..., description="Voice characteristics used"
    )
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    sample_rate: int = Field(..., description="Audio sample rate")
    duration: float = Field(..., description="Audio duration in seconds")


class STTRequest(BaseModel):
    """Request model for STT transcription."""

    language_code: Optional[str] = Field(
        None, description="Specific language code (auto-detect if not provided)"
    )
    auto_detect: bool = Field(True, description="Enable automatic language detection")


class STTResponse(BaseModel):
    """Response model for STT transcription."""

    text: str = Field(..., description="Transcribed text")
    detected_language: str = Field(..., description="Detected language code")
    confidence: float = Field(..., description="Language detection confidence")
    cultural_adaptations: Dict[str, Any] = Field(
        ..., description="Cultural adaptations for detected language"
    )
    voice_characteristics: Dict[str, Any] = Field(
        ..., description="Voice characteristics for detected language"
    )
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    segments: int = Field(..., description="Number of audio segments processed")


class LanguageCapabilitiesResponse(BaseModel):
    """Response model for language capabilities."""

    language_code: str = Field(..., description="Language code")
    tts_supported: bool = Field(..., description="TTS support status")
    stt_supported: bool = Field(..., description="STT support status")
    tts_capabilities: Dict[str, Any] = Field(..., description="TTS capabilities")
    cultural_adaptations: Dict[str, Any] = Field(
        ..., description="Cultural adaptations"
    )
    voice_characteristics: Dict[str, Any] = Field(
        ..., description="Voice characteristics"
    )


class SupportedLanguagesResponse(BaseModel):
    """Response model for supported languages."""

    supported_languages: List[str] = Field(
        ..., description="List of supported language codes"
    )
    total_count: int = Field(..., description="Total number of supported languages")


# Global instances
_multilingual_audio_manager: Optional[MultiLanguageAudioManager] = None


def get_multilingual_audio_manager_instance() -> MultiLanguageAudioManager:
    """Get the multilingual audio manager instance."""
    global _multilingual_audio_manager
    if _multilingual_audio_manager is None:
        _multilingual_audio_manager = get_multilingual_audio_manager()
    return _multilingual_audio_manager


@router.post("/tts", response_model=TTSResponse)
async def synthesize_speech(
    request: TTSRequest, security: Any = Depends(get_security_middleware)
) -> TTSResponse:
    """Synthesize speech with language-specific adaptations."""
    try:
        manager = get_multilingual_audio_manager_instance()

        # Parse language code if provided
        language_code = None
        if request.language_code:
            try:
                language_code = LanguageCode(request.language_code)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid language code: {request.language_code}",
                )

        # Synthesize speech
        result = await manager.synthesize_speech(
            text=request.text,
            language_code=language_code,
            voice_style=request.voice_style,
            cultural_adaptations=request.cultural_adaptations,
        )

        # Encode audio data to base64
        import base64

        audio_b64 = base64.b64encode(result.audio_result.audio_data.data if result.audio_result.audio_data else b"").decode(
            "utf-8"
        )

        return TTSResponse(
            audio_data=audio_b64,
            detected_language=result.detected_language.value,
            confidence=result.confidence,
            cultural_adaptations=result.cultural_adaptations,
            voice_characteristics=result.voice_characteristics,
            processing_time_ms=result.processing_time_ms,
            sample_rate=result.audio_result.audio_data.sample_rate if result.audio_result.audio_data else 0,
            duration=result.audio_result.audio_data.duration if result.audio_result.audio_data else 0.0,
        )

    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")


@router.post("/stt", response_model=STTResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    language_code: Optional[str] = Form(
        None, description="Specific language code (auto-detect if not provided)"
    ),
    auto_detect: bool = Form(True, description="Enable automatic language detection"),
    security: Any = Depends(get_security_middleware),
) -> STTResponse:
    """Transcribe audio with language detection."""
    try:
        manager = get_multilingual_audio_manager_instance()

        # Parse language code if provided
        parsed_language_code = None
        if language_code:
            try:
                parsed_language_code = LanguageCode(language_code)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Invalid language code: {language_code}"
                )

        # Read audio file
        audio_data = await audio_file.read()

        # Create AudioData object
        audio = AudioData(
            data=audio_data,
            sample_rate=16000,  # Default for Whisper
            channels=1,
            duration=0.0,  # Will be calculated
            format="wav",
        )

        # Transcribe audio
        result = await manager.transcribe_audio(
            audio=audio, language_code=parsed_language_code, auto_detect=auto_detect
        )

        return STTResponse(
            text=result.audio_result.text or "",
            detected_language=result.detected_language.value,
            confidence=result.confidence,
            cultural_adaptations=result.cultural_adaptations,
            voice_characteristics=result.voice_characteristics,
            processing_time_ms=result.processing_time_ms,
            segments=result.audio_result.metadata.get("segments", 0) if result.audio_result.metadata else 0,
        )

    except Exception as e:
        logger.error(f"STT transcription failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"STT transcription failed: {str(e)}"
        )


@router.get("/languages", response_model=SupportedLanguagesResponse)
async def get_supported_languages(
    security: Any = Depends(get_security_middleware),
) -> SupportedLanguagesResponse:
    """Get list of supported languages."""
    try:
        manager = get_multilingual_audio_manager_instance()
        languages = await manager.get_supported_languages()

        return SupportedLanguagesResponse(
            supported_languages=[lang.value for lang in languages],
            total_count=len(languages),
        )

    except Exception as e:
        logger.error(f"Failed to get supported languages: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get supported languages: {str(e)}"
        )


@router.get(
    "/languages/{language_code}/capabilities",
    response_model=LanguageCapabilitiesResponse,
)
async def get_language_capabilities(
    language_code: str, security: Any = Depends(get_security_middleware)
) -> LanguageCapabilitiesResponse:
    """Get capabilities for a specific language."""
    try:
        manager = get_multilingual_audio_manager_instance()

        # Validate language code
        try:
            lang_code = LanguageCode(language_code)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid language code: {language_code}"
            )

        # Get capabilities
        capabilities = await manager.get_language_capabilities(lang_code)

        return LanguageCapabilitiesResponse(
            language_code=language_code,
            tts_supported=capabilities.get("tts_supported", False),
            stt_supported=capabilities.get("stt_supported", False),
            tts_capabilities=capabilities.get("tts_capabilities", {}),
            cultural_adaptations=capabilities.get("cultural_adaptations", {}),
            voice_characteristics=capabilities.get("voice_characteristics", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get language capabilities: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get language capabilities: {str(e)}"
        )


@router.get("/tts/stream/{text}")
async def stream_tts_audio(
    text: str,
    language_code: Optional[str] = None,
    voice_style: Optional[str] = None,
    security: Any = Depends(get_security_middleware),
) -> StreamingResponse:
    """Stream TTS audio for real-time playback."""
    try:
        manager = get_multilingual_audio_manager_instance()

        # Parse language code if provided
        parsed_language_code = None
        if language_code:
            try:
                parsed_language_code = LanguageCode(language_code)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Invalid language code: {language_code}"
                )

        # Synthesize speech
        result = await manager.synthesize_speech(
            text=text, language_code=parsed_language_code, voice_style=voice_style
        )

        # Return audio as streaming response
        return StreamingResponse(
            io.BytesIO(result.audio_result.audio_data.data if result.audio_result.audio_data else b""),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline",
                "X-Detected-Language": result.detected_language.value,
                "X-Confidence": str(result.confidence),
                "X-Processing-Time": str(result.processing_time_ms),
            },
        )

    except Exception as e:
        logger.error(f"TTS streaming failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS streaming failed: {str(e)}")


@router.post("/tts/batch")
async def batch_tts_synthesis(
    requests: List[TTSRequest], security: Any = Depends(get_security_middleware)
) -> List[TTSResponse]:
    """Batch TTS synthesis for multiple texts."""
    try:
        manager = get_multilingual_audio_manager_instance()
        results = []

        for request in requests:
            # Parse language code if provided
            language_code = None
            if request.language_code:
                try:
                    language_code = LanguageCode(request.language_code)
                except ValueError:
                    results.append(
                        TTSResponse(
                            audio_data="",
                            detected_language="en",
                            confidence=0.0,
                            cultural_adaptations={},
                            voice_characteristics={},
                            processing_time_ms=0.0,
                            sample_rate=22050,
                            duration=0.0,
                        )
                    )
                    continue

            # Synthesize speech
            result = await manager.synthesize_speech(
                text=request.text,
                language_code=language_code,
                voice_style=request.voice_style,
                cultural_adaptations=request.cultural_adaptations,
            )

            # Encode audio data to base64
            import base64

            audio_b64 = base64.b64encode(result.audio_result.audio_data.data if result.audio_result.audio_data else b"").decode(
                "utf-8"
            )

            results.append(
                TTSResponse(
                    audio_data=audio_b64,
                    detected_language=result.detected_language.value,
                    confidence=result.confidence,
                    cultural_adaptations=result.cultural_adaptations,
                    voice_characteristics=result.voice_characteristics,
                    processing_time_ms=result.processing_time_ms,
                    sample_rate=result.audio_result.audio_data.sample_rate if result.audio_result.audio_data else 0,
                    duration=result.audio_result.audio_data.duration if result.audio_result.audio_data else 0.0,
                )
            )

        return results

    except Exception as e:
        logger.error(f"Batch TTS synthesis failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch TTS synthesis failed: {str(e)}"
        )


@router.get("/health")
async def audio_health_check(
    security: Any = Depends(get_security_middleware),
) -> Dict[str, Any]:
    """Health check for multi-language audio system."""
    try:
        manager = get_multilingual_audio_manager_instance()

        # Check if managers are initialized
        tts_initialized = (
            manager.tts_manager is not None and manager.tts_manager._initialized
        )
        stt_initialized = (
            manager.stt_manager is not None and manager.stt_manager._initialized
        )

        # Get supported languages count
        supported_languages = await manager.get_supported_languages()

        return {
            "status": "healthy" if tts_initialized and stt_initialized else "degraded",
            "tts_initialized": tts_initialized,
            "stt_initialized": stt_initialized,
            "supported_languages_count": len(supported_languages),
            "supported_languages": [lang.value for lang in supported_languages],
        }

    except Exception as e:
        logger.error(f"Audio health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}
