"""
Multi-language support API endpoints.

Provides language detection, localization, and cultural adaptation capabilities.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...algorithms.safety.multilingual_classifier import MultilingualSafetyClassifier
from ...features.localization import (
    CulturalRegion,
    LanguageCode,
    LocalizationService,
    create_localization_manager,
)
from ..security_deps import get_security_middleware

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/language", tags=["Language Support"])


# Pydantic models for API
class LanguageDetectionRequest(BaseModel):
    """Request model for language detection."""

    text: str = Field(..., description="Text to analyze for language detection")
    auto_set: bool = Field(
        True, description="Automatically set detected language as current"
    )


class LanguageDetectionResponse(BaseModel):
    """Response model for language detection."""

    detected_language: str
    confidence: float
    alternative_languages: List[Dict[str, float]]
    processing_time_ms: float
    current_language: str


class LanguagePackInfo(BaseModel):
    """Information about a language pack."""

    language_code: str
    cultural_region: str
    display_name: str
    native_name: str
    text_direction: str
    character_encoding: str
    date_format: str
    time_format: str
    age_appropriate_content: bool


class CulturalAdaptationInfo(BaseModel):
    """Cultural adaptation information."""

    greeting_style: str
    conversation_style: str
    formality_level: str
    emotional_expression: str
    preferred_topics: List[str]
    cultural_taboos: List[str]


class VoiceCharacteristicsInfo(BaseModel):
    """Voice characteristics for a language."""

    preferred_pitch: str
    speaking_rate: str
    emotional_range: str
    formality_level: str
    accent_preference: str
    intonation_patterns: List[str]


class SafetyAnalysisRequest(BaseModel):
    """Request model for safety analysis."""

    text: str = Field(..., description="Text to analyze for safety concerns")
    language_code: Optional[str] = Field(
        None, description="Specific language code to use (auto-detect if not provided)"
    )


class SafetyAnalysisResponse(BaseModel):
    """Response model for safety analysis."""

    safe: bool
    level: str
    confidence: float
    categories: List[str]
    processing_time_ms: float
    language: str
    cultural_adaptations: Dict[str, Any]
    details: Dict[str, Any]


class LanguageStatusResponse(BaseModel):
    """Current language status."""

    current_language: str
    available_languages: List[str]
    fallback_language: str
    auto_detection_enabled: bool


# Factory function for multilingual safety classifier


def get_localization_manager() -> LocalizationService:
    """Dependency to get localization manager instance."""
    return create_localization_manager()


def create_multilingual_safety_classifier() -> MultilingualSafetyClassifier:
    """Create a new multilingual safety classifier instance."""
    return MultilingualSafetyClassifier()


@router.post("/detect", response_model=LanguageDetectionResponse)
async def detect_language(
    request: LanguageDetectionRequest,
    security: Any = Depends(get_security_middleware),
    manager: LocalizationService = Depends(get_localization_manager),
) -> LanguageDetectionResponse:
    """Detect the language of input text."""
    try:
        # manager is now injected via dependency

        # Detect language
        result = manager.detect_and_set_language(request.text)

        # Get alternative languages (not available in current implementation)
        alternatives: list[dict[str, float]] = []

        return LanguageDetectionResponse(
            detected_language=result.detected_language.value,
            confidence=result.confidence,
            alternative_languages=alternatives,
            processing_time_ms=0.0,  # Not available in current implementation
            current_language=manager.current_language.value,
        )

    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Language detection failed: {str(e)}"
        )


@router.get("/status", response_model=LanguageStatusResponse)
async def get_language_status(
    security: Any = Depends(get_security_middleware),
    manager: LocalizationService = Depends(get_localization_manager),
) -> LanguageStatusResponse:
    """Get current language status and available languages."""
    try:
        # manager is now injected via dependency

        return LanguageStatusResponse(
            current_language=manager.current_language.value,
            available_languages=[
                lang.value for lang in manager.get_available_languages()
            ],
            fallback_language=manager.current_language.value,  # Use current as fallback
            auto_detection_enabled=True,
        )

    except Exception as e:
        logger.error(f"Failed to get language status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get language status: {str(e)}"
        )


@router.post("/set")
async def set_language(
    language_code: str = Query(..., description="Language code to set"),
    security: Any = Depends(get_security_middleware),
    manager: LocalizationService = Depends(get_localization_manager),
) -> Dict[str, str]:
    """Set the current language."""
    try:
        # manager is now injected via dependency

        # Validate language code
        try:
            lang_code = LanguageCode(language_code)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid language code: {language_code}"
            )

        # Set language
        success = manager.current_language = lang_code

        if success:
            return {
                "message": f"Language set to {language_code}",
                "current_language": language_code,
            }
        else:
            raise HTTPException(
                status_code=400, detail=f"Language {language_code} not available"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set language: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set language: {str(e)}")


@router.get("/packs", response_model=List[LanguagePackInfo])
async def get_language_packs(
    security: Any = Depends(get_security_middleware),
    manager: LocalizationService = Depends(get_localization_manager),
) -> List[LanguagePackInfo]:
    """Get information about available language packs."""
    try:
        # manager is now injected via dependency
        packs = []

        for lang_code in manager.get_available_languages():
            pack = manager.get_language_pack(lang_code)
            if pack:
                packs.append(
                    LanguagePackInfo(
                        language_code=pack.language_code.value,
                        cultural_region=pack.cultural_region.value,
                        display_name=pack.display_name,
                        native_name=pack.native_name,
                        text_direction="ltr",  # Default to left-to-right
                        character_encoding="utf-8",  # Default encoding
                        date_format="YYYY-MM-DD",  # Default format
                        time_format="HH:MM",  # Default format
                        age_appropriate_content=True,  # Default value
                    )
                )

        return packs

    except Exception as e:
        logger.error(f"Failed to get language packs: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get language packs: {str(e)}"
        )


@router.get("/packs/{language_code}/cultural", response_model=CulturalAdaptationInfo)
async def get_cultural_adaptations(
    language_code: str,
    security: Any = Depends(get_security_middleware),
    manager: LocalizationService = Depends(get_localization_manager),
) -> CulturalAdaptationInfo:
    """Get cultural adaptation information for a specific language."""
    try:
        # manager is now injected via dependency

        # Validate language code
        try:
            lang_code = LanguageCode(language_code)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid language code: {language_code}"
            )

        # Get cultural adaptations
        adaptations = manager.get_cultural_adaptations(lang_code)

        return CulturalAdaptationInfo(
            greeting_style=adaptations.get("greeting_style", "neutral"),
            conversation_style=adaptations.get("conversation_style", "polite"),
            formality_level=adaptations.get("formality_level", "medium"),
            emotional_expression=adaptations.get("emotional_expression", "moderate"),
            preferred_topics=adaptations.get("preferred_topics", []),
            cultural_taboos=adaptations.get("cultural_taboos", []),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cultural adaptations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get cultural adaptations: {str(e)}"
        )


@router.get("/packs/{language_code}/voice", response_model=VoiceCharacteristicsInfo)
async def get_voice_characteristics(
    language_code: str,
    security: Any = Depends(get_security_middleware),
    manager: LocalizationService = Depends(get_localization_manager),
) -> VoiceCharacteristicsInfo:
    """Get voice characteristics for a specific language."""
    try:
        # manager is now injected via dependency

        # Validate language code
        try:
            lang_code = LanguageCode(language_code)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid language code: {language_code}"
            )

        # Get voice characteristics
        characteristics = manager.get_voice_characteristics(lang_code)

        return VoiceCharacteristicsInfo(
            preferred_pitch=characteristics.get("preferred_pitch", "medium"),
            speaking_rate=characteristics.get("speaking_rate", "normal"),
            emotional_range=characteristics.get("emotional_range", "moderate"),
            formality_level=characteristics.get("formality_level", "neutral"),
            accent_preference=characteristics.get("accent_preference", "neutral"),
            intonation_patterns=characteristics.get("intonation_patterns", []),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get voice characteristics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get voice characteristics: {str(e)}"
        )


@router.post("/safety/analyze", response_model=SafetyAnalysisResponse)
async def analyze_safety_multilingual(
    request: SafetyAnalysisRequest,
    security: Any = Depends(get_security_middleware),
    manager: LocalizationService = Depends(get_localization_manager),
) -> SafetyAnalysisResponse:
    """Analyze text for safety concerns with multi-language support."""
    try:
        safety_classifier = create_multilingual_safety_classifier()

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

        # Analyze safety
        result = safety_classifier.classify(
            request.text, language_code.value if language_code else None
        )

        return SafetyAnalysisResponse(
            safe=result.level.value == "safe",
            level=result.level.value,
            confidence=result.confidence,
            categories=result.categories,
            processing_time_ms=result.processing_time_ms,
            language=result.details.get("language", "unknown"),
            cultural_adaptations=result.details.get("cultural_adaptations", {}),
            details=result.details,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Safety analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Safety analysis failed: {str(e)}")


@router.get("/packs/{language_code}/safety-patterns")
async def get_safety_patterns(
    language_code: str,
    security: Any = Depends(get_security_middleware),
    manager: LocalizationService = Depends(get_localization_manager),
) -> Dict[str, Any]:
    """Get safety patterns for a specific language."""
    try:
        # manager is now injected via dependency

        # Validate language code
        try:
            lang_code = LanguageCode(language_code)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid language code: {language_code}"
            )

        # Get safety patterns
        patterns = manager.get_safety_patterns(lang_code)

        return {"patterns": patterns}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get safety patterns: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get safety patterns: {str(e)}"
        )


@router.get("/packs/{language_code}/is-rtl")
async def is_rtl_language(
    language_code: str,
    security: Any = Depends(get_security_middleware),
    manager: LocalizationService = Depends(get_localization_manager),
) -> Dict[str, bool]:
    """Check if a language is right-to-left."""
    try:
        # manager is now injected via dependency

        # Validate language code
        try:
            lang_code = LanguageCode(language_code)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid language code: {language_code}"
            )

        # Check if RTL
        is_rtl = manager.is_rtl_language(lang_code)

        return {"is_rtl": is_rtl}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check RTL status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to check RTL status: {str(e)}"
        )


@router.post("/packs/create-template")
async def create_language_pack_template(
    language_code: str = Query(..., description="Language code for the template"),
    cultural_region: str = Query(..., description="Cultural region for the template"),
    security: Any = Depends(get_security_middleware),
    manager: LocalizationService = Depends(get_localization_manager),
) -> Dict[str, Any]:
    """Create a template for a new language pack."""
    try:
        # manager is now injected via dependency

        # Validate language code
        try:
            lang_code = LanguageCode(language_code)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid language code: {language_code}"
            )

        # Validate cultural region
        try:
            CulturalRegion(cultural_region)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid cultural region: {cultural_region}"
            )

        # Create template
        template = manager.create_language_pack_template(lang_code)

        return template.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create language pack template: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create language pack template: {str(e)}"
        )
