"""
Personalization API endpoints.

Provides endpoints for user preference learning, adaptive conversation styles,
and personalized character recommendations.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..characters.types import (
    Ability,
    Archetype,
    Character,
    CharacterDimensions,
    PersonalityTrait,
    Species,
    Topic,
)
from ..core.personalization import (
    LearningSource,
    PersonalizationManager,
    PreferenceType,
    get_personalization_manager,
)
from .security_deps import get_security_middleware

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/personalization", tags=["Personalization"])


# Pydantic models for API
class PreferenceUpdateRequest(BaseModel):
    """Request model for updating user preferences."""

    preference_type: str = Field(..., description="Type of preference to update")
    value: Any = Field(..., description="Preference value")
    confidence: float = Field(
        0.8, ge=0.0, le=1.0, description="Confidence in the preference"
    )
    source: str = Field("explicit_feedback", description="Source of the preference")


class CharacterRecommendationRequest(BaseModel):
    """Request model for character recommendations."""

    max_recommendations: int = Field(
        5, ge=1, le=20, description="Maximum number of recommendations"
    )
    include_affinities: bool = Field(True, description="Include affinity scores")


class ConversationStyleRequest(BaseModel):
    """Request model for adaptive conversation style."""

    character_id: str = Field(..., description="Character ID for style adaptation")


class UserInsightsResponse(BaseModel):
    """Response model for user insights."""

    user_id: str = Field(..., description="User ID")
    profile_age_days: float = Field(..., description="Age of profile in days")
    total_preferences: int = Field(..., description="Total number of preferences")
    interaction_patterns: int = Field(..., description="Number of interaction patterns")

    character_affinities: Dict[str, float] = Field(
        ..., description="Character affinity scores"
    )
    preferences_summary: Dict[str, Dict[str, Any]] = Field(
        ..., description="Summary of preferences"
    )
    learning_sources: Dict[str, int] = Field(..., description="Learning source counts")
    privacy_compliance: bool = Field(..., description="Privacy compliance status")


class CharacterRecommendationResponse(BaseModel):
    """Response model for character recommendations."""

    recommendations: List[Dict[str, Any]] = Field(
        ..., description="List of recommended characters"
    )
    total_available: int = Field(..., description="Total available characters")
    user_affinities: Dict[str, float] = Field(
        ..., description="User character affinities"
    )


class ConversationStyleResponse(BaseModel):
    """Response model for adaptive conversation style."""

    character_id: str = Field(..., description="Character ID")
    style_adaptations: Dict[str, Any] = Field(..., description="Style adaptations")
    user_preferences: Dict[str, Any] = Field(
        ..., description="Applied user preferences"
    )


class PreferenceResponse(BaseModel):
    """Response model for user preferences."""

    preference_type: str = Field(..., description="Type of preference")
    value: Any = Field(..., description="Preference value")
    confidence: float = Field(..., description="Confidence score")
    source: str = Field(..., description="Learning source")
    last_updated: float = Field(..., description="Last update timestamp")
    interaction_count: int = Field(..., description="Number of interactions")


class PrivacyRequest(BaseModel):
    """Request model for privacy operations."""

    action: str = Field(
        ..., description="Privacy action: 'export', 'clear', or 'status'"
    )


class PrivacyResponse(BaseModel):
    """Response model for privacy operations."""

    action: str = Field(..., description="Action performed")
    success: bool = Field(..., description="Success status")
    data: Optional[Dict[str, Any]] = Field(
        None, description="Exported data (if applicable)"
    )
    message: str = Field(..., description="Status message")


# Global instances
_personalization_manager: Optional[PersonalizationManager] = None


def get_personalization_manager_instance() -> PersonalizationManager:
    """Get the personalization manager instance."""
    global _personalization_manager
    if _personalization_manager is None:
        _personalization_manager = get_personalization_manager()
    return _personalization_manager


@router.post("/preferences", response_model=PreferenceResponse)
async def update_preference(
    request: PreferenceUpdateRequest,
    current_device: str = Query(..., description="Device ID"),
    security: Any = Depends(get_security_middleware),
) -> PreferenceResponse:
    """Update a user preference."""
    try:
        manager = get_personalization_manager_instance()

        # Validate preference type
        try:
            preference_type = PreferenceType(request.preference_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid preference type: {request.preference_type}",
            )

        # Validate source
        try:
            source = LearningSource(request.source)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid source: {request.source}"
            )

        # Update preference
        manager.update_preference(
            user_id=current_device,  # Using device ID as user ID for now
            preference_type=preference_type,
            value=request.value,
            confidence=request.confidence,
            source=source,
        )

        # Get updated preference
        profile = manager.get_or_create_profile(current_device, current_device)
        preference = profile.preferences.get(preference_type)

        if not preference:
            raise HTTPException(
                status_code=500, detail="Failed to retrieve updated preference"
            )

        return PreferenceResponse(
            preference_type=preference.preference_type.value,
            value=preference.value,
            confidence=preference.confidence,
            source=preference.source.value,
            last_updated=preference.last_updated,
            interaction_count=preference.interaction_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update preference: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update preference: {str(e)}"
        )


@router.get("/recommendations", response_model=CharacterRecommendationResponse)
async def get_character_recommendations(
    max_recommendations: int = Query(
        5, ge=1, le=20, description="Maximum number of recommendations"
    ),
    include_affinities: bool = Query(True, description="Include affinity scores"),
    current_device: str = Query(..., description="Device ID"),
    security: Any = Depends(get_security_middleware),
) -> CharacterRecommendationResponse:
    """Get personalized character recommendations."""
    try:
        manager = get_personalization_manager_instance()

        # Get available characters (mock for now - in real implementation, get from character manager)
        available_characters = [
            Character(
                name="Robot Assistant",
                dimensions=CharacterDimensions(
                    species=Species.ROBOT,
                    archetype=Archetype.HELPER,
                    personality_traits=[PersonalityTrait.FRIENDLY],
                    abilities=[Ability.TEACHING],
                    topics=[Topic.SCIENCE, Topic.FRIENDSHIP, Topic.STORIES]
                ),
                voice_style="neutral",
                metadata={},
            ),
            Character(
                name="Friendly Dragon",
                dimensions=CharacterDimensions(
                    species=Species.DRAGON,
                    archetype=Archetype.GUARDIAN,
                    personality_traits=[PersonalityTrait.FRIENDLY],
                    abilities=[Ability.MAGIC],
                    topics=[Topic.ADVENTURES, Topic.MAGIC, Topic.STORIES]
                ),
                voice_style="friendly",
                metadata={},
            ),
        ]

        # Get recommendations
        recommendations = manager.get_personalized_character_recommendations(
            user_id=current_device,
            available_characters=available_characters,
            max_recommendations=max_recommendations,
        )

        # Format recommendations
        recommendation_data = []
        for character, score in recommendations:
            char_data = {
                "name": character.name,
                "character_type": character.dimensions.species.value,
                "voice_style": character.voice_style,
                "conversation_topics": character.dimensions.topics,
                "recommendation_score": score,
            }
            if include_affinities:
                char_data["affinity_score"] = score
            recommendation_data.append(char_data)

        # Get user affinities
        profile = manager.get_or_create_profile(current_device, current_device)
        user_affinities = profile.character_affinities

        return CharacterRecommendationResponse(
            recommendations=recommendation_data,
            total_available=len(available_characters),
            user_affinities=user_affinities,
        )

    except Exception as e:
        logger.error(f"Failed to get character recommendations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get character recommendations: {str(e)}"

        )


@router.post("/conversation-style", response_model=ConversationStyleResponse)
async def get_adaptive_conversation_style(
    request: ConversationStyleRequest,
    current_device: str = Query(..., description="Device ID"),
    security: Any = Depends(get_security_middleware),
) -> ConversationStyleResponse:
    """Get adaptive conversation style for a character."""
    try:
        manager = get_personalization_manager_instance()

        # Create character object (mock for now)
        character = Character(
            name=request.character_id,
            dimensions=CharacterDimensions(
                species=Species.ROBOT,
                archetype=Archetype.HELPER,
                personality_traits=[PersonalityTrait.FRIENDLY],
                abilities=[Ability.TEACHING],
                topics=[]
            ),
            voice_style="neutral",
            metadata={},
        )

        # Get adaptive style
        style_adaptations = manager.get_adaptive_conversation_style(
            user_id=current_device, character=character
        )

        # Get user preferences summary
        profile = manager.get_or_create_profile(current_device, current_device)
        user_preferences = {
            pref_type.value: {"value": pref.value, "confidence": pref.confidence}
            for pref_type, pref in profile.preferences.items()
        }

        return ConversationStyleResponse(
            character_id=request.character_id,
            style_adaptations=style_adaptations,
            user_preferences=user_preferences,
        )

    except Exception as e:
        logger.error(f"Failed to get conversation style: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get conversation style: {str(e)}"
        )


@router.get("/insights", response_model=UserInsightsResponse)
async def get_user_insights(
    current_device: str = Query(..., description="Device ID"),
    security: Any = Depends(get_security_middleware),
) -> UserInsightsResponse:
    """Get user personalization insights."""
    try:
        manager = get_personalization_manager_instance()

        insights = manager.get_user_insights(current_device)

        return UserInsightsResponse(**insights)

    except Exception as e:
        logger.error(f"Failed to get user insights: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get user insights: {str(e)}"
        )


@router.get("/preferences", response_model=List[PreferenceResponse])
async def get_user_preferences(
    current_device: str = Query(..., description="Device ID"),
    security: Any = Depends(get_security_middleware),
) -> List[PreferenceResponse]:
    """Get all user preferences."""
    try:
        manager = get_personalization_manager_instance()

        profile = manager.get_or_create_profile(current_device, current_device)

        preferences = []
        for preference in profile.preferences.values():
            preferences.append(
                PreferenceResponse(
                    preference_type=preference.preference_type.value,
                    value=preference.value,
                    confidence=preference.confidence,
                    source=preference.source.value,
                    last_updated=preference.last_updated,
                    interaction_count=preference.interaction_count,
                )
            )

        return preferences

    except Exception as e:
        logger.error(f"Failed to get user preferences: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get user preferences: {str(e)}"
        )


@router.post("/privacy", response_model=PrivacyResponse)
async def handle_privacy_request(
    request: PrivacyRequest,
    current_device: str = Query(..., description="Device ID"),
    security: Any = Depends(get_security_middleware),
) -> PrivacyResponse:
    """Handle privacy-related requests (export, clear, status)."""
    try:
        manager = get_personalization_manager_instance()

        if request.action == "export":
            data = manager.export_user_data(current_device)
            if data is None:
                return PrivacyResponse(
                    action="export",
                    success=False,
                    message="No personalization data found for user",
                    data=None,
                )

            return PrivacyResponse(
                action="export",
                success=True,
                data=data,
                message="User data exported successfully",
            )

        elif request.action == "clear":
            success = manager.clear_user_data(current_device)
            return PrivacyResponse(
                action="clear",
                success=success,
                message=(
                    "User data cleared successfully"
                    if success
                    else "Failed to clear user data"
                ),
                data=None,
            )

        elif request.action == "status":
            profile = manager.get_or_create_profile(current_device, current_device)
            return PrivacyResponse(
                action="status",
                success=True,
                data={
                    "has_data": len(profile.preferences) > 0
                    or len(profile.interaction_patterns) > 0,
                    "preferences_count": len(profile.preferences),
                    "patterns_count": len(profile.interaction_patterns),
                    "last_updated": profile.last_updated,
                },
                message="Privacy status retrieved successfully",
            )

        else:
            raise HTTPException(
                status_code=400, detail=f"Invalid privacy action: {request.action}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to handle privacy request: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to handle privacy request: {str(e)}"
        )


@router.post("/learn")
async def learn_from_conversation(
    conversation_data: Dict[str, Any],
    current_device: str = Query(..., description="Device ID"),
    security: Any = Depends(get_security_middleware),
) -> Dict[str, str]:
    """Learn from conversation data (for internal use)."""
    try:
        manager = get_personalization_manager_instance()

        # Extract conversation turns from the data
        # This would typically come from the session memory system
        conversation_turns = conversation_data.get("turns", [])
        character_id = (
            conversation_turns[0].get("character_name", "unknown")
            if conversation_turns
            else "unknown"
        )

        # Convert to ConversationTurn objects
        from ..algorithms.conversational_ai.session_memory import ConversationTurn

        turns = []
        for turn_data in conversation_turns:
            turn = ConversationTurn(
                timestamp=turn_data.get("timestamp", time.time()),
                user_input=turn_data.get("user_input", ""),
                character_response=turn_data.get("character_response", ""),
                character_name=turn_data.get("character_name", character_id),
            )
            turns.append(turn)

        # Learn from conversation
        manager.learn_from_conversation(
            user_id=current_device, conversation_turns=turns, character_id=character_id
        )

        return {"status": "success", "message": "Learned from conversation"}

    except Exception as e:
        logger.error(f"Failed to learn from conversation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to learn from conversation: {str(e)}"
        )


@router.get("/health")
async def personalization_health_check(
    security: Any = Depends(get_security_middleware),
) -> Dict[str, Any]:
    """Health check for personalization system."""
    try:
        manager = get_personalization_manager_instance()

        return {
            "status": "healthy",
            "total_profiles": len(manager.user_profiles),
            "storage_path": str(manager.storage_path),
            "preference_learner": "active",
        }

    except Exception as e:
        logger.error(f"Personalization health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}
