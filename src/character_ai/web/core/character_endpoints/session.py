"""
Character session management endpoints.

Handles character selection, creation, and memory management.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException

from ....characters import CharacterType
from ....features.security import DeviceIdentity, DeviceRole
from ....observability import (
    ProcessingTimer,
    create_metrics_collector,
    get_logger,
    report_error,
)
from ...security_deps import require_authentication, require_user_or_admin

logger = get_logger(__name__)

# Create router
session_router = APIRouter(prefix="/api/v1/character", tags=["character-session"])


def get_engine() -> Any:
    """Get the real-time engine instance."""
    from ..character_api import get_engine as _get_engine

    return _get_engine()


@session_router.post("/character/set")
async def set_active_character(request: Dict[str, Any]) -> Dict[str, Any]:
    """Set the active character for interactions."""
    try:
        success = get_engine().set_active_character(request["character_name"])
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Character '{request['character_name']}' not found",
            )

        character_info = get_engine().get_character_info()
        return {
            "success": True,
            "active_character": character_info,
            "message": f"Active character set to {request['character_name']}",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set character: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@session_router.post("/character/create")
async def create_custom_character(
    request: Dict[str, Any],
    device: DeviceIdentity = Depends(require_user_or_admin),
) -> Dict[str, Any]:
    """Create a custom character."""
    try:
        # Log character creation attempt
        logger.log_character_interaction(
            "character_create_start",
            request["name"],
            character_type=request["character_type"],
            device_id=device.device_id,
            topics_count=len(request.get("custom_topics", []))
            if request.get("custom_topics", [])
            else 0,
        )

        # Validate character type
        try:
            character_type = CharacterType(request["character_type"].lower())
        except ValueError:
            logger.warning(
                "Invalid character type",
                character_type=request["character_type"],
                device_id=device.device_id,
                valid_types=[t.value for t in CharacterType],
            )
            raise HTTPException(
                status_code=400,
                detail=f"Invalid character type. Must be one of: {[t.value for t in CharacterType]}",
            )

        # Get engine and performance timer
        engine = get_engine()
        from ....core.performance import performance_timer

        # Create character with timing
        with ProcessingTimer(
            logger, "character_creation", "character_manager"
        ) as timer:
            async with performance_timer(
                "character",
                "create_character",
                {
                    "character_name": request["name"],
                    "character_type": request["character_type"],
                },
            ):
                success = await engine.character_manager.create_custom_character(
                    request["name"], character_type, request.get("custom_topics", [])
                )

        # Record character interaction metrics
        create_metrics_collector().record_character_interaction(
            character_id=request["name"],
            interaction_type="create",
            duration=(
                timer.duration_ms / 1000.0 if hasattr(timer, "duration_ms") else 0.0
            ),
            component="character_manager",
        )

        if not success:
            # Report as error (not crash since it's expected failure)
            report_error(
                error_type="CharacterCreationFailed",
                error_message=f"Character creation failed for {request['name']}",
                component="character_manager",
                severity="ERROR",
                context={
                    "character_name": request["name"],
                    "character_type": request["character_type"],
                    "device_id": device.device_id,
                    "custom_topics": request.get("custom_topics", []),
                },
            )

            logger.error(
                "Character creation failed",
                character_name=request["name"],
                character_type=request["character_type"],
                device_id=device.device_id,
            )
            raise HTTPException(status_code=500, detail="Failed to create character")

        # Log successful creation
        logger.log_character_interaction(
            "character_create_success",
            request["name"],
            character_type=request["character_type"],
            device_id=device.device_id,
            duration_ms=timer.duration_ms if hasattr(timer, "duration_ms") else None,
        )

        return {
            "success": True,
            "character_name": request["name"],
            "character_type": request["character_type"],
            "message": f"Character '{request['name']}' created successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create character: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@session_router.get("/character/info")
async def get_character_info(character_name: Optional[str] = None) -> Dict[str, Any]:
    """Get information about a character."""
    try:
        engine = get_engine()
        if character_name:
            if not engine.character_manager:
                raise HTTPException(
                    status_code=500, detail="Character manager not available"
                )
            info = await engine.character_manager.get_character_info(character_name)
        else:
            info = await engine.get_character_info()

        if "error" in info:
            raise HTTPException(status_code=404, detail=info["error"])

        return info  # type: ignore[no-any-return]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get character info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@session_router.get("/memory/status")
async def get_memory_status(character_name: Optional[str] = None) -> Dict[str, Any]:
    """Get session memory status for a character or all characters."""
    try:
        engine_inst = get_engine()
        memory = engine_inst.session_memory

        if character_name:
            summary = memory.get_conversation_summary(character_name)
            return {"character": summary}
        else:
            summaries = {}
            for char_name in memory.conversations.keys():
                summaries[char_name] = memory.get_conversation_summary(char_name)
            return {"all_characters": summaries}
    except Exception as e:
        logger.error(f"Failed to get memory status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@session_router.post("/memory/clear")
async def clear_memory(
    character_name: Optional[str] = None,
    device: DeviceIdentity = Depends(require_authentication),
) -> Dict[str, Any]:
    """Clear session memory for a character or all characters.

    Requires authentication. Admin role required for clearing all characters.
    """
    try:
        engine_inst = get_engine()
        memory = engine_inst.session_memory

        if character_name:
            # Clear specific character (user can clear their own)
            memory.clear_conversation(character_name)
            return {"success": True, "cleared": character_name}
        else:
            # Clear all characters (requires admin role)
            if device.role != DeviceRole.ADMIN:
                raise HTTPException(
                    status_code=403, detail="Admin role required to clear all memory"
                )

            memory.clear_all_conversations()
            return {"success": True, "cleared": "all_characters"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@session_router.get("/memory/conversation/{character_name}")
async def get_conversation_history(
    character_name: str, max_turns: Optional[int] = None
) -> Dict[str, Any]:
    """Get conversation history for a character."""
    try:
        engine_inst = get_engine()
        memory = engine_inst.session_memory

        conversation = memory.get_conversation_context(character_name, max_turns)
        turns = [turn.to_dict() for turn in conversation]

        return {
            "character_name": character_name,
            "total_turns": len(turns),
            "turns": turns,
        }
    except Exception as e:
        logger.error(f"Failed to get conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
