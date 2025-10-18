"""
Health and system monitoring endpoints.

Handles health checks, safety analysis, and performance monitoring.
"""

import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from ....core.performance import performance_timer
from ....observability import ProcessingTimer, get_logger

logger = get_logger(__name__)

# Create router
health_router = APIRouter(prefix="/api/v1/character", tags=["character-health"])


def get_engine() -> Any:
    """Get the real-time engine instance."""
    from ..character_api import get_engine as _get_engine

    return _get_engine()


async def get_hardware_manager() -> Any:
    """Get the hardware manager instance."""
    from ..character_api import get_hardware_manager as _get_hardware_manager

    return _get_hardware_manager()


@health_router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check for character systems."""
    try:
        health_status = await get_engine().get_health_status()
        return {
            "status": "healthy" if health_status["healthy"] else "unhealthy",
            "details": health_status,
            "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@health_router.get("/characters")
async def get_available_characters() -> Dict[str, Any]:
    """Get list of available characters."""
    try:
        engine = get_engine()
        if not engine.character_manager:
            raise HTTPException(
                status_code=500, detail="Character manager not available"
            )

        characters = engine.character_manager.get_available_characters()
        active_char = engine.character_manager.get_active_character()
        return {
            "characters": characters,
            "active_character": active_char.name if active_char else None,
        }
    except Exception as e:
        logger.error(f"Failed to get characters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@health_router.post("/safety/analyze")
async def analyze_safety(text: str) -> Dict[str, Any]:
    """Analyze text for safety concerns (toxicity and PII detection)."""
    try:
        # Log safety analysis request
        logger.info(
            "Safety analysis requested",
            text_length=len(text),
            text_preview=text[:50] + "..." if len(text) > 50 else text,
        )

        engine_inst = get_engine()
        safety_filter = engine_inst.core_engine.lifecycle.safety_filter
        if not safety_filter:
            raise HTTPException(status_code=500, detail="Safety filter not available")

        # Analyze with timing
        with ProcessingTimer(logger, "safety_analysis", "safety_filter") as timer:
            async with performance_timer(
                "safety", "analyze_safety", {"text_length": len(text)}
            ):
                safety_analysis = safety_filter.get_detailed_safety(text)

        # Log safety events if detected
        if safety_analysis.get("overall_level") != "SAFE":
            logger.log_safety_event(
                "safety_concern_detected",
                safety_analysis.get("overall_confidence", 0.0),
                text,
                overall_level=safety_analysis.get("overall_level"),
                toxicity_score=safety_analysis.get("toxicity", {}).get("score", 0),
                pii_score=safety_analysis.get("pii", {}).get("score", 0),
            )

        # Log successful analysis
        logger.info(
            "Safety analysis completed",
            overall_level=safety_analysis.get("overall_level"),
            confidence=safety_analysis.get("overall_confidence"),
            duration_ms=timer.duration_ms if hasattr(timer, "duration_ms") else None,
        )

        return {"text": text, "analysis": safety_analysis, "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Failed to analyze safety: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@health_router.get("/safety/status")
async def get_safety_status() -> Dict[str, Any]:
    """Get safety system status and configuration."""
    try:
        engine_inst = get_engine()
        safety_filter = engine_inst.core_engine.lifecycle.safety_filter
        if not safety_filter:
            raise HTTPException(status_code=500, detail="Safety filter not available")

        return {
            "classifier_enabled": safety_filter.classifier_enabled,
            "classifier_available": safety_filter.classifier is not None,
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Failed to get safety status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@health_router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics for the character system."""
    try:
        metrics = await get_engine().get_performance_metrics()
        return {
            "performance_metrics": metrics,
            "hardware_status": (
                await get_hardware_manager().get_power_status()  # type: ignore
                if hasattr(get_hardware_manager(), "get_power_status")
                else {}
            ),
            "optimization_status": (
                await get_engine().edge_optimizer.get_edge_optimization_summary()
                if get_engine().edge_optimizer is not None
                else {}
            ),
        }
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
