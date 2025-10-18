"""
Configuration and hardware management endpoints.

Handles character profiles, voice embeddings, and hardware configuration.
"""

import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Header, HTTPException, UploadFile

from ....observability import get_logger

logger = get_logger(__name__)

# Create router
config_router = APIRouter(prefix="/api/v1/character", tags=["character-config"])


async def get_engine() -> Any:
    """Get the real-time engine instance."""
    from ..character_api import get_engine as _get_engine

    return _get_engine()


async def get_hardware_manager() -> Any:
    """Get the hardware manager instance."""
    from ..character_api import get_hardware_manager as _get_hardware_manager

    return _get_hardware_manager()


@config_router.get("/profiles")
async def list_profiles() -> Dict[str, Any]:
    """List installed character profiles with basic metadata."""
    try:
        engine = await get_engine()
        if not engine.character_manager:
            raise HTTPException(
                status_code=500, detail="Character manager not available"
            )

        names = engine.character_manager.get_available_characters()
        info = []
        for n in names:
            info.append(engine.character_manager.get_character_info(n))
        return {"profiles": info}
    except Exception as e:
        logger.error(f"Failed to list profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@config_router.post("/profiles/reload")
async def reload_profiles() -> Dict[str, Any]:
    """Reload character profiles from disk (configs/characters)."""
    try:
        engine = await get_engine()
        if not engine.character_manager:
            raise HTTPException(
                status_code=500, detail="Character manager not available"
            )
        ok = await engine.character_manager.reload_profiles()
        return {"success": bool(ok)}
    except Exception as e:
        logger.error(f"Failed to reload profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@config_router.post("/profiles/upload")
async def upload_profile_archive(archive: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload a zip archive containing a character profile folder (profile.yaml, consent.yaml, voice artifact).

    The archive is extracted under configs/characters/ and profiles are reloaded.
    """
    try:
        from ....core.config import Config

        cfg = Config()
        dest_root = Path(str(cfg.paths.characters_dir)).resolve()
        tmp = tempfile.NamedTemporaryFile(delete=False)
        data = await archive.read()
        tmp.write(data)
        tmp.flush()
        tmp.close()

        with zipfile.ZipFile(tmp.name, "r") as z:
            # Safe extraction (avoid ZipSlip)
            for member in z.infolist():
                member_path = Path(member.filename)
                # skip absolute or parent-traversal paths
                if member_path.is_absolute() or ".." in member_path.parts:
                    continue
                target = (dest_root / member_path).resolve()
                if not str(target).startswith(str(dest_root)):
                    continue
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(member) as src, open(target, "wb") as out:
                        out.write(src.read())

        engine = await get_engine()
        if not engine.character_manager:
            raise HTTPException(
                status_code=500, detail="Character manager not available"
            )
        ok = await engine.character_manager.reload_profiles()
        return {"success": bool(ok)}
    except Exception as e:
        logger.error(f"Failed to upload profile archive: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@config_router.post("/voices/embeddings/recompute")
async def recompute_voice_embeddings(
    payload: Dict[str, Any],
    x_admin_token: Optional[str] = Header(default=None, alias="x-admin-token"),
) -> Dict[str, Any]:
    """Recompute stored voice embeddings for one or all characters.

    Requires admin token provided via `x-admin-token` header and configured in env `CAI_ADMIN_TOKEN`.
    """
    try:
        expected = os.environ.get("CAI_ADMIN_TOKEN")
        if not expected:
            raise HTTPException(status_code=500, detail="Admin token not configured")
        if x_admin_token != expected:
            raise HTTPException(status_code=403, detail="Invalid admin token")

        engine = await get_engine()
        if not engine.core_engine.lifecycle.voice_manager:
            raise HTTPException(status_code=500, detail="Voice manager not available")

        character_name = payload.get("character_name")
        if character_name:
            # Recompute for specific character
            success = (
                await engine.core_engine.lifecycle.voice_manager.recompute_embeddings(
                    character_name
                )
            )
            return {
                "success": success,
                "character": character_name,
                "message": f"Voice embeddings recomputed for {character_name}"
                if success
                else f"Failed to recompute embeddings for {character_name}",
            }
        else:
            # Recompute for all characters
            success = (
                await engine.core_engine.lifecycle.voice_manager.recompute_all_embeddings()
            )
            return {
                "success": success,
                "message": "Voice embeddings recomputed for all characters"
                if success
                else "Failed to recompute embeddings for all characters",
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to recompute voice embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@config_router.get("/hardware/status")
async def get_hardware_status() -> Dict[str, Any]:
    """Get hardware status and constraints."""
    try:
        hardware_manager = await get_hardware_manager()
        return {
            "constraints": {
                "max_memory_gb": hardware_manager.constraints.max_memory_gb,
                "max_cpu_cores": hardware_manager.constraints.max_cpu_cores,
                "battery_life_hours": hardware_manager.constraints.battery_life_hours,
                "target_latency_ms": hardware_manager.constraints.target_latency_ms,
            },
            "power_status": (
                await hardware_manager.power_manager.get_power_status()
                if hasattr(hardware_manager, "power_manager")
                else {}
            ),
            "sensor_status": {
                "microphone": hardware_manager.microphone is not None,
                "speaker": hardware_manager.speaker is not None,
                "buttons": hardware_manager.buttons is not None,
                "leds": hardware_manager.leds is not None,
            },
        }
    except Exception as e:
        logger.error(f"Failed to get hardware status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@config_router.post("/hardware/optimize")
async def optimize_hardware() -> Dict[str, Any]:
    """Optimize hardware for character deployment."""
    try:
        hardware_manager = await get_hardware_manager()
        optimizations = await hardware_manager.optimize_for_toy()
        return {
            "success": True,
            "optimizations": optimizations,
            "message": "Hardware optimized for character deployment",
        }
    except Exception as e:
        logger.error(f"Failed to optimize hardware: {e}")
        raise HTTPException(status_code=500, detail=str(e))
