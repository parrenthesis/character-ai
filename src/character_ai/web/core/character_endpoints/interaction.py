"""
Character interaction endpoints.

Handles real-time voice and text interactions with characters.
"""

import base64
import time
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile

from ....observability import get_logger

logger = get_logger(__name__)

# Create router
interaction_router = APIRouter(
    prefix="/api/v1/character", tags=["character-interaction"]
)


async def get_engine() -> Any:
    """Get the real-time engine instance."""
    # This will be imported from the main character_api module
    from ..character_api import get_engine as _get_engine

    return _get_engine()


@interaction_router.post("/interact")
async def interact_with_character(request: Dict[str, Any]) -> Dict[str, Any]:
    """Interact with the active character in real-time."""
    try:
        start_time = time.time()

        # Create audio data from base64 input
        from ....core.protocols import AudioData

        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(request["audio_data"])
            from ....core.config import Config

            cfg = Config()
            audio_data = AudioData(
                data=audio_bytes,
                sample_rate=cfg.interaction.sample_rate,
                channels=cfg.interaction.channels,
                format="wav",
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio data: {e}")

        # Process with real-time engine
        engine = await get_engine()
        result = await engine.process_realtime_audio(audio_data)

        if result.error:
            raise HTTPException(status_code=500, detail=result.error)

        latency_ms = (time.time() - start_time) * 1000

        return {
            "response_audio": (
                base64.b64encode(result.audio_data.data).decode()
                if result.audio_data and result.audio_data.data
                else ""
            ),
            "response_text": result.text or "",
            "character_name": result.metadata.get("character", "Unknown")
            if result.metadata
            else "Unknown",
            "latency_ms": latency_ms,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@interaction_router.post("/interact/audio")
async def interact_with_audio_file(
    audio_file: UploadFile = File(...),
) -> Dict[str, Any]:
    """Interact with character using uploaded audio file."""
    try:
        start_time = time.time()

        # Read audio file
        audio_content = await audio_file.read()

        # Create audio data
        from ....core.protocols import AudioData

        audio_data = AudioData(
            data=audio_content, sample_rate=16000, channels=1, format="wav"
        )

        # Process with real-time engine
        engine = await get_engine()
        result = await engine.process_realtime_audio(audio_data)

        if result.error:
            raise HTTPException(status_code=500, detail=result.error)

        latency_ms = (time.time() - start_time) * 1000

        return {
            "response_audio": (
                base64.b64encode(result.audio_data.data).decode()
                if result.audio_data and result.audio_data.data
                else ""
            ),
            "response_text": result.text or "",
            "character_name": result.metadata.get("character", "Unknown")
            if result.metadata
            else "Unknown",
            "latency_ms": latency_ms,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process audio interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
