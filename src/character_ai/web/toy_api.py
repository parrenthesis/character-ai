"""
Toy-specific API endpoints for character.ai.

Provides endpoints for character management, real-time interaction, and toy controls.
"""

import base64
import os
import time
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Response,
    UploadFile,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..characters import CharacterType
from ..core.crash_reporting import report_error
from ..core.logging import ProcessingTimer, get_logger
from ..core.metrics import get_metrics_collector
from ..core.performance import performance_timer
from ..core.security import DeviceIdentity, DeviceRole
from ..hardware.toy_hardware_manager import ToyHardwareManager
from ..production.real_time_engine import RealTimeInteractionEngine
from .error_handling_middleware import ErrorHandlingMiddleware, RecoveryMiddleware
from .health_api import health_router
from .language_api import router as language_router
from .log_search_api import log_search_router
from .logging_middleware import (
    LoggingMiddleware,
    PerformanceLoggingMiddleware,
    SecurityLoggingMiddleware,
)
from .metrics_api import metrics_router
from .monitoring_api import monitoring_router
from .multilingual_audio_api import router as multilingual_audio_router
from .parental_controls_api import router as parental_controls_router
from .performance_api import performance_router
from .personalization_api import router as personalization_router
from .security_deps import (
    add_security_headers,
    get_security_manager,
    require_authentication,
    require_user_or_admin,
)
from .streaming_api import streaming_router

logger = get_logger(__name__)

# Create app and router
app = FastAPI(title="Character AI API")

# Add middleware (order matters - add first)
app.add_middleware(ErrorHandlingMiddleware)  # Catch all errors first
app.add_middleware(RecoveryMiddleware)  # Monitor error rates
app.add_middleware(LoggingMiddleware)  # Log requests
app.add_middleware(SecurityLoggingMiddleware)  # Log security events
app.add_middleware(
    PerformanceLoggingMiddleware, slow_request_threshold_ms=1000.0
)  # Monitor performance

# Add security middleware
app.middleware("http")(add_security_headers)

toy_router = APIRouter(prefix="/api/v1/toy", tags=["toy"])

# Lazy singletons to avoid import-time side effects
_hardware_manager = None
_real_time_engine = None


def get_hardware_manager() -> ToyHardwareManager:
    global _hardware_manager
    if _hardware_manager is None:
        _hardware_manager = ToyHardwareManager()
    return _hardware_manager


def get_engine() -> RealTimeInteractionEngine:
    global _real_time_engine
    if _real_time_engine is None:
        _real_time_engine = RealTimeInteractionEngine(get_hardware_manager())
    return _real_time_engine


class _EngineProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(get_engine(), name)


class _HardwareProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(get_hardware_manager(), name)


# Public symbols remain patchable in tests
engine = _EngineProxy()
hardware_manager = _HardwareProxy()
real_time_engine = engine


# Pydantic models for API
class CharacterSetRequest(BaseModel):
    character_name: str


class CharacterCreateRequest(BaseModel):
    name: str
    character_type: str
    custom_topics: Optional[list] = None


class InteractionRequest(BaseModel):
    audio_data: str  # Base64 encoded audio
    character_name: Optional[str] = None


class InteractionResponse(BaseModel):
    response_audio: str  # Base64 encoded response
    response_text: str
    character_name: str
    latency_ms: float


class RecomputeEmbeddingsRequest(BaseModel):
    character: Optional[str] = None
    force: bool = False


class MemoryConfigRequest(BaseModel):
    max_turns: Optional[int] = None
    max_tokens: Optional[int] = None
    max_age_seconds: Optional[int] = None
    enable_memory: Optional[bool] = None


@toy_router.on_event("startup")
async def startup_event() -> None:
    """Initialize toy systems on startup if explicitly enabled.

    To avoid blocking tests or environments without models/hardware, startup
    initialization is disabled by default. Set CAI_ENABLE_API_STARTUP=1 to enable.
    """
    try:
        if os.environ.get("CAI_ENABLE_API_STARTUP", "0") == "1":
            await get_engine().initialize()
            logger.info("Toy API initialized successfully")
        else:
            logger.info(
                "Skipping toy API startup initialization (CAI_ENABLE_API_STARTUP!=1)"
            )
    except Exception as e:
        logger.error(f"Failed to initialize toy API: {e}")
        raise


@toy_router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check for toy systems."""
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


@toy_router.get("/characters")
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


@toy_router.post("/character/set")
async def set_active_character(request: CharacterSetRequest) -> Dict[str, Any]:
    """Set the active character for interactions."""
    try:
        success = await get_engine().set_active_character(request.character_name)
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Character '{request.character_name}' not found",
            )

        character_info = await get_engine().get_character_info()
        return {
            "success": True,
            "active_character": character_info,
            "message": f"Active character set to {request.character_name}",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set character: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@toy_router.post("/character/create")
async def create_custom_character(
    request: CharacterCreateRequest,
    device: DeviceIdentity = Depends(require_user_or_admin),
) -> Dict[str, Any]:
    """Create a custom character."""
    try:
        # Log character creation attempt
        logger.log_character_interaction(
            "character_create_start",
            request.name,
            character_type=request.character_type,
            device_id=device.device_id,
            topics_count=len(request.custom_topics) if request.custom_topics else 0,
        )

        # Validate character type
        try:
            character_type = CharacterType(request.character_type.lower())
        except ValueError:
            logger.warning(
                "Invalid character type",
                character_type=request.character_type,
                device_id=device.device_id,
                valid_types=[t.value for t in CharacterType],
            )
            raise HTTPException(
                status_code=400,
                detail=f"Invalid character type. Must be one of: {[t.value for t in CharacterType]}",
            )

        # Create character with timing
        with ProcessingTimer(
            logger, "character_creation", "character_manager"
        ) as timer:
            async with performance_timer(
                "character",
                "create_character",
                {
                    "character_name": request.name,
                    "character_type": request.character_type,
                },
            ):
                success = await engine.character_manager.create_custom_character(
                    request.name, character_type, request.custom_topics
                )

        # Record character interaction metrics
        get_metrics_collector().record_character_interaction(
            character_id=request.name,
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
                error_message=f"Character creation failed for {request.name}",
                component="character_manager",
                severity="ERROR",
                context={
                    "character_name": request.name,
                    "character_type": request.character_type,
                    "device_id": device.device_id,
                    "custom_topics": request.custom_topics,
                },
            )

            logger.error(
                "Character creation failed",
                character_name=request.name,
                character_type=request.character_type,
                device_id=device.device_id,
            )
            raise HTTPException(status_code=500, detail="Failed to create character")

        # Log successful creation
        logger.log_character_interaction(
            "character_create_success",
            request.name,
            character_type=request.character_type,
            device_id=device.device_id,
            duration_ms=timer.duration_ms if hasattr(timer, "duration_ms") else None,
        )

        return {
            "success": True,
            "character_name": request.name,
            "character_type": request.character_type,
            "message": f"Character '{request.name}' created successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create character: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@toy_router.get("/character/info")
async def get_character_info(character_name: Optional[str] = None) -> Dict[str, Any]:
    """Get information about a character."""
    try:
        engine = get_engine()
        if character_name:
            if not engine.character_manager:
                raise HTTPException(
                    status_code=500, detail="Character manager not available"
                )
            info = engine.character_manager.get_character_info(character_name)
        else:
            info = await engine.get_character_info()

        if "error" in info:
            raise HTTPException(status_code=404, detail=info["error"])

        return info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get character info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@toy_router.post("/interact")
async def interact_with_character(request: InteractionRequest) -> InteractionResponse:
    """Interact with the active character in real-time."""
    try:
        import time

        start_time = time.time()

        # Create audio data from base64 input
        import base64

        from ..core.protocols import AudioData

        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(request.audio_data)
            from ..core.config import Config

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
        result = await get_engine().process_realtime_audio(audio_data)

        if result.error:
            raise HTTPException(status_code=500, detail=result.error)

        latency_ms = (time.time() - start_time) * 1000

        return InteractionResponse(
            response_audio=(
                base64.b64encode(result.audio_data.data).decode()
                if result.audio_data and result.audio_data.data
                else ""
            ),
            response_text=result.text or "",
            character_name=result.metadata.get("character", "Unknown")
            if result.metadata
            else "Unknown",
            latency_ms=latency_ms,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@toy_router.post("/interact/audio")
async def interact_with_audio_file(
    audio_file: UploadFile = File(...),
) -> InteractionResponse:
    """Interact with character using uploaded audio file."""
    try:
        import time

        start_time = time.time()

        # Read audio file
        audio_content = await audio_file.read()

        # Create audio data
        from ..core.protocols import AudioData

        audio_data = AudioData(
            data=audio_content, sample_rate=16000, channels=1, format="wav"
        )

        # Process with real-time engine
        result = await get_engine().process_realtime_audio(audio_data)

        if result.error:
            raise HTTPException(status_code=500, detail=result.error)

        latency_ms = (time.time() - start_time) * 1000

        return InteractionResponse(
            response_audio=(
                base64.b64encode(result.audio_data.data).decode()
                if result.audio_data and result.audio_data.data
                else ""
            ),
            response_text=result.text or "",
            character_name=result.metadata.get("character", "Unknown")
            if result.metadata
            else "Unknown",
            latency_ms=latency_ms,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process audio interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@toy_router.get("/profiles")
async def list_profiles() -> Dict[str, Any]:
    """List installed character profiles with basic metadata."""
    try:
        engine = get_engine()
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


@toy_router.post("/profiles/reload")
async def reload_profiles() -> Dict[str, Any]:
    """Reload character profiles from disk (configs/characters)."""
    try:
        engine = get_engine()
        if not engine.character_manager:
            raise HTTPException(
                status_code=500, detail="Character manager not available"
            )
        ok = await engine.character_manager.reload_profiles()
        return {"success": bool(ok)}
    except Exception as e:
        logger.error(f"Failed to reload profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@toy_router.post("/profiles/upload")
async def upload_profile_archive(archive: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload a zip archive containing a character profile folder (profile.yaml, consent
    .yaml, voice artifact).

        The archive is extracted under configs/characters/ and profiles are reloaded.
    """
    try:
        import tempfile
        import zipfile
        from pathlib import Path

        from ..core.config import Config

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

        engine = get_engine()
        if not engine.character_manager:
            raise HTTPException(
                status_code=500, detail="Character manager not available"
            )
        ok = await engine.character_manager.reload_profiles()
        return {"success": bool(ok)}
    except Exception as e:
        logger.error(f"Failed to upload profile archive: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@toy_router.post("/voices/embeddings/recompute")
async def recompute_voice_embeddings(
    payload: RecomputeEmbeddingsRequest,
    x_admin_token: Optional[str] = Header(default=None, alias="x-admin-token"),
) -> Dict[str, Any]:
    """Recompute stored voice embeddings for one or all characters.

        Requires admin token provided via `x-admin-token` header and configured in env `CAI_
    ADMIN_TOKEN`.
    """
    try:
        expected = os.environ.get("CAI_ADMIN_TOKEN")
        if not expected:
            raise HTTPException(status_code=403, detail="Admin token not configured")
        if x_admin_token != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

        engine_inst = get_engine()
        vm = engine_inst.voice_manager
        if not vm:
            raise HTTPException(status_code=500, detail="Voice manager not available")

        async def _recompute_one(name: str) -> bool:
            # Try mapped voice path first
            path = await vm.get_character_voice_path(name)
            if not path:
                # Fallback to conventional storage path
                from pathlib import Path

                candidate = Path(vm.voice_storage_dir) / f"{name}_voice.wav"
                path = str(candidate) if candidate.exists() else None
            if not path:
                return False
            return bool(
                await vm.recompute_embedding_from_artifact(
                    name, path, force=payload.force
                )
            )

        results: Dict[str, bool] = {}
        if payload.character:
            name = payload.character.lower()
            ok = await _recompute_one(name)
            results[name] = ok
        else:
            try:
                names = await engine_inst.list_character_voices()
            except Exception:
                names = []
            for n in names:
                ok = await _recompute_one(n)
                results[n] = ok

        success = all(results.values()) if results else False
        return {"success": success, "results": results, "forced": payload.force}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to recompute embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@toy_router.get("/memory/status")
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


@toy_router.post("/memory/clear")
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


@toy_router.get("/memory/conversation/{character_name}")
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


@toy_router.post("/safety/analyze")
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
        safety_filter = engine_inst.safety_filter
        if not safety_filter:
            raise HTTPException(status_code=500, detail="Safety filter not available")

        # Analyze with timing
        with ProcessingTimer(logger, "safety_analysis", "safety_filter") as timer:
            async with performance_timer(
                "safety", "analyze_safety", {"text_length": len(text)}
            ):
                safety_analysis = safety_filter.get_detailed_safety(text)  # type: ignore

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


@toy_router.get("/safety/status")
async def get_safety_status() -> Dict[str, Any]:
    """Get safety system status and configuration."""
    try:
        engine_inst = get_engine()
        safety_filter = engine_inst.safety_filter
        if not safety_filter:
            raise HTTPException(status_code=500, detail="Safety filter not available")

        return {
            "classifier_enabled": safety_filter.classifier_enabled,  # type: ignore
            "classifier_available": safety_filter.classifier is not None,  # type: ignore
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Failed to get safety status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Authentication endpoints
@toy_router.post("/auth/register")
async def register_device(device_name: Optional[str] = None) -> Dict[str, Any]:
    """Register a new device and get authentication token."""
    try:
        security_manager = get_security_manager()
        await security_manager.initialize()

        device = security_manager.get_device_identity()
        if not device:
            raise HTTPException(
                status_code=500, detail="Failed to create device identity"
            )

        # Generate JWT token
        token = security_manager.generate_jwt_token()

        return {
            "device_id": device.device_id,
            "device_name": device.device_name,
            "access_token": token,
            "token_type": "bearer",
            "expires_in": security_manager.config.jwt_expiry_seconds,
            "role": device.role.value,
            "capabilities": device.capabilities,
        }
    except Exception as e:
        logger.error(f"Failed to register device: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@toy_router.get("/auth/me")
async def get_current_device_info(
    device: DeviceIdentity = Depends(require_authentication),
) -> Dict[str, Any]:
    """Get current device information."""
    return {
        "device_id": device.device_id,
        "device_name": device.device_name,
        "role": device.role.value,
        "capabilities": device.capabilities,
        "created_at": device.created_at,
        "last_seen": device.last_seen,
        "metadata": device.metadata,
    }


@toy_router.post("/auth/token")
async def generate_token(
    device: DeviceIdentity = Depends(require_authentication),
    additional_claims: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a new JWT token for the authenticated device."""
    try:
        security_manager = get_security_manager()
        token = security_manager.generate_jwt_token(additional_claims)

        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": security_manager.config.jwt_expiry_seconds,
            "device_id": device.device_id,
        }
    except Exception as e:
        logger.error(f"Failed to generate token: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@toy_router.get("/auth/public-key")
async def get_public_key() -> Dict[str, Any]:
    """Get the device's public key for signature verification."""
    try:
        security_manager = get_security_manager()
        public_key = security_manager.get_public_key_pem()

        if not public_key:
            raise HTTPException(status_code=404, detail="Public key not available")

        return {"public_key": public_key, "algorithm": "RSA", "key_size": 2048}
    except Exception as e:
        logger.error(f"Failed to get public key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@toy_router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics for the toy system."""
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
                await get_engine().edge_optimizer.get_edge_optimization_summary()  # type: ignore
                if get_engine().edge_optimizer is not None
                else {}
            ),
        }
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@toy_router.get("/hardware/status")
async def get_hardware_status() -> Dict[str, Any]:
    """Get hardware status and constraints."""
    try:
        return {
            "constraints": {
                "max_memory_gb": get_hardware_manager().constraints.max_memory_gb,
                "max_cpu_cores": get_hardware_manager().constraints.max_cpu_cores,
                "battery_life_hours": get_hardware_manager().constraints.battery_life_hours,
                "target_latency_ms": get_hardware_manager().constraints.target_latency_ms,
            },
            "power_status": (
                await get_hardware_manager().power_manager.get_power_status()
                if hasattr(get_hardware_manager(), "power_manager")
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


@toy_router.post("/hardware/optimize")
async def optimize_hardware() -> Dict[str, Any]:
    """Optimize hardware for toy deployment."""
    try:
        optimizations = await get_hardware_manager().optimize_for_toy()
        return {
            "success": True,
            "optimizations": optimizations,
            "message": "Hardware optimized for toy deployment",
        }
    except Exception as e:
        logger.error(f"Failed to optimize hardware: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mount routers
app.include_router(toy_router)
app.include_router(metrics_router)
app.include_router(health_router)
app.include_router(performance_router)
app.include_router(streaming_router)
app.include_router(log_search_router)
app.include_router(monitoring_router)
app.include_router(language_router)
app.include_router(multilingual_audio_router)
app.include_router(personalization_router)
app.include_router(parental_controls_router)

# --------- Alias endpoints (unprefixed) for back-compat ----------


@app.get("/health")
async def root_health() -> Dict[str, Any]:
    import time

    # Simple health response for back-compat
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/characters")
async def root_characters() -> Dict[str, Any]:
    return await get_available_characters()


@app.get("/characters/active")
async def root_active_character() -> Dict[str, Any]:
    try:
        engine = get_engine()
        if not engine.character_manager:
            raise HTTPException(
                status_code=500, detail="Character manager not available"
            )
        active = engine.character_manager.get_active_character()
        return {"active_character": active.name if active else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/characters/active")
async def root_set_active_character(request: Dict[str, Any]) -> JSONResponse:
    try:
        name = request.get("character_name") if isinstance(request, dict) else None
        if not name:
            return JSONResponse(
                status_code=400, content={"error": "character_name missing"}
            )
        success = await engine.set_active_character(name)
        if not success:
            return JSONResponse(
                status_code=400, content={"error": f"Character '{name}' not found"}
            )
        return JSONResponse(content={"success": True, "active_character": name})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/hardware/status")
async def root_hardware_status() -> Dict[str, Any]:
    data = await get_hardware_status()
    data.setdefault("status", "ok")
    return data


@app.get("/hardware/sensors")
async def root_hardware_sensors() -> Dict[str, Any]:
    try:
        data = await get_hardware_status()
        return {"sensor_data": data.get("sensor_status", {})}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hardware/power")
async def root_hardware_power() -> Dict[str, Any]:
    try:
        if hasattr(get_hardware_manager(), "power_manager") and hasattr(
            get_hardware_manager().power_manager, "get_power_status"
        ):
            power = await get_hardware_manager().power_manager.get_power_status()
        else:
            power = {}
        return {"power_status": power}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interact")
async def root_interact(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import base64

        from ..core.protocols import AudioData

        # Validate fields
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Invalid request body")
        audio_b64 = payload.get("audio_data")
        character_name = payload.get("character_name")
        if not audio_b64:
            raise HTTPException(status_code=400, detail="audio_data missing")
        if not character_name:
            raise HTTPException(status_code=400, detail="character_name missing")

        # decode audio
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception:
            raise HTTPException(
                status_code=400, detail="Invalid audio_data: not base64"
            )

        # set character
        await engine.set_active_character(character_name)

        from ..core.config import Config

        cfg = Config()
        audio = AudioData(
            data=audio_bytes,
            sample_rate=cfg.interaction.sample_rate,
            channels=cfg.interaction.channels,
            format="wav",
        )
        res = await engine.process_realtime_audio(audio)
        if res.error:
            raise HTTPException(status_code=500, detail=res.error)
        resp_audio_b64 = (
            base64.b64encode(res.audio_data.data).decode() if res.audio_data else ""
        )
        return {"response": res.text, "audio_response": resp_audio_b64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def root_metrics() -> Dict[str, Any]:
    metrics = await real_time_engine.get_performance_metrics()
    return {"metrics": metrics}


@app.get("/models/info")
async def root_models_info() -> Dict[str, Any]:
    return {"models": ["wav2vec2", "llama", "coqui"]}


@app.get("/models/status")
async def root_models_status() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/voices/inject")
async def root_voices_inject(data: Dict[str, Any]) -> Dict[str, str]:
    try:
        import base64
        import tempfile

        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Invalid request body")
        character = data.get("character_name")
        voice_b64 = data.get("voice_data")
        if not character:
            raise HTTPException(status_code=400, detail="character_name missing")
        if not voice_b64:
            raise HTTPException(status_code=400, detail="voice_data missing")
        try:
            voice_bytes = base64.b64decode(voice_b64)
        except Exception:
            raise HTTPException(
                status_code=400, detail="Invalid voice_data: not base64"
            )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(voice_bytes)
            tmp_path = tmp.name
        ok = await engine.inject_character_voice(character, tmp_path)
        if isinstance(ok, dict) and ok.get("success") is False:
            raise HTTPException(
                status_code=500, detail=ok.get("error", "Unknown error")
            )
        return {"success": bool(ok), "character_name": character}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/voices")
async def root_voices() -> Dict[str, Any]:
    voices = await get_engine().list_character_voices()
    return {"voices": voices}


@app.options("/health")
async def root_options_health() -> Response:
    response = Response()
    response.headers["access-control-allow-origin"] = "*"
    response.headers["access-control-allow-methods"] = "GET, POST, OPTIONS"
    response.headers["access-control-allow-headers"] = "Content-Type"
    return response
