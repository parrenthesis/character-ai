"""
Character AI API endpoints.

Provides endpoints for character management, real-time interaction, and system controls.
"""

from typing import Any

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from ...hardware.toy_hardware_manager import ToyHardwareManager
from ...observability import get_logger
from ...production.real_time_engine import RealTimeInteractionEngine
from ..features.language_api import router as language_router
from ..features.multilingual_audio_api import router as multilingual_audio_router
from ..features.parental_controls_api import router as parental_controls_router
from ..middleware.error_handling_middleware import (
    ErrorHandlingMiddleware,
    RecoveryMiddleware,
)
from ..middleware.logging_middleware import (
    LoggingMiddleware,
    PerformanceLoggingMiddleware,
    SecurityLoggingMiddleware,
)
from ..monitoring.log_search_api import log_search_router
from ..monitoring.metrics_api import metrics_router
from ..monitoring.monitoring_api import monitoring_router
from ..monitoring.performance_api import performance_router
from ..security_deps import add_security_headers

# Import the modular character API routers
from .character_endpoints.auth import auth_router
from .character_endpoints.config import config_router
from .character_endpoints.health import health_router as character_health_router
from .character_endpoints.interaction import interaction_router
from .character_endpoints.session import session_router
from .health_api import health_router
from .streaming_api import streaming_router

logger = get_logger(__name__)

# Create app
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


def get_hardware_manager() -> ToyHardwareManager:
    """Dependency to get hardware manager instance."""
    return ToyHardwareManager()


def get_engine() -> RealTimeInteractionEngine:
    """Dependency to get real-time engine instance."""
    return RealTimeInteractionEngine(get_hardware_manager())


def get_security_manager() -> Any:
    """Get the security manager instance."""
    from ..security_deps import get_security_manager as _get_security_manager

    return _get_security_manager()


# Startup event
@app.on_event("startup")
async def startup_event() -> None:
    """Initialize character systems on startup if explicitly enabled.

    To avoid blocking tests or environments without models/hardware, startup
    initialization is disabled by default. Set CAI_ENABLE_API_STARTUP=1 to enable.
    """
    import os

    if os.getenv("CAI_ENABLE_API_STARTUP") == "1":
        logger.info("API startup enabled - initializing character systems")
        try:
            # Initialize hardware manager
            hardware_manager = get_hardware_manager()
            await hardware_manager.initialize()

            # Initialize real-time engine
            engine = get_engine()
            await engine.initialize()

            logger.info("Character systems initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize character systems: {e}")
            # Don't fail startup - allow API to run without full initialization
    else:
        logger.info(
            "API startup disabled - character systems will initialize on first use"
        )


# Mount all routers
app.include_router(interaction_router)
app.include_router(session_router)
app.include_router(character_health_router)
app.include_router(config_router)
app.include_router(auth_router)
app.include_router(metrics_router)
app.include_router(health_router)
app.include_router(performance_router)
app.include_router(streaming_router)
app.include_router(log_search_router)
app.include_router(monitoring_router)
app.include_router(language_router)
app.include_router(multilingual_audio_router)
app.include_router(parental_controls_router)


# Root endpoint
@app.get("/")
async def root() -> JSONResponse:
    """Root endpoint with API information."""
    return JSONResponse(
        content={
            "name": "Character AI API",
            "version": "1.0.0",
            "description": "Character interaction and management API",
            "endpoints": {
                "health": "/health",
                "characters": "/api/v1/character/characters",
                "interact": "/api/v1/character/interact",
                "docs": "/docs",
            },
        }
    )


# CORS options endpoint
@app.options("/health")
async def root_options_health() -> Response:
    """CORS options for health endpoint."""
    response = Response()
    response.headers["access-control-allow-origin"] = "*"
    response.headers["access-control-allow-methods"] = "GET, POST, OPTIONS"
    response.headers["access-control-allow-headers"] = "Content-Type"
    return response
