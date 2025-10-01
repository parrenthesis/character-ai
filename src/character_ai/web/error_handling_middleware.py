"""
Error handling middleware for crash reporting and graceful error handling.

This middleware provides:
- Global exception handling with crash reporting
- Graceful error recovery
- Error context preservation
- Automatic crash report generation
"""

import time
from typing import Callable

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..core.crash_reporting import get_crash_reporter
from ..core.logging import get_logger

logger = get_logger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for global error handling and crash reporting."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.crash_reporter = get_crash_reporter()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle requests with error catching and crash reporting."""
        try:
            response = await call_next(request)
            return response  # type: ignore
        except HTTPException as http_exc:
            # HTTP exceptions are expected and should not be reported as crashes
            logger.warning(
                "HTTP exception occurred",
                status_code=http_exc.status_code,
                detail=http_exc.detail,
                path=request.url.path,
                method=request.method,
            )
            return JSONResponse(
                status_code=http_exc.status_code, content={"detail": http_exc.detail}
            )
        except Exception as e:
            # Unexpected exceptions should be reported as crashes
            crash_id = self._handle_unexpected_error(e, request)

            logger.error(
                "Unexpected error occurred",
                crash_id=crash_id,
                error_type=type(e).__name__,
                error_message=str(e),
                path=request.url.path,
                method=request.method,
            )

            # Return a generic error response
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "crash_id": crash_id,
                    "timestamp": time.time(),
                },
            )

    def _handle_unexpected_error(self, error: Exception, request: Request) -> str:
        """Handle unexpected errors and report them as crashes."""
        # Determine component based on request path
        component = self._determine_component(request.url.path)

        # Create context with request information
        context = {
            "path": request.url.path,
            "method": request.method,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "request_id": request.headers.get("x-request-id"),
            "trace_id": request.headers.get("x-trace-id"),
        }

        # Report the crash
        crash_id = self.crash_reporter.report_crash(
            error=error, component=component, severity="ERROR", context=context
        )

        return crash_id

    def _determine_component(self, path: str) -> str:
        """Determine the component based on the request path."""
        if path.startswith("/api/v1/toy/auth"):
            return "authentication"
        elif path.startswith("/api/v1/toy/character"):
            return "character_management"
        elif path.startswith("/api/v1/toy/memory"):
            return "session_memory"
        elif path.startswith("/api/v1/toy/safety"):
            return "safety_filter"
        elif path.startswith("/api/v1/toy/audio"):
            return "audio_processing"
        elif path.startswith("/api/v1/toy/hardware"):
            return "hardware_management"
        elif path.startswith("/health"):
            return "health_monitoring"
        elif path.startswith("/metrics"):
            return "metrics_collection"
        else:
            return "api_general"


class GracefulShutdownMiddleware(BaseHTTPMiddleware):
    """Middleware for graceful shutdown handling."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.shutdown_requested = False

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle requests with graceful shutdown support."""
        if self.shutdown_requested:
            return JSONResponse(
                status_code=503, content={"detail": "Service is shutting down"}
            )

        response: Response = await call_next(request)
        return response

    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        self.shutdown_requested = True
        logger.info("Graceful shutdown requested")


class RecoveryMiddleware(BaseHTTPMiddleware):
    """Middleware for error recovery and system health monitoring."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.crash_reporter = get_crash_reporter()
        self.error_count: float = 0.0
        self.last_error_time: float = 0.0
        self.recovery_threshold = 10  # errors per minute
        self.recovery_window = 60  # seconds

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle requests with recovery logic."""
        try:
            response = await call_next(request)

            # Reset error count on successful requests
            if response.status_code < 400:
                self.error_count = max(0, self.error_count - 1)

            return response  # type: ignore
        except Exception:
            # Track error rate
            self.error_count += 1.0
            self.last_error_time = time.time()

            # Check if we're in a degraded state
            if self._is_degraded_state():
                logger.warning(
                    "System in degraded state",
                    error_count=self.error_count,
                    time_window=self.recovery_window,
                )

                # Report degraded state
                self.crash_reporter.report_error(
                    error_type="SystemDegraded",
                    error_message=f"High error rate detected: {self.error_count} errors in {self.recovery_window} seconds",
                    component="recovery_middleware",
                    severity="WARNING",
                    context={
                        "error_count": self.error_count,
                        "time_window": self.recovery_window,
                        "path": request.url.path,
                        "method": request.method,
                    },
                )

            # Re-raise the exception for other middleware to handle
            raise

    def _is_degraded_state(self) -> bool:
        """Check if the system is in a degraded state."""
        current_time = time.time()

        # Reset error count if we're outside the recovery window
        if current_time - self.last_error_time > self.recovery_window:
            self.error_count = 0
            return False

        return self.error_count >= self.recovery_threshold

    def get_health_status(self) -> dict:
        """Get recovery middleware health status."""
        return {
            "error_count": self.error_count,
            "last_error_time": self.last_error_time,
            "is_degraded": self._is_degraded_state(),
            "recovery_threshold": self.recovery_threshold,
            "recovery_window": self.recovery_window,
        }
