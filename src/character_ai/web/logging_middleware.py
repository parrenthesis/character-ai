"""
FastAPI middleware for request/trace ID correlation and structured logging.

Provides automatic request ID generation, correlation tracking, and structured
logging for all API requests and responses.
"""

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from character_ai.core.logging import (
    clear_request_context,
    generate_request_id,
    generate_trace_id,
    get_logger,
    set_request_context,
)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request correlation and structured logging."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger("api.middleware")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with correlation IDs and structured logging."""

        # Generate correlation IDs
        request_id = generate_request_id()
        trace_id = generate_trace_id()

        # Extract device ID from headers if available
        device_id = request.headers.get("x-device-id")

        # Set request context
        set_request_context(
            request_id=request_id, trace_id=trace_id, device_id=device_id
        )

        # Log request start
        self.logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        # Process request
        start_time = time.time()

        try:
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log successful response
            self.logger.log_api_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
                response_size=response.headers.get("content-length"),
            )

            # Add correlation headers to response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Trace-ID"] = trace_id

            return response  # type: ignore

        except Exception as e:
            # Calculate duration for error case
            duration_ms = (time.time() - start_time) * 1000

            # Log error
            self.logger.error(
                "Request failed",
                method=request.method,
                path=request.url.path,
                duration_ms=duration_ms,
                error=str(e),
                error_type=type(e).__name__,
            )

            # Re-raise the exception
            raise

        finally:
            # Clear request context
            clear_request_context()


class SecurityLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for security-related logging."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger("security.middleware")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log security-related events."""

        # Log authentication attempts
        auth_header = request.headers.get("authorization")
        if auth_header:
            self.logger.info(
                "Authentication attempt",
                method=request.method,
                path=request.url.path,
                auth_type=auth_header.split()[0] if " " in auth_header else "unknown",
            )

        # Log admin token usage
        admin_token = request.headers.get("x-admin-token")
        if admin_token:
            self.logger.warning(
                "Admin token usage",
                method=request.method,
                path=request.url.path,
                client_ip=request.client.host if request.client else None,
            )

        # Process request
        response: Response = await call_next(request)

        # Log security events based on response
        if response.status_code == 401:
            self.logger.warning(
                "Authentication failed",
                method=request.method,
                path=request.url.path,
                client_ip=request.client.host if request.client else None,
            )
        elif response.status_code == 403:
            self.logger.warning(
                "Authorization failed",
                method=request.method,
                path=request.url.path,
                client_ip=request.client.host if request.client else None,
            )
        elif response.status_code >= 500:
            self.logger.error(
                "Server error",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                client_ip=request.client.host if request.client else None,
            )

        return response


class PerformanceLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and logging."""

    def __init__(self, app: ASGIApp, slow_request_threshold_ms: float = 1000.0):
        super().__init__(app)
        self.logger = get_logger("performance.middleware")
        self.slow_request_threshold_ms = slow_request_threshold_ms

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor and log performance metrics."""

        start_time = time.time()

        # Process request
        response: Response = await call_next(request)

        # Calculate metrics
        duration_ms = (time.time() - start_time) * 1000

        # Log performance metrics
        self.logger.info(
            "Performance metrics",
            method=request.method,
            path=request.url.path,
            duration_ms=duration_ms,
            status_code=response.status_code,
            is_slow=duration_ms > self.slow_request_threshold_ms,
        )

        # Log slow requests
        if duration_ms > self.slow_request_threshold_ms:
            self.logger.warning(
                "Slow request detected",
                method=request.method,
                path=request.url.path,
                duration_ms=duration_ms,
                threshold_ms=self.slow_request_threshold_ms,
            )

        return response
