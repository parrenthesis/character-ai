"""
Structured logging configuration with request/trace ID correlation.

Provides centralized logging configuration with correlation IDs for tracking
requests across all components (STT, LLM, TTS, Safety, API).
"""

import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

import structlog
from structlog.stdlib import LoggerFactory

# Context variables for request correlation
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
device_id_var: ContextVar[Optional[str]] = ContextVar("device_id", default=None)
character_id_var: ContextVar[Optional[str]] = ContextVar("character_id", default=None)


class StructuredLogger:
    """Structured logger with request correlation support and aggregation."""

    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
        self._aggregation_enabled = True

    def _get_context(self) -> Dict[str, Any]:
        """Get current request context for logging."""
        context = {}

        if request_id := request_id_var.get():
            context["request_id"] = request_id
        if trace_id := trace_id_var.get():
            context["trace_id"] = trace_id
        if device_id := device_id_var.get():
            context["device_id"] = device_id
        if character_id := character_id_var.get():
            context["character_id"] = character_id

        return context

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with context."""
        self.logger.debug(message, **self._get_context(), **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with context."""
        self.logger.info(message, **self._get_context(), **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with context."""
        self.logger.warning(message, **self._get_context(), **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with context."""
        self.logger.error(message, **self._get_context(), **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message with context."""
        self.logger.critical(message, **self._get_context(), **kwargs)

    def log_processing_step(
        self,
        step: str,
        component: str,
        duration_ms: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Log a processing step with timing information."""
        log_data = {
            "step": step,
            "component": component,
            **self._get_context(),
            **kwargs,
        }

        if duration_ms is not None:
            log_data["duration_ms"] = duration_ms

        self.logger.info(f"Processing step: {step}", **log_data)

    def log_api_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        **kwargs: Any,
    ) -> None:
        """Log API request with timing and status."""
        self.logger.info(
            f"API request: {method} {path}",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            **self._get_context(),
            **kwargs,
        )

    def log_safety_event(
        self, event_type: str, confidence: float, content: str, **kwargs: Any
    ) -> None:
        """Log safety-related events."""
        self.logger.warning(
            f"Safety event: {event_type}",
            event_type=event_type,
            confidence=confidence,
            content_length=len(content),
            **self._get_context(),
            **kwargs,
        )

    def log_character_interaction(
        self, interaction_type: str, character_id: str, **kwargs: Any
    ) -> None:
        """Log character interactions."""
        self.logger.info(
            f"Character interaction: {interaction_type}",
            interaction_type=interaction_type,
            character_id=character_id,
            **self._get_context(),
            **kwargs,
        )

    async def _send_to_aggregation(
        self, level: str, message: str, **kwargs: Any
    ) -> None:
        """Send log entry to aggregation system."""
        if not self._aggregation_enabled:
            return

        try:
            # Import here to avoid circular imports
            from .log_aggregation import LogLevel, LogSource, add_log_entry

            # Map log level
            level_map = {
                "DEBUG": LogLevel.DEBUG,
                "INFO": LogLevel.INFO,
                "WARNING": LogLevel.WARNING,
                "ERROR": LogLevel.ERROR,
                "CRITICAL": LogLevel.CRITICAL,
            }

            # Map source based on logger name
            source_map = {
                "api": LogSource.API,
                "security": LogSource.SECURITY,
                "performance": LogSource.PERFORMANCE,
                "safety": LogSource.SAFETY,
                "character": LogSource.CHARACTER,
                "audio": LogSource.AUDIO,
                "llm": LogSource.LLM,
                "system": LogSource.SYSTEM,
            }

            log_level = level_map.get(level, LogLevel.INFO)
            log_source = LogSource.APPLICATION

            for key, source in source_map.items():
                if key in self.logger.name.lower():
                    log_source = source
                    break

            # Send to aggregation
            await add_log_entry(
                level=log_level,
                source=log_source,
                message=message,
                logger_name=self.logger.name,
                **kwargs,
            )

        except Exception as e:
            # Don't let aggregation errors break logging
            print(f"Log aggregation error: {e}")


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)


def set_request_context(
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    device_id: Optional[str] = None,
    character_id: Optional[str] = None,
) -> None:
    """Set request context for correlation."""
    if request_id:
        request_id_var.set(request_id)
    if trace_id:
        trace_id_var.set(trace_id)
    if device_id:
        device_id_var.set(device_id)
    if character_id:
        character_id_var.set(character_id)


def clear_request_context() -> None:
    """Clear request context."""
    request_id_var.set(None)
    trace_id_var.set(None)
    device_id_var.set(None)
    character_id_var.set(None)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def generate_trace_id() -> str:
    """Generate a unique trace ID."""
    return str(uuid.uuid4())


def configure_logging(level: str = "INFO", json_format: bool = True) -> None:
    """Configure structured logging for the application."""

    # Configure structlog
    processors: List[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.add_log_level,
    ]

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level.upper())
        ),
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )


class ProcessingTimer:
    """Context manager for timing processing steps."""

    def __init__(
        self, logger: StructuredLogger, step: str, component: str, **kwargs: Any
    ):
        self.logger = logger
        self.step = step
        self.component = component
        self.kwargs = kwargs
        self.start_time: Optional[float] = None
        self.duration_ms: Optional[float] = None

    def __enter__(self) -> Any:
        self.start_time = time.time()
        self.logger.log_processing_step(
            f"{self.step}_start", self.component, **self.kwargs
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time:
            self.duration_ms = (time.time() - self.start_time) * 1000
            status = "success" if exc_type is None else "error"

            self.logger.log_processing_step(
                f"{self.step}_end",
                self.component,
                duration_ms=self.duration_ms,
                status=status,
                **self.kwargs,
            )


# Initialize logging configuration
configure_logging()
