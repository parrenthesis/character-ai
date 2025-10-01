"""
Exception hierarchy for VoiceAI Research Toolkit.

Provides structured error handling with specific error types for different
components and failure modes.
"""

from typing import Any, Callable, Dict, Optional, TypeVar

F = TypeVar('F', bound=Callable[..., Any])


class VoiceAIError(Exception):
    """Base exception for all VoiceAI-related errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        component: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.component = component

    def __str__(self) -> str:
        """String representation of the error."""
        base_msg = f"[{self.component or 'VoiceAI'}] {self.message}"
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        return base_msg

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "component": self.component,
            "details": self.details,
        }


class ConfigurationError(VoiceAIError):
    """Exception raised when configuration is invalid or missing."""

    pass


class GPUError(VoiceAIError):
    """Exception raised when GPU operations fail."""

    pass


class AlgorithmError(VoiceAIError):
    """Exception raised when algorithm operations fail."""

    pass


class AudioProcessingError(VoiceAIError):
    """Exception raised when audio processing fails."""

    pass


class ModelError(VoiceAIError):
    """Exception raised when model operations fail."""

    pass


class InferenceError(VoiceAIError):
    """Exception raised when inference operations fail."""

    pass


class APIError(VoiceAIError):
    """Exception raised when API operations fail."""

    pass


class ResourceError(VoiceAIError):
    """Exception raised when resource allocation fails."""

    pass


# Specific error types for common failure modes


class ModelNotFoundError(ModelError):
    """Exception raised when a model is not found."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__(
            f"Model not found: {model_name}",
            error_code="MODEL_NOT_FOUND",
            details={"model_name": model_name},
            **kwargs,
        )


class ModelLoadError(ModelError):
    """Exception raised when model loading fails."""

    def __init__(self, model_name: str, reason: str, **kwargs: Any) -> None:
        super().__init__(
            f"Failed to load model {model_name}: {reason}",
            error_code="MODEL_LOAD_ERROR",
            details={"model_name": model_name, "reason": reason},
            **kwargs,
        )


class GPUOutOfMemoryError(GPUError):
    """Exception raised when GPU runs out of memory."""

    def __init__(self, requested_memory: str, available_memory: str, **kwargs: Any) -> None:
        super().__init__(
            f"GPU out of memory: requested {requested_memory}, available {available_memory}",
            error_code="GPU_OUT_OF_MEMORY",
            details={
                "requested_memory": requested_memory,
                "available_memory": available_memory,
            },
            **kwargs,
        )


class AudioFormatError(AudioProcessingError):
    """Exception raised when audio format is unsupported."""

    def __init__(self, audio_format: str, supported_formats: list, **kwargs: Any) -> None:
        super().__init__(
            f"Unsupported audio format: {audio_format}. Supported: {supported_formats}",

            error_code="AUDIO_FORMAT_ERROR",
            details={
                "audio_format": audio_format,
                "supported_formats": supported_formats,
            },
            **kwargs,
        )


class AudioLengthError(AudioProcessingError):
    """Exception raised when audio is too long or too short."""

    def __init__(
        self, audio_length: float, min_length: float, max_length: float, **kwargs: Any
    ) -> None:
        super().__init__(
            f"Audio length {audio_length}s is outside range [{min_length}s, {max_length}s]",
            error_code="AUDIO_LENGTH_ERROR",
            details={
                "audio_length": audio_length,
                "min_length": min_length,
                "max_length": max_length,
            },
            **kwargs,
        )


class InferenceTimeoutError(InferenceError):
    """Exception raised when inference times out."""

    def __init__(self, timeout_seconds: float, **kwargs: Any) -> None:
        super().__init__(
            f"Inference timed out after {timeout_seconds} seconds",
            error_code="INFERENCE_TIMEOUT",
            details={"timeout_seconds": timeout_seconds},
            **kwargs,
        )


class RateLimitError(APIError):
    """Exception raised when API rate limit is exceeded."""

    def __init__(self, rate_limit: int, window_seconds: int, **kwargs: Any) -> None:
        super().__init__(
            f"Rate limit exceeded: {rate_limit} requests per {window_seconds} seconds",
            error_code="RATE_LIMIT_EXCEEDED",
            details={
                "rate_limit": rate_limit,
                "window_seconds": window_seconds,
            },
            **kwargs,
        )


class ValidationError(VoiceAIError):
    """Exception raised when input validation fails."""

    def __init__(self, field: str, value: Any, reason: str, **kwargs: Any) -> None:
        super().__init__(
            f"Validation failed for field '{field}': {reason}",
            error_code="VALIDATION_ERROR",
            details={
                "field": field,
                "value": str(value),
                "reason": reason,
            },
            **kwargs,
        )


class ResourceExhaustedError(ResourceError):
    """Exception raised when system resources are exhausted."""

    def __init__(
        self, resource_type: str, current_usage: str, limit: str, **kwargs: Any
    ) -> None:
        super().__init__(
            f"{resource_type} resource exhausted: {current_usage}/{limit}",
            error_code="RESOURCE_EXHAUSTED",
            details={
                "resource_type": resource_type,
                "current_usage": current_usage,
                "limit": limit,
            },
            **kwargs,
        )


# Error handling utilities


def handle_gpu_error(func: F) -> F:
    """Decorator to handle GPU-related errors."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise GPUOutOfMemoryError(
                    requested_memory="unknown",
                    available_memory="unknown",
                    component=func.__name__,
                ) from e
            else:
                raise GPUError(
                    message=f"GPU error in {func.__name__}: {str(e)}",
                    component=func.__name__,
                ) from e

    return wrapper  # type: ignore


def handle_model_error(func: F) -> F:
    """Decorator to handle model-related errors."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise ModelNotFoundError(
                model_name="unknown", component=func.__name__
            ) from e
        except Exception as e:
            raise ModelError(
                message=f"Model error in {func.__name__}: {str(e)}",
                component=func.__name__,
            ) from e

    return wrapper  # type: ignore


def handle_audio_error(func: F) -> F:
    """Decorator to handle audio processing errors."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise AudioProcessingError(
                message=f"Audio processing error in {func.__name__}: {str(e)}",
                component=func.__name__,
            ) from e

    return wrapper  # type: ignore
