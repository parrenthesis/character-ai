"""
Protocols and interfaces for Character AI Platform.

Defines the contracts that all components must implement for type-safe
interaction and flexible architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Protocol, TypeVar

# Type variables for generic protocols
T = TypeVar("T")


@dataclass
class AudioData:
    """Audio data container."""

    data: bytes
    sample_rate: int
    channels: int
    duration: float = 0.0
    format: str = "wav"
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AudioResult:
    """Result from audio processing."""

    audio_data: Optional[AudioData] = None
    text: Optional[str] = None
    embeddings: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    error: Optional[str] = None
    quality_score: Optional[float] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TextResult:
    """Result from text processing."""

    text: str
    embeddings: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    error: Optional[str] = None
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""

    embeddings: List[float]
    metadata: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    error: Optional[str] = None
    dimension: Optional[int] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}
        if self.dimension is None:
            self.dimension = len(self.embeddings)


@dataclass
class ModelInfo:
    """Information about a loaded model."""

    name: str
    type: str
    size: str
    memory_usage: str
    precision: str
    quantization: str
    loaded_at: float
    status: str = "loaded"


# Core Protocols


class AudioProcessor(Protocol):
    """Protocol for audio processing components."""

    async def process_audio(self, audio: AudioData) -> AudioResult:
        """Process audio data and return result."""
        ...

    async def get_embeddings(self, audio: AudioData) -> EmbeddingResult:
        """Extract embeddings from audio data."""
        ...

    async def get_model_info(self) -> ModelInfo:
        """Get information about the loaded model."""
        ...

    async def is_ready(self) -> bool:
        """Check if the processor is ready to handle requests."""
        ...


class TextProcessor(Protocol):
    """Protocol for text processing components."""

    async def process_text(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> TextResult:
        """Process text and return result."""
        ...

    async def get_embeddings(self, text: str) -> EmbeddingResult:
        """Extract embeddings from text."""
        ...

    async def generate_text(self, prompt: str, **kwargs: Any) -> TextResult:
        """Generate text from prompt."""
        ...

    async def get_model_info(self) -> ModelInfo:
        """Get information about the loaded model."""
        ...


class VoiceSynthesizer(Protocol):
    """Protocol for voice synthesis components."""

    async def synthesize(
        self, text: str, voice_style: Optional[str] = None
    ) -> AudioResult:
        """Synthesize speech from text."""
        ...

    async def clone_voice(self, reference_audio: AudioData, text: str) -> AudioResult:
        """Clone voice from reference audio."""
        ...

    async def get_available_voices(self) -> List[str]:
        """Get list of available voice styles."""
        ...

    async def get_model_info(self) -> ModelInfo:
        """Get information about the loaded model."""
        ...


class MultimodalProcessor(Protocol):
    """Protocol for multimodal processing components."""

    async def process_audio_text(self, audio: AudioData, text: str) -> AudioResult:
        """Process audio and text together."""
        ...

    async def fuse_embeddings(
        self, audio_embeddings: List[float], text_embeddings: List[float]
    ) -> EmbeddingResult:
        """Fuse audio and text embeddings."""
        ...

    async def get_context_understanding(
        self, audio: AudioData, text: str
    ) -> TextResult:
        """Get contextual understanding from audio and text."""
        ...

    async def get_model_info(self) -> ModelInfo:
        """Get information about the loaded model."""
        ...


class ModelManager(Protocol):
    """Protocol for model management components."""

    async def load_model(self, model_name: str, **kwargs: Any) -> bool:
        """Load a model by name."""
        ...

    async def unload_model(self, model_name: str) -> bool:
        """Unload a model by name."""
        ...

    async def get_loaded_models(self) -> List[ModelInfo]:
        """Get list of currently loaded models."""
        ...

    async def get_model_status(self, model_name: str) -> str:
        """Get status of a specific model."""
        ...

    async def optimize_for_gpu(
        self, model_name: str, gpu_config: Dict[str, Any]
    ) -> bool:
        """Optimize model for GPU configuration."""
        ...


class GPUManager(Protocol):
    """Protocol for GPU resource management."""

    async def get_available_memory(self) -> str:
        """Get available GPU memory."""
        ...

    async def get_total_memory(self) -> str:
        """Get total GPU memory."""
        ...

    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information."""
        ...

    async def optimize_memory(self) -> bool:
        """Optimize GPU memory usage."""
        ...

    async def clear_cache(self) -> bool:
        """Clear GPU cache."""
        ...


class RequestHandler(Protocol):
    """Protocol for request handling components."""

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request and return response."""
        ...

    async def validate_request(self, request: Dict[str, Any]) -> bool:
        """Validate request format and content."""
        ...

    async def get_request_metrics(self) -> Dict[str, Any]:
        """Get metrics about handled requests."""
        ...


class Monitor(Protocol):
    """Protocol for monitoring and metrics components."""

    async def track_metric(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Track a metric value."""
        ...

    async def track_latency(self, operation: str, latency: float) -> None:
        """Track operation latency."""
        ...

    async def track_error(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track an error occurrence."""
        ...

    async def get_metrics(self, time_range: Optional[str] = None) -> Dict[str, Any]:
        """Get collected metrics."""
        ...

    async def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        ...


# Streaming Protocols


class StreamingAudioProcessor(Protocol):
    """Protocol for streaming audio processing."""

    async def process_stream(
        self, audio_stream: AsyncGenerator[AudioData, None]
    ) -> AsyncGenerator[AudioResult, None]:
        """Process streaming audio data."""
        ...

    async def start_stream(self) -> None:
        """Start streaming processing."""
        ...

    async def stop_stream(self) -> None:
        """Stop streaming processing."""
        ...


class StreamingTextProcessor(Protocol):
    """Protocol for streaming text processing."""

    async def process_stream(
        self, text_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[TextResult, None]:
        """Process streaming text data."""
        ...

    async def generate_stream(
        self, prompt: str, **kwargs: Any
    ) -> AsyncGenerator[TextResult, None]:
        """Generate streaming text from prompt."""
        ...


# Base Classes for Implementation


class BaseComponent(ABC):
    """Base class for all Character AI Platform components."""

    def __init__(self, name: str, config: Optional[Any] = None) -> None:
        self.name = name
        self.config = config or {}
        self._initialized = False
        self._shutdown = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the component."""
        pass

    async def is_ready(self) -> bool:
        """Check if component is ready."""
        return self._initialized and not self._shutdown

    async def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "shutdown": self._shutdown,
            "ready": await self.is_ready(),
        }


class BaseAudioProcessor(BaseComponent, AudioProcessor):
    """Base implementation for audio processors."""

    def __init__(self, name: str, config: Optional[Any] = None) -> None:
        super().__init__(name, config)
        self.model_info: Optional[ModelInfo] = None

    async def get_model_info(self) -> ModelInfo:
        """Get model information."""
        if self.model_info is None:
            raise RuntimeError("Model not loaded")
        return self.model_info


class BaseTextProcessor(BaseComponent, TextProcessor):
    """Base implementation for text processors."""

    def __init__(self, name: str, config: Optional[Any] = None) -> None:
        super().__init__(name, config)
        self.model_info: Optional[ModelInfo] = None

    async def get_model_info(self) -> ModelInfo:
        """Get model information."""
        if self.model_info is None:
            raise RuntimeError("Model not loaded")
        return self.model_info


# Utility Functions


async def validate_audio_data(audio: AudioData) -> bool:
    """Validate audio data format and content."""
    if not audio.data:
        return False
    if audio.sample_rate <= 0:
        return False
    if audio.duration <= 0:
        return False
    if audio.channels <= 0:
        return False
    return True


async def validate_text_data(text: str) -> bool:
    """Validate text data."""
    if not text or not text.strip():
        return False
    if len(text) > 10000:  # Reasonable limit
        return False
    return True


async def create_error_result(error_message: str, component: str) -> AudioResult:
    """Create an error result."""
    return AudioResult(
        error=error_message, metadata={"component": component, "error": True}
    )


async def create_success_result(
    audio_data: Optional[AudioData] = None,
    text: Optional[str] = None,
    embeddings: Optional[List[float]] = None,
    processing_time: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> AudioResult:
    """Create a success result."""
    return AudioResult(
        audio_data=audio_data,
        text=text,
        embeddings=embeddings,
        processing_time=processing_time,
        metadata=metadata or {},
        error=None,
    )
