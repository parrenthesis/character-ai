"""
Base service implementation with common functionality.

Provides shared patterns for service initialization, processor management,
and error handling across STT, LLM, and TTS services.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ..core.protocols import BaseServiceProtocol
from .error_messages import ServiceErrorMessages

if TYPE_CHECKING:
    from ..core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class BaseService(BaseServiceProtocol, ABC):
    """Base service implementation with common functionality."""

    def __init__(self, resource_manager: "ResourceManager"):
        """Initialize base service with resource manager."""
        self.resource_manager = resource_manager
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the service."""
        if not self._initialized:
            await self._do_initialize()
            self._initialized = True
            logger.info(f"{self.__class__.__name__} initialized")

    async def shutdown(self) -> None:
        """Shutdown the service."""
        if self._initialized:
            await self._do_shutdown()
            self._initialized = False
            logger.info(f"{self.__class__.__name__} shutdown")

    async def health_check(self) -> bool:
        """Check if the service is healthy."""
        try:
            return self._initialized and await self._do_health_check()
        except Exception as e:
            logger.error(f"Health check failed for {self.__class__.__name__}: {e}")
            return False

    async def get_or_create_processor(self, model_type: str) -> Any:
        """Get or create a processor for the given model type.

        Common pattern used by STT, LLM, and TTS services to handle
        processor initialization with fallback logic.

        Args:
            model_type: Type of model ('stt', 'llm', 'tts')

        Returns:
            Processor instance

        Raises:
            RuntimeError: If processor cannot be initialized
        """
        # Get processor from ResourceManager
        processor = self._get_processor(model_type)
        if processor is None:
            # Use ResourceManager's preload_models to create processor
            await self.resource_manager.preload_models([model_type])
            processor = self._get_processor(model_type)

        if processor is None:
            error_msg = ServiceErrorMessages.get_processor_error(model_type)
            raise RuntimeError(error_msg)

        # Mark model as used
        self.resource_manager.mark_model_used(model_type)
        return processor

    @abstractmethod
    def _get_processor(self, model_type: str) -> Any:
        """Get processor for the given model type from resource manager."""
        pass

    async def _do_initialize(self) -> None:
        """Service-specific initialization logic."""
        pass

    async def _do_shutdown(self) -> None:
        """Service-specific shutdown logic."""
        pass

    async def _do_health_check(self) -> bool:
        """Service-specific health check logic."""
        return True
