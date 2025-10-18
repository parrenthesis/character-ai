"""Engine lifecycle management for real-time interaction."""

import asyncio
import logging
from typing import Any, Optional

from ...algorithms.conversational_ai.session_memory import SessionMemory
from ...algorithms.conversational_ai.text_normalizer import TextNormalizer
from ...characters import (
    Character,
    CharacterResponseFilter,
    CharacterService,
    ChildSafetyFilter,
)
from ...characters.voices import SchemaVoiceService
from ...core.audio_io.voice_activity_detection import VADConfig
from ...core.caching import ResponseCache
from ...core.config import Config
from ...core.llm.template_prompt_builder import TemplatePromptBuilder
from ...core.resource_manager import ResourceManager
from ...hardware.toy_hardware_manager import ToyHardwareManager
from ...services import HardwareProfileService
from ...services.edge_optimizer import EdgeModelOptimizer

logger = logging.getLogger(__name__)


class EngineLifecycle:
    """Manages engine initialization, shutdown, and session state."""

    def __init__(
        self,
        hardware_manager: ToyHardwareManager,
        text_normalizer: Optional[Any] = None,
        prompt_builder: Optional[Any] = None,
        resource_manager: Optional[Any] = None,
        hardware_profile: Optional[str] = None,
    ):
        self.hardware_manager: Optional[ToyHardwareManager] = hardware_manager
        self.character_manager: Optional[CharacterService] = CharacterService()

        # Use HardwareProfileService for unified hardware detection
        hardware_service = HardwareProfileService()
        self.hardware_profile, self.hardware_config = hardware_service.load_or_detect(
            hardware_profile
        )

        # Create or inject services
        if text_normalizer is None:
            self.text_normalizer = TextNormalizer()
        else:
            self.text_normalizer = text_normalizer

        if prompt_builder is None:
            self.prompt_builder = TemplatePromptBuilder()
        else:
            self.prompt_builder = prompt_builder

        # Character-specific response filter (initialized per character)
        self.character_response_filter: Optional[CharacterResponseFilter] = None

        if resource_manager is None:
            # edge_optimizer will be set later, so pass None for now
            # Pass hardware config to ResourceManager for proper device selection
            self.resource_manager = ResourceManager(
                Config(), None, self.hardware_config
            )
        else:
            self.resource_manager = resource_manager

        # Initialize safety filter and voice manager directly
        self.safety_filter: Optional[ChildSafetyFilter] = ChildSafetyFilter()
        self.voice_manager: Optional[SchemaVoiceService] = SchemaVoiceService()
        self.edge_optimizer: Optional[EdgeModelOptimizer] = (
            EdgeModelOptimizer(hardware_manager.constraints)
            if hardware_manager
            else None
        )

        # Update ResourceManager with edge_optimizer if it was created
        if self.resource_manager and self.edge_optimizer:
            self.resource_manager.edge_optimizer = self.edge_optimizer

        # Session memory management
        self.session_memory = SessionMemory()

        # Response cache
        self.response_cache = ResponseCache()

        # Active character
        self.active_character: Optional[Character] = None

        # VAD configuration
        self.vad_config = VADConfig()

        # Real-time session state
        self.is_realtime_session_active = False
        self.realtime_session_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize the real-time interaction engine."""
        try:
            logger.info("Initializing real-time interaction engine...")

            # Initialize character manager
            if self.character_manager:
                await self.character_manager.initialize()

            # Initialize safety filter
            if self.safety_filter:
                await self.safety_filter.initialize()

            # Initialize voice manager
            if self.voice_manager:
                await self.voice_manager.initialize()

            logger.info("Real-time interaction engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize real-time engine: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the engine and clean up resources."""
        try:
            logger.info("Shutting down real-time interaction engine...")

            # Stop real-time session if active
            if self.is_realtime_session_active:
                await self._stop_realtime_session()

            # Shutdown services
            if self.resource_manager:
                # Unload all models
                await self.resource_manager.unload_idle_models()

            # Shutdown character manager
            if self.character_manager:
                await self.character_manager.shutdown()

            # Shutdown safety filter
            if self.safety_filter:
                await self.safety_filter.shutdown()

            # Shutdown voice manager
            if self.voice_manager:
                await self.voice_manager.shutdown()

            logger.info("Real-time interaction engine shutdown complete")
        except Exception as e:
            logger.error(f"Error during engine shutdown: {e}")

    async def _stop_realtime_session(self) -> None:
        """Stop the real-time session."""
        if self.realtime_session_task and not self.realtime_session_task.done():
            self.realtime_session_task.cancel()
            try:
                await self.realtime_session_task
            except asyncio.CancelledError:
                pass
        self.is_realtime_session_active = False
        self.realtime_session_task = None
        logger.info("Real-time session stopped")

    def set_active_character(self, character: Character) -> None:
        """Set the active character for interaction."""
        self.active_character = character
        logger.info(f"Active character set to: {character.name}")

    def get_active_character(self) -> Optional[Character]:
        """Get the currently active character."""
        return self.active_character
