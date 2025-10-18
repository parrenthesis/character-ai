"""
Core real-time interaction engine for character.ai.

Handles initialization, main processing, and shutdown.
"""

# CRITICAL: Import torch_init FIRST to set environment variables before any torch imports
# isort: off
from ...core import torch_init  # noqa: F401

# isort: on

import time
from typing import Any, Dict, Optional

from ...core.protocols import AudioData, AudioResult
from ...hardware.toy_hardware_manager import ToyHardwareManager
from ...observability import get_logger
from .audio_system import AudioSystem
from .engine_lifecycle import EngineLifecycle
from .performance_monitor import PerformanceMonitor
from .processing_pipeline import ProcessingPipeline

logger = get_logger(__name__)


class CoreRealTimeEngine:
    """Core engine for real-time toy interaction."""

    def __init__(
        self,
        hardware_manager: ToyHardwareManager,
        # Optional dependency injection for new services
        text_normalizer: Optional[Any] = None,
        prompt_builder: Optional[Any] = None,
        resource_manager: Optional[Any] = None,
        hardware_profile: Optional[str] = None,
    ):
        # Initialize component systems
        self.lifecycle = EngineLifecycle(
            hardware_manager=hardware_manager,
            text_normalizer=text_normalizer,
            prompt_builder=prompt_builder,
            resource_manager=resource_manager,
            hardware_profile=hardware_profile,
        )

        self.audio_system = AudioSystem(hardware_manager)
        self.performance_monitor = PerformanceMonitor()

        # Processing pipeline will be initialized after lifecycle
        self.processing_pipeline: Optional[ProcessingPipeline] = None

    async def initialize(self) -> None:
        """Initialize the real-time interaction engine."""
        try:
            logger.info("Initializing core real-time engine...")

            # Initialize lifecycle components
            await self.lifecycle.initialize()

            # Initialize audio system
            await self.audio_system.initialize()

            # Initialize processing pipeline
            self.processing_pipeline = ProcessingPipeline(
                resource_manager=self.lifecycle.resource_manager,
                text_normalizer=self.lifecycle.text_normalizer,
                prompt_builder=self.lifecycle.prompt_builder,
                session_memory=self.lifecycle.session_memory,
                voice_manager=self.lifecycle.voice_manager
                or self._create_default_voice_manager(),
                safety_filter=self.lifecycle.safety_filter
                or self._create_default_safety_filter(),
                response_cache=self.lifecycle.response_cache,
            )
            await self.processing_pipeline.initialize()

            logger.info("Core real-time engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize core real-time engine: {e}")
            raise

    def _create_default_voice_manager(self) -> Any:
        """Create a default voice manager if none is provided."""
        from ...characters.voices.schema_voice_manager import SchemaVoiceService

        return SchemaVoiceService()

    def _create_default_safety_filter(self) -> Any:
        """Create a default safety filter if none is provided."""
        from ...characters.safety.safety_filter import ChildSafetyFilter

        return ChildSafetyFilter()

    async def preload_models(self) -> Dict[str, bool]:
        """Pre-load all models for real-time interaction."""
        if self.processing_pipeline:
            return await self.processing_pipeline.preload_models()
        return {"stt": False, "llm": False, "tts": False}

    async def process_realtime_audio(self, audio_stream: AudioData) -> AudioResult:
        """Process audio in real-time for natural interaction."""
        start_time = time.time()

        try:
            if not self.lifecycle.active_character:
                raise ValueError("No active character set")

            if not self.processing_pipeline:
                raise ValueError("Processing pipeline not initialized")

            # Process audio through pipeline
            result = await self.processing_pipeline.process_realtime_audio(
                audio_stream, self.lifecycle.active_character
            )

            # Update performance metrics
            latency = time.time() - start_time
            success = result.error is None
            self.performance_monitor.update_metrics(latency, success)

            return result

        except Exception as e:
            logger.error(f"Error processing real-time audio: {e}")
            latency = time.time() - start_time
            self.performance_monitor.update_metrics(latency, False)
            raise

    async def process_audio_file(
        self, input_file: str, character_name: str, output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process audio file with character interaction."""
        try:
            # Load character
            if not self.lifecycle.character_manager:
                raise ValueError("Character manager not initialized")

            character = self.lifecycle.character_manager.get_character(character_name)
            if not character:
                raise ValueError(f"Character '{character_name}' not found")

            # Set as active character
            self.lifecycle.set_active_character(character)

            # Process audio file
            if self.processing_pipeline:
                return await self.processing_pipeline.process_audio_file(
                    input_file, character, output_file
                )
            else:
                raise ValueError("Processing pipeline not initialized")

        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            return {
                "transcription": "",
                "response": "",
                "output_file": output_file,
                "success": False,
                "error": str(e),
            }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_monitor.get_metrics()

    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        return self.performance_monitor.get_success_rate()

    def log_performance_summary(self) -> None:
        """Log a summary of current performance metrics."""
        self.performance_monitor.log_performance_summary()

    async def shutdown(self) -> None:
        """Shutdown the engine and clean up resources."""
        try:
            logger.info("Shutting down core real-time engine...")

            # Shutdown processing pipeline
            if self.processing_pipeline:
                await self.processing_pipeline.shutdown()

            # Shutdown audio system
            await self.audio_system.shutdown()

            # Shutdown lifecycle
            await self.lifecycle.shutdown()

            logger.info("Core real-time engine shutdown complete")
        except Exception as e:
            logger.error(f"Error during core engine shutdown: {e}")
