"""
Real-time interaction engine for character.ai.

Optimized for real-time toy interaction with sub-500ms latency.
"""

# CRITICAL: Import torch_init FIRST to set environment variables before any torch imports
# isort: off
from ..core import torch_init  # noqa: F401

# isort: on

import time
from typing import Any, Dict, List, Optional

from ..characters import Character
from ..core.protocols import AudioData, AudioResult
from ..hardware.toy_hardware_manager import ToyHardwareManager
from ..observability import get_logger
from .engine.audio_processor import AudioProcessor
from .engine.character_manager import CharacterInteractionController
from .engine.core_engine import CoreRealTimeEngine
from .engine.performance_monitor import PerformanceMonitor

logger = get_logger(__name__)


class RealTimeInteractionEngine:
    """Optimized for real-time toy interaction."""

    def __init__(
        self,
        hardware_manager: ToyHardwareManager,
        # Optional dependency injection for new services
        text_normalizer: Optional[Any] = None,
        prompt_builder: Optional[Any] = None,
        resource_manager: Optional[Any] = None,
        hardware_profile: Optional[str] = None,
    ):
        # Initialize core engine
        self.core_engine = CoreRealTimeEngine(
            hardware_manager=hardware_manager,
            text_normalizer=text_normalizer,
            prompt_builder=prompt_builder,
            resource_manager=resource_manager,
            hardware_profile=hardware_profile,
        )

        # Initialize specialized modules
        self.audio_processor = AudioProcessor()
        self.character_controller = CharacterInteractionController(
            character_manager=self.core_engine.lifecycle.character_manager,
            voice_manager=self.core_engine.lifecycle.voice_manager,
            resource_manager=self.core_engine.lifecycle.resource_manager,
            hardware_config=self.core_engine.lifecycle.hardware_config,
        )
        self.performance_monitor = PerformanceMonitor()

        # Store hardware manager for external access
        self.hardware_manager: Optional[ToyHardwareManager] = hardware_manager
        self.active_character: Optional[Character] = None

    async def initialize(self) -> None:
        """Initialize the real-time interaction engine."""
        await self.core_engine.initialize()

        # Initialize audio processor with services
        if self.core_engine.processing_pipeline is None:
            raise ValueError("Processing pipeline not initialized")

        self.audio_processor = AudioProcessor(
            stt_service=self.core_engine.processing_pipeline.stt_service,
            llm_service=self.core_engine.processing_pipeline.llm_service,
            tts_service=self.core_engine.processing_pipeline.tts_service,
            pipeline_orchestrator=self.core_engine.processing_pipeline.pipeline_orchestrator,
        )

    # Accessor methods for internal components (encapsulation)
    @property
    def character_manager(self) -> CharacterInteractionController:
        """Get the character interaction controller."""
        return self.character_controller

    async def preload_models(self) -> Dict[str, bool]:
        """Pre-load all models for real-time interaction."""
        return await self.core_engine.preload_models()

    async def process_realtime_audio(self, audio_stream: AudioData) -> AudioResult:
        """Process audio in real-time for natural interaction."""
        start_time = time.time()

        if not self.active_character:
            raise ValueError("No active character set")

        try:
            # Use audio processor for optimized processing
            result = await self.audio_processor.process_audio_with_character(
                audio_stream, self.active_character, optimized=True
            )

            # Update performance metrics
            latency = time.time() - start_time
            self.performance_monitor.update_metrics(latency, True)

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
        return await self.core_engine.process_audio_file(
            input_file, character_name, output_file
        )

    # Audio Processing API
    async def transcribe_audio(self, audio: AudioData) -> str:
        """PUBLIC API: Transcribe audio using STT."""
        return await self.audio_processor.transcribe_audio(audio)

    async def transcribe_audio_file(self, input_file: str) -> dict:
        """PUBLIC API: Transcribe audio file using STT."""
        try:
            from ..core.audio_io.audio_utils import load_audio_file

            # Load audio file using centralized utility
            audio_data = load_audio_file(input_file)
            if audio_data is None:
                return {
                    "success": False,
                    "error": "Failed to load audio file",
                    "file": input_file,
                }

            # Transcribe
            transcription = await self.transcribe_audio(audio_data)

            return {"success": True, "transcription": transcription, "file": input_file}
        except Exception as e:
            logger.error(f"Failed to transcribe audio file {input_file}: {e}")
            return {"success": False, "error": str(e), "file": input_file}

    async def generate_response(self, text: str, character: Character) -> dict:
        """PUBLIC API: Generate character response using LLM."""
        try:
            response = await self.audio_processor.generate_response(text, character)
            return {"success": True, "response": response}
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return {"success": False, "error": str(e)}

    async def synthesize_voice(
        self, text: str, character: Character, output_file: Optional[str] = None
    ) -> dict:
        """PUBLIC API: Synthesize speech using TTS."""
        try:
            audio_data = await self.audio_processor.synthesize_voice(text, character)

            # If output file is specified, save the audio
            if output_file:
                from ..core.audio_io.audio_utils import write_wav_file

                if hasattr(audio_data, "data") and hasattr(audio_data, "sample_rate"):
                    write_wav_file(
                        output_file,
                        audio_data.data,
                        audio_data.sample_rate,
                        subtype="PCM_16",
                        channels=1,
                    )
                else:
                    # Assume it's bytes and use default sample rate
                    write_wav_file(
                        output_file, audio_data, 22050, subtype="PCM_16", channels=1
                    )
                logger.info(f"TTS audio saved to: {output_file}")

            return {
                "success": True,
                "audio_data": audio_data,
                "output_file": output_file,
            }
        except Exception as e:
            logger.error(f"Failed to synthesize voice: {e}")
            return {"success": False, "error": str(e)}

    async def process_audio_with_character(
        self, audio: AudioData, character: Character, optimized: bool = True
    ) -> AudioResult:
        """PUBLIC API: Process audio with specific character."""
        return await self.audio_processor.process_audio_with_character(
            audio, character, optimized
        )

    async def process_speech_segment(
        self, audio_array: Any, character: Character
    ) -> AudioResult:
        """Process a detected speech segment."""
        return await self.audio_processor.process_speech_segment(audio_array, character)

    # Character Management API
    async def set_active_character(self, character_name: str) -> bool:
        """Set the active character for interactions."""
        success = await self.character_manager.set_active_character(character_name)
        if success:
            character = self.character_manager.get_active_character()
            if character is not None:
                self.active_character = character
        return success

    async def get_character_info(self) -> Dict[str, Any]:
        """Get information about the active character."""
        if self.active_character is None:
            return {"error": "No active character"}
        result = self.character_manager.get_character_info(self.active_character.name)
        return result if result is not None else {"error": "Character not found"}

    async def inject_character_voice(
        self, character_name: str, voice_file_path: str
    ) -> bool:
        """Inject a voice for a character (used during toy manufacturing/setup)."""
        return await self.character_manager.inject_character_voice(
            character_name, voice_file_path
        )

    async def list_character_voices(self) -> List[str]:
        """List all characters that have injected voices."""
        return await self.character_manager.list_character_voices()

    async def start_realtime_session(
        self,
        character: Character,
        duration: int,
        device_pattern: Optional[str] = None,
        vad_config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Start real-time voice interaction session."""
        return await self.character_manager.start_realtime_session(
            character, duration, device_pattern, vad_config, self.audio_processor
        )

    # Performance and Optimization API
    async def optimize_for_toy(self) -> None:
        """Optimize entire system for toy deployment."""
        # Performance optimization for toy hardware
        pass

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the real-time engine."""
        metrics = self.performance_monitor.get_metrics()
        # Add success rate to the metrics
        metrics["success_rate"] = self.performance_monitor.get_success_rate()
        return metrics

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the real-time engine."""
        character_info = await self.get_character_info()
        character_healthy = "error" not in character_info
        # Get health status
        return {
            "status": "healthy" if character_healthy else "error",
            "character_info": character_info,
            "timestamp": time.time(),
        }

    async def shutdown(self) -> None:
        """Shutdown the engine and clean up resources."""
        await self.core_engine.shutdown()

        # Clear references after shutdown
        self.hardware_manager = None
        self.character_controller = None  # type: ignore

    # Performance metrics properties
    @property
    def total_interactions(self) -> int:
        """Total interactions count."""
        return int(self.performance_monitor.performance_metrics["total_interactions"])

    @property
    def successful_interactions(self) -> int:
        """Successful interactions count."""
        return int(
            self.performance_monitor.performance_metrics["successful_interactions"]
        )

    @property
    def average_latency(self) -> float:
        """Average latency."""
        return self.performance_monitor.performance_metrics["average_latency"]

    @property
    def target_latency(self) -> float:
        """Target latency."""
        return 5.0  # Default target latency in seconds
