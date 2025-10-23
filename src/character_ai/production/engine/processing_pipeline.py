"""Processing pipeline for real-time audio interaction."""

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ...algorithms.conversational_ai.hybrid_memory import HybridMemorySystem

from ...algorithms.conversational_ai.session_memory import SessionMemory
from ...algorithms.conversational_ai.text_normalizer import TextNormalizer
from ...characters import Character, ChildSafetyFilter
from ...characters.voices import SchemaVoiceService
from ...core.caching import ResponseCache
from ...core.llm.template_prompt_builder import TemplatePromptBuilder
from ...core.protocols import AudioData, AudioResult
from ...core.resource_manager import ResourceManager
from ...services import LLMService, PipelineOrchestrator, STTService, TTSService

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """Manages the audio processing pipeline and service orchestration."""

    def __init__(
        self,
        resource_manager: ResourceManager,
        text_normalizer: TextNormalizer,
        prompt_builder: TemplatePromptBuilder,
        session_memory: SessionMemory,
        voice_manager: SchemaVoiceService,
        safety_filter: ChildSafetyFilter,
        response_cache: ResponseCache,
        hybrid_memory: Optional["HybridMemorySystem"] = None,
    ):
        self.resource_manager = resource_manager
        self.text_normalizer = text_normalizer
        self.prompt_builder = prompt_builder
        self.session_memory = session_memory
        self.voice_manager = voice_manager
        self.safety_filter = safety_filter
        self.response_cache = response_cache
        self.hybrid_memory = hybrid_memory

        # Services
        self.stt_service: Optional[STTService] = None
        self.llm_service: Optional[LLMService] = None
        self.tts_service: Optional[TTSService] = None
        self.pipeline_orchestrator: Optional[PipelineOrchestrator] = None

    async def initialize(self) -> None:
        """Initialize the processing pipeline services."""
        try:
            logger.info("Initializing processing pipeline...")

            # Initialize services
            self.stt_service = STTService(self.resource_manager)
            self.llm_service = LLMService(
                self.resource_manager,
                self.text_normalizer,
                self.prompt_builder,
                self.session_memory,
                self.hybrid_memory,
            )
            self.tts_service = TTSService(
                self.resource_manager, self.text_normalizer, self.voice_manager
            )

            # Create streaming config from runtime config
            streaming_config = None
            if hasattr(self.resource_manager.config, "tts") and hasattr(
                self.resource_manager.config.tts, "streaming"
            ):
                streaming_config = self.resource_manager.config.tts.streaming

            self.pipeline_orchestrator = PipelineOrchestrator(
                self.stt_service,
                self.llm_service,
                self.tts_service,
                self.session_memory,
                self.response_cache,
                self.safety_filter,
                streaming_config,
                self.hybrid_memory,
            )

            logger.info("Processing pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize processing pipeline: {e}")
            raise

    async def preload_models(self) -> Dict[str, bool]:
        """Pre-load all models for real-time interaction."""
        results = {}
        try:
            if self.resource_manager:
                # Preload all models including TTS for performance
                await self.resource_manager.preload_models(["tts", "stt", "llm"])
                results = {"stt": True, "llm": True, "tts": True}
            return results
        except Exception as e:
            logger.error(f"Failed to preload models: {e}")
            return {"stt": False, "llm": False, "tts": False}

    async def process_realtime_audio(
        self, audio_stream: AudioData, character: Character
    ) -> AudioResult:
        """Process audio in real-time for natural interaction."""
        time.time()

        try:
            # Use pipeline orchestrator for optimized processing
            if self.pipeline_orchestrator:
                result = await self.pipeline_orchestrator.process_pipeline(
                    audio_stream, character
                )
            else:
                # Fallback to individual service processing
                result = await self._process_audio_fallback(audio_stream, character)

            # Process turn through hybrid memory system if available
            if self.hybrid_memory and result.text:
                logger.debug(
                    f"Processing turn through hybrid memory system for {character.name}"
                )
                self.hybrid_memory.process_turn(
                    character.name,
                    result.metadata.get("transcribed_text", "")
                    if result.metadata
                    else "",
                    result.text,
                )
            else:
                logger.debug(
                    f"Hybrid memory system not available: hybrid_memory={self.hybrid_memory}, result.text={result.text}"
                )

            return result

        except Exception as e:
            logger.error(f"Error processing real-time audio: {e}")
            raise

    async def _process_audio_fallback(
        self, audio_stream: AudioData, character: Character
    ) -> AudioResult:
        """Fallback audio processing when pipeline orchestrator is not available."""
        try:
            # STT processing
            if not self.stt_service:
                raise ValueError("STT service not available")

            transcription = await self.stt_service.transcribe(audio_stream)
            if not transcription or not transcription.strip():
                return AudioResult(text="", audio_data=None, error="No speech detected")

            # LLM processing
            if not self.llm_service:
                raise ValueError("LLM service not available")

            response = await self.llm_service.generate_response(
                transcription, character
            )
            if not response or not response.strip():
                return AudioResult(
                    text="",
                    audio_data=None,
                    error="No response generated",
                    metadata={"transcribed_text": transcription},
                )

            # TTS processing
            if not self.tts_service:
                raise ValueError("TTS service not available")

            audio_bytes = await self.tts_service.synthesize_blocking(
                response, character
            )

            # Convert bytes to AudioData
            if audio_bytes:
                from ...core.protocols import AudioData

                audio_data = AudioData(
                    data=audio_bytes,
                    sample_rate=22050,  # Default TTS sample rate
                    duration=len(audio_bytes) / 22050,
                    channels=1,
                )
            else:
                audio_data = None

            return AudioResult(
                text=response,
                audio_data=audio_data,
                metadata={"transcribed_text": transcription},
            )

        except Exception as e:
            logger.error(f"Fallback audio processing failed: {e}")
            return AudioResult(text="", audio_data=None, error=str(e))

    async def process_audio_file(
        self, input_file: str, character: Character, output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process audio file with character interaction."""
        try:
            logger.info(f"Processing audio file: {input_file}")

            # Load audio file using centralized utility (DRY principle)
            try:
                from ...core.audio_io.audio_utils import load_audio_file

                audio_stream = load_audio_file(input_file)
            except Exception as e:
                logger.error(f"Failed to load audio file {input_file}: {e}")
                return {
                    "transcription": "",
                    "response": "",
                    "output_file": output_file,
                    "success": False,
                    "error": f"Failed to load audio file: {e}",
                }

            # Process audio
            if audio_stream is None:
                return {
                    "transcription": "",
                    "response": "",
                    "output_file": output_file,
                    "success": False,
                    "error": "Failed to load audio stream",
                }
            result = await self.process_realtime_audio(audio_stream, character)

            # Save output if specified using centralized utility (DRY principle)
            if output_file and result.audio_data:
                from ...core.audio_io.audio_utils import write_wav_file

                write_wav_file(
                    output_file, result.audio_data.data, result.audio_data.sample_rate
                )
                logger.info(f"Audio output saved to: {output_file}")

            return {
                "transcription": result.metadata.get("transcribed_text", "")
                if result.metadata
                else "",
                "response": result.text or "",
                "output_file": output_file,
                "success": result.error is None,
                "error": result.error,
            }

        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            return {
                "transcription": "",
                "response": "",
                "output_file": output_file,
                "success": False,
                "error": str(e),
            }

    async def shutdown(self) -> None:
        """Shutdown the processing pipeline."""
        try:
            logger.info("Shutting down processing pipeline...")

            # Services will be cleaned up by their respective managers
            self.stt_service = None
            self.llm_service = None
            self.tts_service = None
            self.pipeline_orchestrator = None

            logger.info("Processing pipeline shutdown complete")
        except Exception as e:
            logger.error(f"Error during processing pipeline shutdown: {e}")
