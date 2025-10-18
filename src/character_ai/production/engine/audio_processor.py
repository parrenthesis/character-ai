"""
Audio processing module for real-time character interaction.

Handles transcription, synthesis, and audio processing with character personality.
"""

from typing import Any, Optional

import numpy as np

from ...characters import Character
from ...core.audio_io.audio_utils import AudioNormalizer
from ...core.protocols import AudioData, AudioResult
from ...observability import get_logger
from ...services import LLMService, PipelineOrchestrator, STTService, TTSService

logger = get_logger(__name__)


class AudioProcessor:
    """Handles audio processing for character interactions."""

    def __init__(
        self,
        stt_service: Optional[STTService] = None,
        llm_service: Optional[LLMService] = None,
        tts_service: Optional[TTSService] = None,
        pipeline_orchestrator: Optional[PipelineOrchestrator] = None,
    ):
        self.stt_service = stt_service
        self.llm_service = llm_service
        self.tts_service = tts_service
        self.pipeline_orchestrator = pipeline_orchestrator

    async def transcribe_audio(self, audio: AudioData) -> str:
        """PUBLIC API: Transcribe audio using STT.

        Args:
            audio: Audio data to transcribe

        Returns:
            Transcribed text string
        """
        if not self.stt_service:
            raise ValueError("STT service not available")
        return await self.stt_service.transcribe(audio)

    async def generate_response(self, text: str, character: Character) -> str:
        """PUBLIC API: Generate character response using LLM.

        Args:
            text: User input text
            character: Character to generate response for

        Returns:
            Generated response text
        """
        if not self.llm_service:
            raise ValueError("LLM service not available")
        return await self.llm_service.generate_response(text, character)

    async def synthesize_voice(self, text: str, character: Character) -> bytes:
        """PUBLIC API: Synthesize speech using TTS.

        Args:
            text: Text to synthesize
            character: Character for voice synthesis

        Returns:
            Audio data as bytes (WAV format)
        """
        if not self.tts_service:
            raise ValueError("TTS service not available")
        return await self.tts_service.synthesize_blocking(text, character)

    async def process_audio_with_character(
        self, audio: AudioData, character: Character, optimized: bool = True
    ) -> AudioResult:
        """PUBLIC API: Process audio with specific character.

        Args:
            audio: Audio data to process
            character: Character to use for processing
            optimized: Use optimized pipeline (caching, parallel warmup)

        Returns:
            AudioResult with transcription, response, and audio
        """
        if not self.pipeline_orchestrator:
            raise ValueError("Pipeline orchestrator not available")
        return await self.pipeline_orchestrator.process_pipeline(
            audio, character, optimized=optimized
        )

    async def process_with_character_personality(
        self, audio: AudioData, character: Character
    ) -> AudioResult:
        """Process audio with character personality (non-optimized path)."""
        if not self.pipeline_orchestrator:
            raise ValueError("Pipeline orchestrator not available")
        # Delegate to pipeline orchestrator (non-optimized mode)
        return await self.pipeline_orchestrator.process_pipeline(
            audio, character, optimized=False
        )

    async def process_with_character_personality_optimized(
        self, audio: AudioData, character: Character
    ) -> AudioResult:
        """Optimized pipeline with parallel processing and caching."""
        if not self.pipeline_orchestrator:
            raise ValueError("Pipeline orchestrator not available")
        # Delegate to pipeline orchestrator (optimized mode with caching & parallel warmup)
        return await self.pipeline_orchestrator.process_pipeline(
            audio, character, optimized=True
        )

    async def process_speech_segment(
        self, audio_array: Any, character: Character
    ) -> AudioResult:
        """Process a detected speech segment."""

        logger.info(f"Processing speech segment for {character.name}")

        # Convert audio array to numpy if needed
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)

        # Ensure audio is in the right format
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()

        # Check audio quality
        audio_level = np.max(np.abs(audio_array))
        if audio_level < 1e-6:
            logger.warning("Audio level too low, skipping processing")
            return AudioResult(text="Audio too quiet to process", audio_data=None)

        # Create AudioData object for processing
        audio_data = AudioData(
            data=audio_array,
            sample_rate=44100,  # Default sample rate for real-time processing
            duration=len(audio_array) / 44100,
            channels=1,
        )

        # Process through the character personality pipeline
        try:
            result = await self.process_with_character_personality(
                audio_data, character
            )

            # Log processing statistics
            if result.metadata:
                processing_time = result.metadata.get("processing_time", 0.0)
                logger.info(f"Processed speech in {processing_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Error processing speech segment: {e}")
            return AudioResult(
                text=f"Error processing speech: {str(e)}", audio_data=None
            )

    def validate_audio_quality(self, audio_array: np.ndarray) -> bool:
        """Validate audio quality before processing.

        Args:
            audio_array: Audio data to validate

        Returns:
            True if audio quality is acceptable, False otherwise
        """
        if audio_array.size == 0:
            logger.warning("Empty audio array")
            return False

        # Check audio level
        audio_level = np.max(np.abs(audio_array))
        if audio_level < 1e-6:
            logger.warning("Audio level too low")
            return False

        # Check for clipping
        if audio_level > 0.95:
            logger.warning("Audio may be clipped")
            return False

        return True

    def normalize_audio(self, audio_array: np.ndarray) -> np.ndarray:
        """Normalize audio array for processing using centralized utility."""
        return AudioNormalizer.normalize_if_needed(audio_array)
