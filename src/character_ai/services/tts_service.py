"""Text-to-Speech service with voice cloning."""

import logging
import os
import re
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

from ..characters import Character
from ..core.exceptions import handle_audio_error
from .base_service import BaseService

if TYPE_CHECKING:
    from ..algorithms.conversational_ai.text_normalizer import TextNormalizer
    from ..core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class TTSService(BaseService):
    """Text-to-speech service with voice cloning support.

    Handles both blocking and streaming TTS synthesis with character voice management.
    """

    def __init__(
        self,
        resource_manager: "ResourceManager",
        text_normalizer: "TextNormalizer",
        voice_manager: Any,
    ):
        super().__init__(resource_manager)
        self.text_normalizer = text_normalizer
        self.voice_manager = voice_manager

    def _get_processor(self, model_type: str) -> Any:
        """Get TTS processor from resource manager."""
        return self.resource_manager.get_tts_processor()

    async def _get_voice_path_for_character(
        self, character: Character
    ) -> Optional[str]:
        """Get voice path for character with metadata fallback.

        Args:
            character: Character to get voice path for

        Returns:
            Voice file path or None if not found
        """
        character_name = (
            character.name.lower()
            if hasattr(character, "name")
            else str(character).lower()
        )
        voice_path = None

        # First check if character already has a voice path in metadata
        if hasattr(character, "metadata") and character.metadata:
            voice_path = character.metadata.get("voice_path")
            if voice_path:
                logger.info(f"Using voice path from character metadata: {voice_path}")

        # If no voice path in metadata, try voice manager
        if not voice_path and self.voice_manager:
            # Get franchise from character metadata or use character name as fallback
            franchise = (
                getattr(character, "franchise", None)
                or (
                    character.metadata.get("franchise")
                    if hasattr(character, "metadata") and character.metadata
                    else None
                )
                or character_name
            )
            voice_info = await self.voice_manager.get_character_voice_path(
                character_name, franchise
            )
            voice_path = voice_info.get("voice_file_path") if voice_info else None

        return voice_path

    def _get_tts_config(self, character: Character) -> tuple[float, Optional[str]]:
        """Get TTS configuration from character metadata.

        Args:
            character: Character to get TTS config for

        Returns:
            Tuple of (speed, model_name)
        """
        speed = 1.0  # Default speed
        tts_model_name = None

        if hasattr(character.metadata, "get") and character.metadata:
            tts_config = character.metadata.get("tts_config", {})
            speed = (
                tts_config.get("speed", 1.0) if isinstance(tts_config, dict) else 1.0
            )
            tts_model_name = (
                tts_config.get("model") if isinstance(tts_config, dict) else None
            )

        return speed, tts_model_name

    @handle_audio_error
    async def synthesize_blocking(self, text: str, character: Character) -> bytes:
        """Blocking TTS synthesis.

        Args:
            text: Text to synthesize
            character: Character for voice synthesis

        Returns:
            Audio data as bytes (WAV format)
        """
        # Get TTS config from character metadata
        speed, tts_model_name = self._get_tts_config(character)
        logger.info(f"Character metadata: {character.metadata}")
        logger.info(
            f"TTS config from metadata: {character.metadata.get('tts_config', {}) if hasattr(character.metadata, 'get') and character.metadata else {}}"
        )
        logger.info(f"Using TTS speed: {speed}")

        # Load character-specific TTS model if specified
        if tts_model_name:
            logger.info(f"Loading character-specific TTS model: {tts_model_name}")
            await self.resource_manager.preload_models_with_config(
                {"tts": {"model": tts_model_name}}
            )

        # Use common processor initialization pattern
        tts_processor = await self.get_or_create_processor("tts")

        # Get voice configuration
        voice_path = await self._get_voice_path_for_character(character)

        # Use text normalizer to prepare for TTS
        tts_text = self.text_normalizer.prepare_for_tts(text)

        # Use processor
        logger.info(
            f"TTS synthesis: text='{tts_text}', voice_path='{voice_path}', speed={speed}"
        )
        if voice_path is None:
            logger.warning(
                "TTS synthesis called with voice_path=None - this will cause synthesis to fail!"
            )
        result = await tts_processor.synthesize_speech(
            text=tts_text, voice_path=voice_path, language="en", speed=speed
        )
        logger.info(
            f"TTS result: audio_data={result.audio_data is not None}, error={result.error}"
        )
        if result.audio_data and result.audio_data.data is not None:
            logger.info(
                f"TTS audio data size: {len(result.audio_data.data)} bytes, sample_rate: {result.audio_data.sample_rate} Hz"
            )
        else:
            logger.warning("TTS result has no audio_data!")

        # Mark model as used
        self.resource_manager.mark_model_used("tts")
        return result.audio_data if result.audio_data else b""

    async def synthesize_streaming(
        self, text: str, character: Character
    ) -> AsyncGenerator[bytes, None]:
        """Streaming TTS synthesis.

        Synthesizes sentence-by-sentence and yields audio chunks.

        Args:
            text: Text to synthesize
            character: Character for voice synthesis

        Yields:
            Audio chunks as bytes (WAV format)
        """
        try:
            # Get TTS configuration (same as blocking method)
            speed, tts_model_name = self._get_tts_config(character)

            # Load character-specific TTS model if specified
            if tts_model_name:
                logger.info(
                    f"Loading character-specific TTS model for streaming: {tts_model_name}"
                )
                await self.resource_manager.preload_models_with_config(
                    {"tts": {"model": tts_model_name}}
                )

            # Get TTS processor
            tts_processor = self.resource_manager.get_tts_processor()
            if tts_processor is None:
                await self.resource_manager.preload_models(["tts"])
                tts_processor = self.resource_manager.get_tts_processor()

            if tts_processor is None:
                raise RuntimeError("Failed to initialize TTS processor")

            # Get voice configuration
            voice_path = await self._get_voice_path_for_character(character)

            # Debug logging for voice path
            if voice_path is None:
                logger.warning(
                    "Streaming TTS called with voice_path=None - this will cause synthesis to fail!"
                )
            else:
                logger.info(f"Streaming TTS using voice_path: {voice_path}")

            # IMPORTANT: Split into sentences BEFORE text normalization
            # The text normalizer replaces periods with commas, which breaks sentence splitting
            quiet_mode = os.getenv("CAI_QUIET_MODE") == "1"
            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            if not sentences:
                sentences = [text]

            if not quiet_mode:
                print(
                    f"DEBUG _synthesize_character_voice_stream: Split into {len(sentences)} sentences"
                )
            logger.info(f"Starting streaming TTS synthesis: {len(sentences)} sentences")

            chunk_count = 0
            for idx, sentence in enumerate(sentences):
                # Normalize each sentence individually
                tts_text = self.text_normalizer.prepare_for_tts(sentence)
                if not quiet_mode:
                    print(
                        f"DEBUG: Sentence {idx+1}/{len(sentences)}: '{tts_text[:50]}...'"
                    )

                # Synthesize this sentence
                result = await tts_processor.synthesize_speech(
                    text=tts_text, voice_path=voice_path, language="en", speed=speed
                )

                if (
                    result.audio_data
                    and result.audio_data.data is not None
                    and len(result.audio_data.data) > 0
                ):
                    chunk_count += 1
                    if not quiet_mode:
                        print(
                            f"DEBUG _synthesize_character_voice_stream: Yielding chunk {chunk_count}"
                        )
                    yield result.audio_data
                else:
                    logger.warning(f"Sentence {idx+1} synthesis returned no audio data")

            if not quiet_mode:
                print(
                    f"DEBUG _synthesize_character_voice_stream: Completed, yielded {chunk_count} chunks"
                )

            # Mark model as used
            self.resource_manager.mark_model_used("tts")

        except Exception as e:
            logger.error(f"Voice synthesis streaming failed: {e}")
            raise

    async def warmup(self) -> None:
        """Pre-warm TTS model for faster synthesis."""
        if not self.resource_manager.get_tts_processor():
            await self.resource_manager.preload_models(["tts"])
