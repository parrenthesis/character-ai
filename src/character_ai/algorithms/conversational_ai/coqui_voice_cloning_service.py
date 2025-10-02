import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ...core.config import Config
from ...core.exceptions import ModelError
from ...core.protocols import AudioResult

logger = logging.getLogger(__name__)


class CoquiVoiceCloningService:
    """
    Native voice cloning service using Coqui TTS.

    Provides native Coqui TTS voice cloning capabilities.
    Provides high-quality voice cloning without container overhead.
    """

    def __init__(self, config: Config):
        self.config = config
        self.tts_processor: Optional[Any] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the Coqui TTS voice cloning service."""
        if self._initialized:
            logger.info("CoquiVoiceCloningService already initialized.")
            return

        logger.info(f"Initializing Coqui voice cloning service on {self.device}")
        try:
            from TTS.api import TTS

            # Initialize TTS with a model that supports voice cloning
            model_name = getattr(
                self.config.tts, "model_name", "tts_models/en/ljspeech/tacotron2-DDC"
            )
            self.tts_processor = TTS(model_name=model_name, progress_bar=False)

            # Move to device if CUDA is available
            if self.device == "cuda":
                self.tts_processor.to(self.device)

            self._initialized = True
            logger.info("Coqui voice cloning service initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Coqui voice cloning service: {e}")
            raise ModelError(f"Coqui voice cloning service initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown the voice cloning service and release resources."""
        if not self._initialized:
            logger.info(
                "CoquiVoiceCloningService not initialized, nothing to shut down."
            )
            return

        logger.info("Shutting down Coqui voice cloning service.")
        self.tts_processor = None
        self._initialized = False

        # Attempt to clear CUDA cache if on GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("Coqui voice cloning service shutdown complete.")

    async def clone_voice_from_samples(
        self,
        character_name: str,
        voice_samples: List[str],
        text: str,
        output_path: Optional[str] = None,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Clone a character's voice from audio samples and synthesize text.

        Args:
            character_name: Name of the character
            voice_samples: List of paths to voice sample audio files
            text: Text to synthesize with the cloned voice
            output_path: Optional output path for the synthesized audio
            language: Language code for synthesis

        Returns:
            Dictionary with cloning results
        """
        if not self._initialized:
            raise RuntimeError(
                "CoquiVoiceCloningService not initialized. Call initialize() first."
            )

        start_time = time.time()
        try:
            logger.info(
                f"Cloning voice for character '{character_name}' with {len(voice_samples)} samples"
            )

            # Use the first voice sample as the reference
            reference_audio = voice_samples[0]
            if not os.path.exists(reference_audio):
                raise FileNotFoundError(f"Voice sample not found: {reference_audio}")

            # Clone voice and synthesize text
            if self.tts_processor is None:
                raise RuntimeError("TTS processor not initialized")
            audio_array = self.tts_processor.tts(
                text=text, speaker_wav=reference_audio, language=language
            )

            # Generate output path if not provided
            if output_path is None:
                output_dir = Path(self.config.paths.voices_dir) / character_name
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(output_dir / f"{character_name}_cloned_voice.wav")

            # Save the synthesized audio
            import soundfile as sf

            # Get sample rate from config
            sample_rate = getattr(self.config.tts, "voice_cloning_sample_rate", 22050)
            sf.write(output_path, audio_array, sample_rate)

            end_time = time.time()
            processing_time = end_time - start_time

            logger.info(
                f"Voice cloning completed for '{character_name}' in {processing_time:.2f}s"
            )

            return {
                "success": True,
                "character_name": character_name,
                "output_path": output_path,
                "processing_time": processing_time,
                "voice_samples_used": len(voice_samples),
                "reference_audio": reference_audio,
                "synthesized_text": text,
                "language": language,
            }

        except Exception as e:
            logger.error(f"Error cloning voice for character '{character_name}': {e}")
            return {
                "success": False,
                "character_name": character_name,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    async def synthesize_with_character_voice(
        self,
        character_name: str,
        text: str,
        voice_path: Optional[str] = None,
        language: str = "en",
    ) -> AudioResult:
        """
        Synthesize text using a character's cloned voice.

        Args:
            character_name: Name of the character
            text: Text to synthesize
            voice_path: Path to the character's voice file
            language: Language code for synthesis

        Returns:
            AudioResult with synthesized audio
        """
        if not self._initialized:
            raise RuntimeError(
                "CoquiVoiceCloningService not initialized. Call initialize() first."
            )

        start_time = time.time()
        try:
            logger.info(
                f"Synthesizing text for character '{character_name}': '{text[:50]}...'"
            )

            # Find character voice if not provided
            if voice_path is None:
                voice_path = await self._find_character_voice(character_name)

            if self.tts_processor is None:
                raise RuntimeError("TTS processor not initialized")

            if voice_path and os.path.exists(voice_path):
                # Use cloned voice
                audio_array = self.tts_processor.tts(
                    text=text, speaker_wav=voice_path, language=language
                )
            else:
                # Use default voice
                audio_array = self.tts_processor.tts(text=text, language=language)
                logger.warning(
                    f"No voice found for character '{character_name}', using default voice"
                )

            # Convert to bytes
            audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()

            end_time = time.time()
            processing_time = end_time - start_time
            sample_rate = getattr(self.config.tts, "voice_cloning_sample_rate", 22050)

            logger.info(
                f"Voice synthesis completed for '{character_name}' in {processing_time:.2f}s"
            )

            return AudioResult(
                text=text,
                audio_data=audio_bytes,
                metadata={
                    "character_name": character_name,
                    "voice_path": voice_path,
                    "language": language,
                    "processing_time": processing_time,
                    "sample_rate": sample_rate,
                    "model": "Coqui TTS",
                },
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(
                f"Error synthesizing voice for character '{character_name}': {e}"
            )
            return AudioResult(
                text=text,
                error=f"Voice synthesis failed: {e}",
                metadata={
                    "character_name": character_name,
                    "error": str(e),
                    "processing_time": time.time() - start_time,
                },
                processing_time=time.time() - start_time,
            )

    async def _find_character_voice(self, character_name: str) -> Optional[str]:
        """Find the voice file for a character."""
        try:
            voices_dir = Path(self.config.paths.voices_dir) / character_name
            if voices_dir.exists():
                # Look for common voice file patterns
                for pattern in ["*_cloned_voice.wav", "*_voice.wav", "voice.wav"]:
                    voice_files = list(voices_dir.glob(pattern))
                    if voice_files:
                        return str(voice_files[0])
            return None
        except Exception as e:
            logger.warning(f"Error finding voice for character '{character_name}': {e}")
            return None

    async def get_character_voice_info(self, character_name: str) -> Dict[str, Any]:
        """Get information about a character's voice."""
        try:
            voice_path = await self._find_character_voice(character_name)
            if voice_path and os.path.exists(voice_path):
                # Get file info
                stat = os.stat(voice_path)
                return {
                    "character_name": character_name,
                    "voice_path": voice_path,
                    "file_size": stat.st_size,
                    "modified_time": stat.st_mtime,
                    "exists": True,
                }
            else:
                return {
                    "character_name": character_name,
                    "voice_path": None,
                    "exists": False,
                }
        except Exception as e:
            logger.error(
                f"Error getting voice info for character '{character_name}': {e}"
            )
            return {
                "character_name": character_name,
                "error": str(e),
                "exists": False,
            }

    async def list_characters_with_voices(self) -> List[Dict[str, Any]]:
        """List all characters that have voice files."""
        try:
            voices_dir = Path(self.config.paths.voices_dir)
            characters = []

            if voices_dir.exists():
                for character_dir in voices_dir.iterdir():
                    if character_dir.is_dir():
                        voice_info = await self.get_character_voice_info(
                            character_dir.name
                        )
                        characters.append(voice_info)

            return characters
        except Exception as e:
            logger.error(f"Error listing characters with voices: {e}")
            return []

    async def remove_character_voice(self, character_name: str) -> Dict[str, Any]:
        """Remove a character's voice file."""
        try:
            voice_path = await self._find_character_voice(character_name)
            if voice_path and os.path.exists(voice_path):
                os.remove(voice_path)
                logger.info(
                    f"Removed voice file for character '{character_name}': {voice_path}"
                )
                return {
                    "success": True,
                    "character_name": character_name,
                    "removed_path": voice_path,
                }
            else:
                return {
                    "success": False,
                    "character_name": character_name,
                    "error": "No voice file found",
                }
        except Exception as e:
            logger.error(f"Error removing voice for character '{character_name}': {e}")
            return {
                "success": False,
                "character_name": character_name,
                "error": str(e),
            }
