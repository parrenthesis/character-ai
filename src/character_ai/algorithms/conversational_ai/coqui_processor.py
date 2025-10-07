import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ...core.config import Config
from ...core.exceptions import ModelError
from ...core.protocols import AudioData, AudioResult, BaseAudioProcessor, ModelInfo

logger = logging.getLogger(__name__)


class CoquiProcessor(BaseAudioProcessor):
    """
    Coqui TTS processor for text-to-speech synthesis and voice cloning.

    Implements Coqui TTS for high-quality text-to-speech conversion with
    native voice cloning capabilities.
    """

    def __init__(
        self, config: Config, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"
    ):
        self.config = config
        self.model_name = model_name
        self.tts: Any = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the Coqui TTS model."""
        if self._initialized:
            logger.info("CoquiProcessor already initialized.")
            return

        logger.info(f"Initializing Coqui TTS model: {self.model_name} on {self.device}")
        try:
            # Suppress GPT2InferenceModel warning
            import warnings

            warnings.filterwarnings("ignore", message=".*GPT2InferenceModel.*")

            import torch.serialization
            from TTS.api import TTS

            # Configure PyTorch to allow XTTS model loading (PyTorch 2.6+ security)
            if "xtts" in self.model_name.lower():
                # For XTTS models, we need to disable weights_only due to complex TTS class dependencies
                # This is safe since we trust the Coqui TTS source
                import torch

                original_load = torch.load

                def patched_load(*args: Any, **kwargs: Any) -> Any:
                    kwargs["weights_only"] = False
                    return original_load(*args, **kwargs)

                torch.load = patched_load
                logger.info(
                    "Patched torch.load to disable weights_only for XTTS model loading"
                )

            # Initialize TTS with the specified model
            self.tts = TTS(model_name=self.model_name, progress_bar=False)

            # Move to device if CUDA is available
            if self.device == "cuda":
                self.tts.to(self.device)

            self._initialized = True
            logger.info("CoquiProcessor initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize CoquiProcessor: {e}")
            raise ModelError(f"Coqui TTS model initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown the Coqui TTS model and release resources."""
        if not self._initialized:
            logger.info("CoquiProcessor not initialized, nothing to shut down.")
            return

        logger.info("Shutting down CoquiProcessor.")
        self.tts = None
        self._initialized = False

        # Attempt to clear CUDA cache if on GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("CoquiProcessor shutdown complete.")

    async def process_audio(
        self, audio: AudioData, language: Optional[str] = None
    ) -> AudioResult:
        """Process text to generate speech audio."""
        if not self._initialized:
            raise RuntimeError(
                "CoquiProcessor not initialized. Call initialize() first."
            )

        time.time()
        try:
            # For TTS, we need text input, not audio input
            # This method signature is from the base class but TTS works differently
            # We'll handle this in the voice cloning service instead
            logger.warning(
                "CoquiProcessor.process_audio called with audio input - this is for TTS, not STT"
            )

            return await self._create_error_result(
                "CoquiProcessor is for TTS, not audio processing"
            )

        except Exception as e:
            logger.error(f"Error processing audio with Coqui TTS: {e}")
            return await self._create_error_result(f"Coqui TTS processing failed: {e}")

    async def synthesize_speech(
        self, text: str, voice_path: Optional[str] = None, language: str = "en"
    ) -> AudioResult:
        """Synthesize speech from text using Coqui TTS."""
        if not self._initialized:
            raise RuntimeError(
                "CoquiProcessor not initialized. Call initialize() first."
            )

        start_time = time.time()
        try:
            logger.info(f"Coqui TTS synthesizing: '{text[:50]}...'")

            # Determine multilingual capability safely
            is_multilingual = False
            try:
                # Some models expose `is_multi_lingual` or `is_multilingual`
                if hasattr(self.tts, "is_multi_lingual"):
                    is_multilingual = bool(getattr(self.tts, "is_multi_lingual"))
                elif hasattr(self.tts, "is_multilingual"):
                    is_multilingual = bool(getattr(self.tts, "is_multilingual"))
            except Exception:
                is_multilingual = False

            try:
                bool(getattr(self.tts, "is_multi_speaker", False))
            except Exception:
                pass

            # Build kwargs conditionally to avoid passing unsupported parameters
            tts_kwargs: Dict[str, Any] = {"text": text}

            # XTTS v2 models require speaker_wav parameter for voice cloning
            if voice_path:
                tts_kwargs["speaker_wav"] = voice_path

            # XTTS v2 models always require a language parameter
            if is_multilingual:
                tts_kwargs["language"] = (
                    language or "en"
                )  # Default to English if not specified

            # Synthesize speech
            audio_array = self.tts.tts(**tts_kwargs)

            # Coqui may return a list of chunks; normalize to a single numpy array
            if isinstance(audio_array, list):
                try:
                    import numpy as _np

                    audio_array = _np.concatenate(
                        [
                            _np.asarray(chunk, dtype=_np.float32).reshape(-1)
                            for chunk in audio_array
                        ]
                    )
                except Exception:
                    # Fallback: take first chunk
                    audio_array = _np.asarray(
                        audio_array[0], dtype=_np.float32
                    ).reshape(-1)

            # Convert to proper WAV format using soundfile
            import io

            import soundfile as sf

            # Convert to float32 for soundfile
            audio_float = audio_array.astype(np.float32)

            # Create WAV data in memory
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_float, 22050, format="WAV", subtype="PCM_16")
            wav_bytes = wav_buffer.getvalue()
            wav_buffer.close()

            end_time = time.time()
            processing_time = end_time - start_time

            logger.info(f"Coqui TTS synthesis completed in {processing_time:.2f}s")

            from ...core.protocols import AudioData

            return AudioResult(
                text=text,
                audio_data=AudioData(
                    data=wav_bytes,
                    sample_rate=22050,  # Coqui TTS default
                    channels=1,  # Mono
                ),
                metadata={
                    "model": "Coqui TTS",
                    "language": language,
                    "voice_path": voice_path,
                    "processing_time": processing_time,
                    "sample_rate": 22050,  # Coqui TTS default
                },
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Error synthesizing speech with Coqui TTS: {e}")
            return await self._create_error_result(f"Coqui TTS synthesis failed: {e}")

    async def clone_voice(
        self, reference_audio_path: str, text: str, output_path: str
    ) -> Dict[str, Any]:
        """Clone a voice from reference audio and synthesize text."""
        if not self._initialized:
            raise RuntimeError(
                "CoquiProcessor not initialized. Call initialize() first."
            )

        start_time = time.time()
        try:
            logger.info(
                f"Cloning voice from {reference_audio_path} for text: '{text[:50]}...'"
            )

            # Clone voice and synthesize
            audio_array = self.tts.tts(text=text, speaker_wav=reference_audio_path)

            # Save to output path
            import soundfile as sf

            sf.write(output_path, audio_array, 22050)

            end_time = time.time()
            processing_time = end_time - start_time

            logger.info(
                f"Voice cloning completed in {processing_time:.2f}s, saved to {output_path}"
            )

            return {
                "success": True,
                "output_path": output_path,
                "processing_time": processing_time,
                "reference_audio": reference_audio_path,
                "synthesized_text": text,
            }

        except Exception as e:
            logger.error(f"Error cloning voice with Coqui TTS: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    async def _create_error_result(self, error_message: str) -> AudioResult:
        """Helper to create an error AudioResult."""
        return AudioResult(
            text="",
            error=error_message,
            metadata={"component": "CoquiProcessor", "error": error_message},
            processing_time=0.0,
        )

    async def get_model_info(self) -> ModelInfo:
        """Get information about the loaded model."""
        if not self._initialized:
            raise RuntimeError("CoquiProcessor not initialized.")
        return ModelInfo(
            name="Coqui TTS",
            type="tts",
            size=f"Coqui TTS model: {self.model_name}",
            memory_usage="Unknown",
            precision="fp32",
            quantization="none",
            loaded_at=time.time(),
        )

    async def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        # Coqui TTS supports multiple languages
        return [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "pl",
            "tr",
            "ru",
            "nl",
            "cs",
            "ar",
            "zh",
            "ja",
            "hi",
        ]

    async def get_model_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities."""
        return {
            "supported_languages": await self.get_supported_languages(),
            "voice_cloning": True,
            "multilingual": True,
            "sample_rate": 22050,
            "channels": 1,
            "memory_usage_gb": await self._estimate_memory_usage(),
            "processing_speed": "fast",
            "quality": "high",
        }

    async def get_embeddings(self, audio: AudioData) -> Any:
        """Extract embeddings from audio data for voice cloning."""
        try:
            # Coqui TTS doesn't typically extract embeddings in the same way as STT models
            # This would be more relevant for voice similarity analysis
            return {
                "embeddings": [],
                "metadata": {
                    "component": "CoquiProcessor",
                    "note": "Voice embeddings not implemented",
                },
                "processing_time": 0.0,
            }
        except Exception as e:
            logger.error(f"Error extracting embeddings: {e}")
            return {
                "embeddings": [],
                "metadata": {"component": "CoquiProcessor", "error": str(e)},
                "processing_time": 0.0,
                "error": str(e),
            }

    async def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of the model in GB."""
        if self.tts:
            # Estimate memory usage based on model parameters
            try:
                # Coqui TTS models vary in size, estimate based on typical usage
                return 2.0  # Typical Coqui TTS model size
            except Exception:
                return 2.0
        return 0.0
