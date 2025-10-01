"""
Whisper processor for speech recognition.

Implements OpenAI Whisper for high-quality speech-to-text conversion
optimized for edge devices.
"""

import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from ...core.config import Config
from ...core.exceptions import AudioProcessingError, ModelError
from ...core.protocols import AudioData, AudioResult, BaseAudioProcessor, ModelInfo

# Expose patchable placeholders for heavy deps; real modules bound lazily
whisper = SimpleNamespace(load_model=None, log_mel_spectrogram=None)

logger = logging.getLogger(__name__)


class WhisperProcessor(BaseAudioProcessor):
    """Whisper-based speech recognition processor."""

    def __init__(self, config: Config, model_name: str = "base"):
        super().__init__(f"whisper_{model_name}")
        self.config: Config = config
        self.model_name = model_name
        self.model = None
        self.device: Optional[Any] = None  # Will be set during initialization

    async def initialize(self) -> None:
        """Initialize Whisper model."""
        try:
            # Lazy import heavy dependencies
            global whisper
            import torch

            # Bind real whisper module if placeholder or non-callable is present
            try:
                _load = getattr(whisper, "load_model", None)
                if not callable(_load):
                    import whisper as _whisper

                    whisper = _whisper
            except Exception:
                import whisper as _whisper

                whisper = _whisper

            logger.info(f"Loading Whisper model: {self.model_name}")

            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load model with optimization
            self.model = whisper.load_model(
                self.config.models.whisper_model or self.model_name,
                device=self.device,
                download_root=str(Path(self.config.models_dir) / "whisper"),
            )

            # Create model info
            self.model_info = ModelInfo(
                name="whisper",
                type="stt",
                size=f"{self.model_name} model",
                memory_usage=f"{self._estimate_memory_usage():.2f}GB",
                precision="fp16",
                quantization="none",
                loaded_at=time.time(),
                status="loaded",
            )

            self._initialized = True
            logger.info(f"Whisper model {self.model_name} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise ModelError(
                f"Failed to load Whisper model: {e}", component="WhisperProcessor"
            )

    async def shutdown(self) -> None:
        """Shutdown Whisper processor."""
        try:
            self.model = None

            # Clear GPU cache
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass  # torch not available

            self._initialized = False
            logger.info("Whisper processor shutdown complete")

        except Exception as e:
            logger.error(f"Error during Whisper processor shutdown: {e}")

    async def process_audio(
        self, audio: AudioData, language: Optional[str] = None
    ) -> AudioResult:
        """Process audio and return transcription result."""
        if not self._initialized:
            raise AudioProcessingError(
                "Whisper processor not initialized", component="WhisperProcessor"
            )

        try:
            start_time = time.time()

            # Validate audio data
            if not await self._validate_audio(audio):
                return await self._create_error_result("Invalid audio data")

            # Convert audio to numpy array for Whisper
            audio_array = await self._prepare_audio(audio)

            # Transcribe audio
            result = self.model.transcribe(  # type: ignore
                audio_array,
                language=language or self.config.interaction.stt_language,
                fp16=(self.config.gpu.precision == "fp16"),
                verbose=False,
            )

            processing_time = time.time() - start_time

            # Extract text and confidence
            text = result["text"].strip()
            confidence = self._calculate_confidence(result)

            return AudioResult(
                text=text,
                metadata={
                    "model": self.model_name,
                    "language": result.get("language", "en"),
                    "confidence": confidence,
                    "segments": len(result.get("segments", [])),
                    "processing_time": processing_time,
                },
                processing_time=processing_time,
                quality_score=confidence,
            )

        except Exception as e:
            logger.error(f"Error processing audio with Whisper: {e}")
            return await self._create_error_result(f"Whisper processing failed: {e}")

    async def get_embeddings(self, audio: AudioData) -> AudioResult:  # type: ignore
        """Extract embeddings from audio using Whisper encoder."""
        if not self._initialized:
            raise AudioProcessingError(
                "Whisper processor not initialized", component="WhisperProcessor"
            )

        try:
            start_time = time.time()

            # Validate audio data
            if not await self._validate_audio(audio):
                return await self._create_error_result("Invalid audio data")

            # Convert audio to numpy array
            audio_array = await self._prepare_audio(audio)

            # Get encoder embeddings
            import torch

            # Ensure whisper reference is bound (tests may patch it)
            global whisper
            _logmel = getattr(whisper, "log_mel_spectrogram", None)
            if not callable(_logmel):
                import whisper as _whisper

                whisper = _whisper

            with torch.no_grad():
                # Encode audio
                mel = whisper.log_mel_spectrogram(audio_array).to(self.device)
                audio_features = self.model.encoder(mel.unsqueeze(0))  # type: ignore

                # Get embeddings (average pooling)
                embeddings = (
                    audio_features.mean(dim=1).squeeze(0).cpu().numpy().tolist()
                )

            processing_time = time.time() - start_time

            return AudioResult(
                embeddings=embeddings,
                metadata={
                    "model": self.model_name,
                    "embedding_dim": len(embeddings),
                    "processing_time": processing_time,
                },
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Error extracting embeddings with Whisper: {e}")
            return await self._create_error_result(
                f"Whisper embedding extraction failed: {e}"
            )

    async def _validate_audio(self, audio: AudioData) -> bool:
        """Validate audio data for Whisper processing."""
        if not audio.data:
            return False

        if audio.sample_rate != self.config.interaction.sample_rate:
            logger.warning(
                f"Audio sample rate {audio.sample_rate}Hz, expected "
                f"{self.config.interaction.sample_rate}Hz"
            )

        if audio.duration < 0.1:  # Minimum 100ms
            logger.warning("Audio too short for reliable transcription")
            return False

        if audio.duration > 30.0:  # Maximum 30 seconds
            logger.warning("Audio too long, may cause memory issues")

        return True

    async def _prepare_audio(self, audio: AudioData) -> bytes:
        """Prepare audio data for Whisper processing."""
        import io

        import numpy as np
        import soundfile as sf

        # Load audio from bytes
        audio_io = io.BytesIO(audio.data)
        audio_array, sample_rate = sf.read(audio_io)

        # Resample to configured rate if needed
        target_sr = self.config.interaction.sample_rate
        if sample_rate != target_sr:
            import librosa

            audio_array = librosa.resample(
                audio_array, orig_sr=sample_rate, target_sr=target_sr
            )

        # Convert to float32 and normalize
        audio_array = audio_array.astype(np.float32)
        if audio_array.max() > 1.0:
            audio_array = audio_array / audio_array.max()

        return audio_array  # type: ignore

    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score from Whisper result."""
        try:
            segments = result.get("segments", [])
            if not segments:
                return 0.0

            # Average confidence across segments
            confidences = [seg.get("avg_logprob", 0.0) for seg in segments]
            avg_confidence = sum(confidences) / len(confidences)

            # Convert log probability to confidence (0-1)
            confidence = min(1.0, max(0.0, (avg_confidence + 1.0) / 2.0))
            return confidence  # type: ignore

        except Exception:
            return 0.5  # Default confidence

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage for the model."""
        # Rough estimates for different Whisper models
        memory_estimates = {
            "tiny": 1.0,
            "base": 1.5,
            "small": 2.0,
            "medium": 5.0,
            "large": 10.0,
        }
        return memory_estimates.get(self.model_name, 2.0)

    async def _create_error_result(self, error_message: str) -> AudioResult:
        """Create an error result."""
        return AudioResult(
            error=error_message,
            metadata={"component": "WhisperProcessor", "error": True},
        )

    async def get_available_languages(self) -> List[str]:
        """Get list of supported languages."""
        return [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "ru",
            "ja",
            "ko",
            "zh",
            "ar",
            "hi",
            "th",
            "vi",
            "tr",
            "pl",
            "nl",
            "sv",
            "da",
            "no",
        ]

    async def set_language(self, language: str) -> None:
        """Set default language for transcription."""
        if language not in await self.get_available_languages():
            raise ValueError(f"Unsupported language: {language}")

        # Note: This would need to be handled differently in a real implementation
        # For now, we'll just log the language change
        logger.info(f"Whisper language set to: {language}")

    async def get_model_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities and limitations."""
        return {
            "supported_languages": await self.get_available_languages(),
            "max_audio_length": 30.0,  # seconds
            "min_audio_length": 0.1,  # seconds
            "supported_formats": ["wav", "mp3", "flac", "m4a"],
            "sample_rate": 16000,
            "channels": 1,
            "memory_usage_gb": self._estimate_memory_usage(),
            "processing_speed": "real_time",
            "accuracy": "high",
        }
