"""
Wav2Vec2 processor for speech recognition.

Implements Facebook's Wav2Vec2 for high-quality speech-to-text conversion
with secure PyTorch compatibility.
"""

# CRITICAL: Import torch_init FIRST to set environment variables before any torch imports
# isort: off
from .....core import torch_init  # noqa: F401

# isort: on

import logging
import time
from typing import Any, Dict, List, Optional

import torch

try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    torchaudio = None
    TORCHAUDIO_AVAILABLE = False

from .....core.config import Config
from .....core.exceptions import AudioProcessingError, ModelError
from .....core.model_utils import get_local_model_path
from .....core.protocols import AudioData, AudioResult, BaseAudioProcessor, ModelInfo

logger = logging.getLogger(__name__)


class Wav2Vec2Processor(BaseAudioProcessor):
    """Wav2Vec2-based speech recognition processor."""

    def __init__(
        self,
        config: Config,
        model_name: str = "facebook/wav2vec2-base-960h",
        force_cpu: bool = False,
    ):
        super().__init__(f"wav2vec2_{model_name.split('/')[-1]}")
        self.config: Config = config
        self.model_name = model_name
        self.force_cpu = force_cpu
        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None
        self.device: Optional[torch.device] = None

    async def initialize(self) -> None:
        """Initialize Wav2Vec2 model from local path."""
        try:
            logger.info(f"Loading Wav2Vec2 model: {self.model_name}")

            # Get local model path from registry using shared utility
            local_path = get_local_model_path(self.config, "stt", self.model_name)

            if not local_path:
                raise ModelError(
                    f"Local model not found for {self.model_name}. "
                    f"Run 'make download-models' to download models locally."
                )

            logger.info(f"Using local model: {local_path}")

            # Set device - respect force_cpu setting to avoid CUDA conflicts
            if self.force_cpu:
                self.device = torch.device("cpu")
                logger.info(
                    "Using CPU for STT (forced by hardware config to avoid CUDA conflicts)"
                )
            else:
                try:
                    self.device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    logger.info(f"Using device: {self.device}")
                except Exception as e:
                    logger.warning(f"GPU detection failed, using CPU: {e}")
                    self.device = torch.device("cpu")
                    logger.info(f"Using device: {self.device}")

            # Import transformers
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            # Load model and processor using local paths
            self.processor = Wav2Vec2Processor.from_pretrained(
                local_path,
                local_files_only=True,  # nosec B615 - local_files_only=True prevents network access
            )
            self.model = Wav2Vec2ForCTC.from_pretrained(
                local_path,
                local_files_only=True,  # nosec B615 - local_files_only=True prevents network access
            )

            # Move model to device
            self.model.to(self.device)
            self.model.eval()

            # Create model info
            self.model_info = ModelInfo(
                name="wav2vec2",
                type="stt",
                size=f"{self.model_name} model",
                memory_usage=f"{self._estimate_memory_usage():.2f}GB",
                precision="fp16" if self.device.type == "cuda" else "fp32",
                quantization="none",
                loaded_at=time.time(),
                status="loaded",
            )

            self._initialized = True
            logger.info(f"Wav2Vec2 model {self.model_name} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Wav2Vec2 model: {e}")
            raise ModelError(
                f"Failed to load Wav2Vec2 model: {e}", component="Wav2Vec2Processor"
            )

    async def shutdown(self) -> None:
        """Shutdown Wav2Vec2 processor."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            if self.processor is not None:
                del self.processor
                self.processor = None

            self._initialized = False
            logger.info("Wav2Vec2 processor shutdown complete")

        except Exception as e:
            logger.error(f"Error during Wav2Vec2 processor shutdown: {e}")

    async def process_audio(self, audio: AudioData) -> AudioResult:
        """Process audio and return transcription."""
        if not self._initialized or self.model is None or self.processor is None:
            raise AudioProcessingError(
                "Wav2Vec2 processor not initialized", component="Wav2Vec2Processor"
            )

        try:
            start_time = time.time()

            # Convert audio to numpy array (support bytes or ndarray)
            raw_data: Any = audio.data  # allow runtime to pass bytes or ndarray
            try:
                from src.character_ai.core.audio_io.audio_utils import (
                    convert_audio_input,
                )

                audio_array = convert_audio_input(raw_data)
            except ValueError as e:
                return await self._create_error_result(str(e))

            # Guard against empty input after conversion
            if audio_array.size == 0:
                return await self._create_error_result(
                    "Empty audio array after conversion"
                )

            # Resample if necessary
            if audio.sample_rate != 16000:
                from src.character_ai.core.audio_io.audio_utils import resample_audio

                audio_array = resample_audio(audio_array, audio.sample_rate, 16000)

            # Process with Wav2Vec2
            inputs = self.processor(
                audio_array, sampling_rate=16000, return_tensors="pt", padding=True
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                logits = self.model(**inputs).logits

            # Decode predictions
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            processing_time = time.time() - start_time

            return AudioResult(
                text=transcription,
                metadata={
                    "model": "wav2vec2",
                    "model_name": self.model_name,
                    "processing_time": processing_time,
                    "sample_rate": audio.sample_rate,
                    "duration": audio.duration,
                    "confidence": self._calculate_confidence(logits),
                },
                processing_time=processing_time,
                quality_score=0.9,
            )

        except Exception as e:
            logger.error(f"Error processing audio with Wav2Vec2: {e}")
            return await self._create_error_result(f"Wav2Vec2 processing failed: {e}")

    def _calculate_confidence(self, logits: torch.Tensor) -> float:
        """Calculate confidence score from model logits."""
        try:
            # Calculate softmax probabilities
            probs = torch.softmax(logits, dim=-1)
            # Get max probability for each token
            max_probs = torch.max(probs, dim=-1)[0]
            # Return average confidence
            return float(torch.mean(max_probs).item())
        except Exception:
            return 0.5  # Default confidence

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage for Wav2Vec2."""
        # Wav2Vec2 base model is approximately 300MB
        return 0.3

    async def _create_error_result(self, error_message: str) -> AudioResult:
        """Create an error result."""
        return AudioResult(
            error=error_message,
            metadata={"component": "Wav2Vec2Processor", "error": True},
        )

    async def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        # Wav2Vec2 base model supports English
        return ["en"]

    async def get_model_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities."""
        return {
            "supported_languages": await self.get_supported_languages(),
            "max_audio_length": 30.0,  # seconds
            "min_audio_length": 0.1,  # seconds
            "sample_rate": 16000,
            "channels": 1,
            "memory_usage_gb": self._estimate_memory_usage(),
            "processing_speed": "fast",
            "accuracy": "high",
        }

    async def get_embeddings(self, audio: AudioData) -> Any:
        """Extract embeddings from audio data."""
        try:
            # Wav2Vec2 doesn't typically extract embeddings for STT
            # This is more relevant for other audio processing tasks
            return {
                "embeddings": [],
                "metadata": {
                    "component": "Wav2Vec2Processor",
                    "note": "No embeddings available for STT",
                },
                "processing_time": 0.0,
            }
        except Exception as e:
            logger.error(f"Error extracting embeddings: {e}")
            return {
                "embeddings": [],
                "metadata": {"component": "Wav2Vec2Processor", "error": str(e)},
                "processing_time": 0.0,
                "error": str(e),
            }
