"""
Configuration and initialization for Coqui TTS processor.
"""

import logging
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Optional

import torch
from TTS.api import TTS

from ......core.config import Config
from ......core.exceptions import ModelError
from ......core.model_utils import get_local_model_path
from ......core.protocols import ModelInfo

logger = logging.getLogger(__name__)


class CoquiConfig:
    """Configuration manager for Coqui TTS processor."""

    def __init__(
        self,
        config: Config,
        model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
        gpu_device: Optional[str] = None,
        use_half_precision: Optional[bool] = None,
    ):
        self.config = config
        self.model_name = model_name
        self.tts: Any = None

        # Store GPU device preference - actual detection happens in initialize()
        self.gpu_device = gpu_device
        self.device = "cpu"  # Default to CPU, will be updated in initialize()
        self.use_gpu = False
        self.use_half_precision = use_half_precision

        # Model configuration
        self.model_loaded = False
        self.model_info: Optional[ModelInfo] = None

    async def initialize(self) -> None:
        """Initialize the Coqui TTS model from local path."""
        try:
            logger.info(f"Initializing Coqui TTS with model: {self.model_name}")

            # Get local model path from registry using shared utility
            local_path = get_local_model_path(self.config, "tts", self.model_name)

            if not local_path:
                raise ModelError(
                    f"Local model not found for {self.model_name}. "
                    f"Run 'make download-models' to download models locally."
                )

            logger.info(f"Using local TTS model: {local_path}")

            # Determine device configuration
            self._configure_device()

            # Suppress TTS warnings during initialization
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with redirect_stdout(open("/dev/null", "w")):
                    with redirect_stderr(open("/dev/null", "w")):
                        # Initialize TTS with local paths
                        from pathlib import Path

                        model_dir = Path(local_path)
                        config_path = model_dir / "config.json"

                        self.tts = TTS(
                            model_path=str(model_dir),
                            config_path=str(config_path),
                            progress_bar=False,
                        )

            # Move model to appropriate device
            if self.use_gpu and hasattr(self.tts, "to"):
                self.tts.to(self.device)

            # Set half precision if requested and supported
            if self.use_half_precision and self.use_gpu:
                if hasattr(self.tts, "half"):
                    self.tts.half()
                    logger.info("Enabled half precision for TTS model")

            self.model_loaded = True
            logger.info(f"Coqui TTS initialized successfully on {self.device}")

            # Get model information
            self.model_info = await self._get_model_info()

        except Exception as e:
            logger.error(f"Failed to initialize Coqui TTS: {e}")
            self.model_loaded = False
            raise ModelError(f"Coqui TTS initialization failed: {e}")

    def _configure_device(self) -> None:
        """Configure GPU/CPU device for the model."""
        if self.gpu_device:
            self.device = self.gpu_device
            self.use_gpu = True
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.use_gpu = True
            logger.info("CUDA available, using GPU for TTS")
        else:
            self.device = "cpu"
            self.use_gpu = False
            logger.info("CUDA not available, using CPU for TTS")

    async def shutdown(self) -> None:
        """Shutdown the Coqui TTS model and release resources."""
        try:
            if self.tts is not None:
                # Clear model from memory
                del self.tts
                self.tts = None

            # Clear CUDA cache if using GPU
            if self.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.model_loaded = False
            logger.info("Coqui TTS model shutdown complete")

        except Exception as e:
            logger.error(f"Error during Coqui TTS shutdown: {e}")

    async def _get_model_info(self) -> ModelInfo:
        """Get information about the loaded model."""
        try:
            if not self.model_loaded or self.tts is None:
                return ModelInfo(
                    name=self.model_name,
                    type="tts",
                    size="unknown",
                    memory_usage="0.0",
                    precision="float32",
                    quantization="none",
                    loaded_at=0.0,
                )

            # Get model capabilities
            await self._get_model_capabilities()

            # Estimate memory usage
            memory_usage = await self._estimate_memory_usage()

            return ModelInfo(
                name=self.model_name,
                type="tts",
                size="unknown",
                memory_usage=str(memory_usage),
                precision="float32",
                quantization="none",
                loaded_at=0.0,
            )

        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return ModelInfo(
                name=self.model_name,
                type="tts",
                size="unknown",
                memory_usage="0.0",
                precision="float32",
                quantization="none",
                loaded_at=0.0,
            )

    async def _get_model_capabilities(self) -> list[str]:
        """Get model capabilities."""
        capabilities = ["text_to_speech", "voice_cloning"]

        # Check if model supports streaming
        if hasattr(self.tts, "synthesize_stream"):
            capabilities.append("streaming")

        # Check if model supports multiple languages
        if hasattr(self.tts, "languages"):
            capabilities.append("multilingual")

        return capabilities

    async def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of the model in GB."""
        try:
            if not self.use_gpu or not torch.cuda.is_available():
                return 0.0

            # Get current memory usage
            current_memory = torch.cuda.memory_allocated() / (1024**3)

            # Estimate model memory (rough approximation)
            model_memory = 0.5  # Base model size in GB

            return current_memory + model_memory

        except Exception:
            return 0.0
