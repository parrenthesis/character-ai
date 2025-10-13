"""Wake word detection engines for voice activation."""

import asyncio
import logging
import time
from typing import List, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class WakeWordEngine(Protocol):
    """Protocol for wake word detection engines."""

    async def detect(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """Return (detected, confidence)."""


class EnergyBasedWakeWord:
    """Simple energy-based wake word detection."""

    def __init__(self, threshold: float = 0.6, min_duration_s: float = 0.3):
        self.threshold = threshold
        self.min_duration_s = min_duration_s
        self._speech_start_time: Optional[float] = None

    async def detect(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """Detect based on audio energy."""
        energy = np.mean(np.abs(audio_chunk))
        current_time = time.time()

        if energy > self.threshold:
            if self._speech_start_time is None:
                self._speech_start_time = current_time
            else:
                # Check if we've had sustained speech for min_duration_s
                duration = current_time - self._speech_start_time
                if duration >= self.min_duration_s:
                    self._speech_start_time = None  # Reset
                    return (True, float(energy))
        else:
            self._speech_start_time = None  # Reset on silence

        return (False, float(energy))


class OpenWakeWordEngine:
    """OpenWakeWord integration (Apache 2.0 license)."""

    def __init__(self, model_path: str, threshold: float = 0.5):
        self.model_path = model_path
        self.threshold = threshold
        self._model = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize OpenWakeWord model."""
        try:
            from openwakeword import Model

            self._model = Model(wakeword_models=[self.model_path])
            self._initialized = True
            logger.info(f"✅ OpenWakeWord model loaded: {self.model_path}")
        except ImportError:
            raise ImportError(
                "openwakeword not installed. Install with: pip install openwakeword"
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenWakeWord: {e}")
            raise

    async def detect(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """Detect wake word using OpenWakeWord."""
        if not self._initialized:
            return (False, 0.0)

        if self._model is None:
            return (False, 0.0)  # pragma: no cover

        try:  # type: ignore[unreachable]
            # OpenWakeWord expects 16kHz mono audio
            if audio_chunk.ndim > 1:
                audio_chunk = np.mean(audio_chunk, axis=1)

            # Ensure correct sample rate (16kHz)
            if len(audio_chunk) != 16000:  # Assuming 1 second chunks
                # Resample if needed (simple linear interpolation)
                import scipy.signal

                target_length = 16000
                if len(audio_chunk) != target_length:
                    audio_chunk = scipy.signal.resample(audio_chunk, target_length)

            prediction = self._model.predict(audio_chunk)
            confidence = max(prediction.values()) if prediction else 0.0
            return (confidence > self.threshold, confidence)
        except Exception as e:
            logger.error(f"OpenWakeWord detection error: {e}")
            return (False, 0.0)


class WakeWordDetector:
    """Manages wake word detection with pluggable engines."""

    def __init__(self, config: dict, character_wake_words: Optional[List[str]] = None):
        self.config = config
        self.wake_words = character_wake_words or []
        self.engine = self._create_engine()
        self.enabled = config.get("enabled", False)
        self.cooldown_period = config.get("global_settings", {}).get(
            "cooldown_period_s", 2.0
        )
        self.last_detection_time = 0.0
        self._initialization_task: Optional[asyncio.Task] = None

    def _create_engine(self) -> WakeWordEngine:
        """Create wake word engine based on config."""
        engine_name = self.config.get("engine", "energy_based")
        engine_config = self.config.get("engines", {}).get(engine_name, {})

        if engine_name == "energy_based":
            return EnergyBasedWakeWord(
                threshold=engine_config.get("threshold", 0.6),
                min_duration_s=engine_config.get("min_duration_s", 0.3),
            )
        elif engine_name == "openwakeword":
            engine = OpenWakeWordEngine(
                model_path=engine_config.get(
                    "model_path", "models/wake_words/openwakeword.onnx"
                ),
                threshold=engine_config.get("threshold", 0.5),
            )
            # Initialize asynchronously
            self._initialization_task = asyncio.create_task(engine.initialize())
            return engine
        else:
            raise ValueError(f"Unknown wake word engine: {engine_name}")

    async def process_audio(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """Check if wake word detected in audio chunk."""
        if not self.enabled:
            return (True, 1.0)  # Always pass through if disabled

        # Wait for initialization if needed
        if self._initialization_task and not self._initialization_task.done():
            try:
                await self._initialization_task
            except Exception as e:
                logger.error(f"Wake word engine initialization failed: {e}")
                return (False, 0.0)

        # Check cooldown
        current_time = time.time()
        if current_time - self.last_detection_time < self.cooldown_period:
            return (False, 0.0)

        detected, confidence = await self.engine.detect(audio_chunk)
        if detected:
            self.last_detection_time = current_time
            logger.info(f"Wake word detected (confidence: {confidence:.2f})")

        return (detected, confidence)

    def reset_cooldown(self) -> None:
        """Reset cooldown period (useful for testing)."""
        self.last_detection_time = 0.0
