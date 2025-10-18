"""Model warmup utilities for optimizing inference performance."""

import asyncio
import logging
from typing import Any, Dict, Optional

import numpy as np

from .protocols import AudioData

logger = logging.getLogger(__name__)


class ModelWarmup:
    """Handles model warmup to optimize inference performance."""

    def __init__(self, loaded_models: Dict[str, Any]):
        self.loaded_models = loaded_models

    async def warmup_all_models(
        self, character: Optional[Any] = None
    ) -> Dict[str, bool]:
        """Warm up all loaded models with dummy inference."""
        results = {}

        # Parallel warmup for speed
        warmup_tasks = []
        if "stt" in self.loaded_models:
            warmup_tasks.append(self._warmup_stt())
        if "llm" in self.loaded_models:
            warmup_tasks.append(self._warmup_llm())
        if "tts" in self.loaded_models:
            warmup_tasks.append(self._warmup_tts(character))

        if warmup_tasks:
            warmup_results = await asyncio.gather(*warmup_tasks, return_exceptions=True)
            for i, model_type in enumerate(["stt", "llm", "tts"]):
                if model_type in self.loaded_models:
                    results[model_type] = not isinstance(warmup_results[i], Exception)

        return results

    async def _warmup_stt(self) -> None:
        """Run dummy audio through STT to warm up caches."""
        logger.info("Warming up STT model...")
        processor = self.loaded_models.get("stt")
        if processor:
            # 1 second of silence
            dummy_audio = AudioData(
                data=np.zeros(16000, dtype=np.float32).tobytes(),
                sample_rate=16000,
                channels=1,
                duration=1.0,
            )
            await processor.process_audio(dummy_audio)
            logger.info("✅ STT warmed up")

    async def _warmup_llm(self) -> None:
        """Run dummy prompt through LLM to warm up."""
        logger.info("Warming up LLM model...")
        processor = self.loaded_models.get("llm")
        if processor:
            # Short dummy prompt to trigger JIT/cache
            await processor.process_text("Hello")
            logger.info("✅ LLM warmed up")

    async def _warmup_tts(self, character: Optional[Any] = None) -> None:
        """Run dummy text through TTS to warm up."""
        logger.info("Warming up TTS model...")
        processor = self.loaded_models.get("tts")
        if processor:
            try:
                if character and hasattr(processor, "synthesize_speech"):
                    # Get character's voice path for proper warmup
                    voice_path = None
                    if hasattr(character, "metadata") and character.metadata:
                        voice_path = character.metadata.get("voice_path")

                    if voice_path:
                        logger.info(
                            f"Warming up TTS with character voice: {voice_path}"
                        )
                        # Use the character's voice for warmup
                        await processor.synthesize_speech(
                            text="Hello",
                            language="en",
                            speed=1.0,
                            voice_path=voice_path,
                        )
                        logger.info("✅ TTS warmed up with character voice")
                    else:
                        logger.warning(
                            "No voice path found for character - TTS warmup skipped"
                        )
                else:
                    logger.warning(
                        "No character provided or processor doesn't support synthesize_speech - TTS warmup skipped"
                    )
            except Exception as e:
                logger.warning(f"TTS warmup failed: {e}")
                # This is expected for voice cloning models without a proper voice path
