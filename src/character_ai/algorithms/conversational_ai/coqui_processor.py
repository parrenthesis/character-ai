# CRITICAL: Import torch_init FIRST to set environment variables before any torch imports
# isort: off
from ...core import torch_init  # noqa: F401

# isort: on

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from ...core.config import Config
from ...core.protocols import AudioData, AudioResult, BaseAudioProcessor, ModelInfo
from .coqui import CoquiConfig, CoquiTTSProcessor, CoquiVoiceCloner

logger = logging.getLogger(__name__)


class CoquiProcessor(BaseAudioProcessor):
    """
    Coqui TTS processor for text-to-speech synthesis and voice cloning.

    Implements Coqui TTS for high-quality text-to-speech conversion with
    native voice cloning capabilities.
    """

    def __init__(
        self,
        config: Config,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        gpu_device: Optional[str] = None,
        use_half_precision: Optional[bool] = None,
    ):
        self.config = config
        self.model_name = model_name

        # Initialize components
        self.coqui_config = CoquiConfig(
            config=config,
            model_name=model_name,
            gpu_device=gpu_device,
            use_half_precision=use_half_precision,
        )

        self.tts_processor: Optional[CoquiTTSProcessor] = None
        self.voice_cloner: Optional[CoquiVoiceCloner] = None

    async def initialize(self) -> None:
        """Initialize the Coqui TTS model."""
        await self.coqui_config.initialize()

        # Initialize processors
        self.tts_processor = CoquiTTSProcessor(
            tts_model=self.coqui_config.tts, config=self.coqui_config
        )

        self.voice_cloner = CoquiVoiceCloner(tts_model=self.coqui_config.tts)

    async def shutdown(self) -> None:
        """Shutdown the Coqui TTS model and release resources."""
        await self.coqui_config.shutdown()
        self.tts_processor = None
        self.voice_cloner = None

    async def process_audio(
        self, audio: AudioData, language: Optional[str] = None
    ) -> AudioResult:
        """Process audio data."""
        if not self.tts_processor:
            return await self._create_error_result("TTS processor not initialized")
        return await self.tts_processor.process_audio(audio, language)

    async def synthesize_speech(
        self,
        text: str,
        language: str = "en",
        speaker_id: Optional[str] = None,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> AudioResult:
        """Synthesize speech from text."""
        if not self.tts_processor:
            return await self._create_error_result("TTS processor not initialized")
        return await self.tts_processor.synthesize_speech(
            text, language, speaker_id, speed, **kwargs
        )

    async def synthesize_speech_stream(
        self,
        text: str,
        language: str = "en",
        speaker_id: Optional[str] = None,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> AsyncGenerator[AudioData, None]:
        """Synthesize speech in streaming mode."""
        if not self.tts_processor:
            logger.error("TTS processor not initialized")
            return
        async for audio_data in self.tts_processor.synthesize_speech_stream(
            text, language, speaker_id, speed, **kwargs
        ):
            yield audio_data

    async def clone_voice(
        self, reference_audio_path: str, text: str, output_path: str
    ) -> AudioResult:
        """Clone a voice from reference audio."""
        if not self.voice_cloner:
            return await self._create_error_result("Voice cloner not initialized")
        return await self.voice_cloner.clone_voice(
            reference_audio_path, text, output_path
        )

    async def get_model_info(self) -> ModelInfo:
        """Get information about the loaded model."""
        return self.coqui_config.model_info or ModelInfo(
            name=self.model_name,
            type="tts",
            size="unknown",
            memory_usage="0.0",
            precision="float32",
            quantization="none",
            loaded_at=0.0,
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
            "zh-cn",
            "ja",
            "hu",
            "ko",
            "hi",
            "th",
            "sv",
            "da",
            "no",
            "fi",
            "el",
            "he",
            "uk",
            "ca",
            "eu",
            "bg",
            "hr",
            "sk",
            "sl",
            "et",
            "lv",
            "lt",
            "mt",
            "ga",
            "cy",
            "is",
            "mk",
            "sq",
            "sr",
            "bs",
            "ro",
            "vi",
            "id",
            "ms",
            "tl",
            "sw",
            "am",
            "az",
            "bn",
            "my",
            "gu",
            "ha",
            "ig",
            "yo",
            "zu",
            "xh",
            "af",
            "st",
            "tn",
            "ts",
            "ss",
            "nr",
            "nso",
            "ve",
            "ts",
        ]

    async def get_model_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities."""
        return {
            "text_to_speech": True,
            "voice_cloning": True,
            "streaming": True,
            "multilingual": True,
            "speed_control": True,
            "speaker_selection": True,
        }

    async def get_embeddings(self, audio: AudioData) -> Any:
        """Extract embeddings from audio data."""
        if not self.voice_cloner:
            return None
        return await self.voice_cloner.get_embeddings(audio)

    async def _create_error_result(self, error_message: str) -> AudioResult:
        """Helper to create an error AudioResult."""
        return AudioResult(
            text="",
            audio_data=None,
            metadata={"error": error_message},
            processing_time=0.0,
        )
