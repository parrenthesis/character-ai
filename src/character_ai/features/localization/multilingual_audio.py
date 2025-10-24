"""
Multi-language audio processing system.

Provides language-aware TTS and STT processing with automatic language detection,
cultural adaptations, and voice characteristics per language.

⚠️  BETA FEATURE: Multi-language audio is experimental and may change.
    Use with caution in production environments.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from ...algorithms.conversational_ai.processors.tts.coqui_processor import (
    CoquiProcessor,
)
from ...core.config import Config
from ...core.exceptions import AudioProcessingError
from ...core.protocols import AudioData, AudioResult
from .language_support import LanguageCode, create_localization_manager

logger = logging.getLogger(__name__)


class AudioLanguageMode(Enum):
    """Audio processing language modes."""

    AUTO_DETECT = "auto_detect"
    SPECIFIC_LANGUAGE = "specific_language"
    FALLBACK_ENGLISH = "fallback_english"


@dataclass
class LanguageAudioConfig:
    """Language-specific audio configuration."""

    language_code: LanguageCode
    tts_model: str
    stt_model: str
    voice_characteristics: Dict[str, Any]
    cultural_adaptations: Dict[str, Any]
    sample_rate: int = 22050
    channels: int = 1
    quality: str = "high"


@dataclass
class MultiLanguageAudioResult:
    """Result of multi-language audio processing."""

    audio_result: AudioResult
    detected_language: LanguageCode
    confidence: float
    cultural_adaptations: Dict[str, Any]
    voice_characteristics: Dict[str, Any]
    processing_time_ms: float


class MultiLanguageTTSService:
    """Multi-language TTS manager with cultural adaptations."""

    def __init__(self, config: Config):
        self.config = config
        self.localization_manager: Any = create_localization_manager()
        self.tts_processor: Optional[CoquiProcessor] = None
        self.language_configs: Dict[LanguageCode, LanguageAudioConfig] = {}
        self._initialized = False

        # Initialize language-specific configurations
        self._setup_language_configs()

    def _setup_language_configs(self) -> None:
        """Setup language-specific audio configurations."""
        # English configuration
        self.language_configs[LanguageCode.ENGLISH] = LanguageAudioConfig(
            language_code=LanguageCode.ENGLISH,
            tts_model="coqui",
            stt_model="wav2vec2-base",
            voice_characteristics={
                "preferred_pitch": "medium",
                "speaking_rate": "normal",
                "emotional_range": "moderate",
                "formality_level": "neutral",
            },
            cultural_adaptations={
                "greeting_style": "friendly",
                "conversation_style": "casual",
            },
        )

        # Spanish configuration
        self.language_configs[LanguageCode.SPANISH] = LanguageAudioConfig(
            language_code=LanguageCode.SPANISH,
            tts_model="coqui",
            stt_model="wav2vec2-base",
            voice_characteristics={
                "preferred_pitch": "medium",
                "speaking_rate": "normal",
                "emotional_range": "expressive",
                "formality_level": "friendly",
            },
            cultural_adaptations={
                "greeting_style": "warm",
                "conversation_style": "friendly",
            },
        )

        # French configuration
        self.language_configs[LanguageCode.FRENCH] = LanguageAudioConfig(
            language_code=LanguageCode.FRENCH,
            tts_model="coqui",
            stt_model="wav2vec2-base",
            voice_characteristics={
                "preferred_pitch": "medium",
                "speaking_rate": "moderate",
                "emotional_range": "sophisticated",
                "formality_level": "formal",
            },
            cultural_adaptations={
                "greeting_style": "formal",
                "conversation_style": "polite",
            },
        )

        # Chinese configuration
        self.language_configs[LanguageCode.CHINESE] = LanguageAudioConfig(
            language_code=LanguageCode.CHINESE,
            tts_model="coqui",
            stt_model="wav2vec2-base",
            voice_characteristics={
                "preferred_pitch": "medium",
                "speaking_rate": "measured",
                "emotional_range": "controlled",
                "formality_level": "respectful",
            },
            cultural_adaptations={
                "greeting_style": "respectful",
                "conversation_style": "hierarchical",
            },
        )

        # Japanese configuration
        self.language_configs[LanguageCode.JAPANESE] = LanguageAudioConfig(
            language_code=LanguageCode.JAPANESE,
            tts_model="coqui",
            stt_model="wav2vec2-base",
            voice_characteristics={
                "preferred_pitch": "medium",
                "speaking_rate": "measured",
                "emotional_range": "controlled",
                "formality_level": "respectful",
            },
            cultural_adaptations={
                "greeting_style": "respectful",
                "conversation_style": "polite",
            },
        )

        # Korean configuration
        self.language_configs[LanguageCode.KOREAN] = LanguageAudioConfig(
            language_code=LanguageCode.KOREAN,
            tts_model="coqui",
            stt_model="wav2vec2-base",
            voice_characteristics={
                "preferred_pitch": "medium",
                "speaking_rate": "measured",
                "emotional_range": "controlled",
                "formality_level": "respectful",
            },
            cultural_adaptations={
                "greeting_style": "respectful",
                "conversation_style": "polite",
            },
        )

        # Arabic configuration
        self.language_configs[LanguageCode.ARABIC] = LanguageAudioConfig(
            language_code=LanguageCode.ARABIC,
            tts_model="coqui",
            stt_model="wav2vec2-base",
            voice_characteristics={
                "preferred_pitch": "medium",
                "speaking_rate": "measured",
                "emotional_range": "controlled",
                "formality_level": "respectful",
            },
            cultural_adaptations={
                "greeting_style": "respectful",
                "conversation_style": "polite",
            },
        )

    async def initialize(self) -> None:
        """Initialize the multi-language TTS manager."""
        try:
            logger.info("Initializing multi-language TTS manager")

            # Initialize Coqui processor with multilingual model
            from .config import DEFAULT_COQUI_MODEL

            tts_model = getattr(self.config.models, "coqui_model", DEFAULT_COQUI_MODEL)
            self.tts_processor = CoquiProcessor(self.config, model_name=tts_model)
            await self.tts_processor.initialize()

            self._initialized = True
            logger.info("Multi-language TTS manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize multi-language TTS manager: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the multi-language TTS manager."""
        try:
            if self.tts_processor:
                await self.tts_processor.shutdown()
                self.tts_processor = None

            self._initialized = False
            logger.info("Multi-language TTS manager shutdown complete")

        except Exception as e:
            logger.error(f"Error during multi-language TTS manager shutdown: {e}")

    async def synthesize_speech(
        self,
        text: str,
        language_code: Optional[LanguageCode] = None,
        voice_style: Optional[str] = None,
        cultural_adaptations: Optional[Dict[str, Any]] = None,
    ) -> MultiLanguageAudioResult:
        """Synthesize speech with language-specific adaptations."""
        if not self._initialized:
            raise RuntimeError("Multi-language TTS manager not initialized")

        start_time = time.time()

        try:
            # Auto-detect language if not provided
            if language_code is None:
                detection_result = self.localization_manager.detect_and_set_language(
                    text
                )
                language_code = detection_result.detected_language

            # Get language configuration
            lang_config = self.language_configs.get(language_code)
            if not lang_config:
                # Fallback to English
                lang_config = self.language_configs[LanguageCode.ENGLISH]
                language_code = LanguageCode.ENGLISH

            # Get cultural adaptations
            if cultural_adaptations is None:
                cultural_adaptations = (
                    self.localization_manager.get_cultural_adaptations(language_code)
                )

            # Get voice characteristics
            voice_characteristics = self.localization_manager.get_voice_characteristics(
                language_code
            )

            # Apply cultural adaptations to text
            adapted_text = self._apply_cultural_adaptations(
                text, cultural_adaptations, language_code
            )

            # Synthesize with Coqui
            if self.tts_processor is None:
                raise AudioProcessingError("Coqui processor not initialized")
            audio_result = await self.tts_processor.synthesize_speech(
                text=adapted_text,
                voice_path=None,  # No voice cloning for now
                language=language_code.value,
            )

            processing_time = (time.time() - start_time) * 1000

            return MultiLanguageAudioResult(
                audio_result=audio_result,
                detected_language=language_code,
                confidence=1.0,  # TTS doesn't have confidence like STT
                cultural_adaptations=cultural_adaptations,
                voice_characteristics=voice_characteristics,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            raise

    def _apply_cultural_adaptations(
        self,
        text: str,
        cultural_adaptations: Dict[str, Any],
        language_code: LanguageCode,
    ) -> str:
        """Apply cultural adaptations to text."""
        # For now, return text as-is
        # In the future, this could include:
        # - Formality adjustments
        # - Cultural greeting patterns
        # - Emotional tone adjustments
        # - Regional dialect preferences

        return text

    async def get_supported_languages(self) -> List[LanguageCode]:
        """Get list of supported languages."""
        return list(self.language_configs.keys())

    async def get_language_capabilities(
        self, language_code: LanguageCode
    ) -> Dict[str, Any]:
        """Get capabilities for a specific language."""
        lang_config = self.language_configs.get(language_code)
        if not lang_config:
            return {}

        return {
            "language_code": language_code.value,
            "tts_model": lang_config.tts_model,
            "stt_model": lang_config.stt_model,
            "voice_characteristics": lang_config.voice_characteristics,
            "cultural_adaptations": lang_config.cultural_adaptations,
            "sample_rate": lang_config.sample_rate,
            "channels": lang_config.channels,
            "quality": lang_config.quality,
        }


class MultiLanguageSTTService:
    """Multi-language STT manager with automatic language detection."""

    def __init__(self, config: Config):
        self.config = config
        self.localization_manager: Any = create_localization_manager()
        self.wav2vec2_processor: Optional[Any] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the multi-language STT manager."""
        try:
            logger.info("Initializing multi-language STT manager")

            # Initialize Wav2Vec2 processor
            from ..algorithms.conversational_ai.processors.stt.wav2vec2_processor import (
                Wav2Vec2Processor,
            )

            self.wav2vec2_processor = Wav2Vec2Processor(self.config)
            await self.wav2vec2_processor.initialize()

            self._initialized = True
            logger.info("Multi-language STT manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize multi-language STT manager: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the multi-language STT manager."""
        try:
            if self.wav2vec2_processor:
                await self.wav2vec2_processor.shutdown()
                self.wav2vec2_processor = None

            self._initialized = False
            logger.info("Multi-language STT manager shutdown complete")

        except Exception as e:
            logger.error(f"Error during multi-language STT manager shutdown: {e}")

    async def transcribe_audio(
        self,
        audio: AudioData,
        language_code: Optional[LanguageCode] = None,
        auto_detect: bool = True,
    ) -> MultiLanguageAudioResult:
        """Transcribe audio with language detection."""
        if not self._initialized:
            raise RuntimeError("Multi-language STT manager not initialized")

        start_time = time.time()

        try:
            # Process audio with Wav2Vec2
            if self.wav2vec2_processor is None:
                raise AudioProcessingError("Wav2Vec2 processor not initialized")
            audio_result = await self.wav2vec2_processor.process_audio(
                audio, language=language_code.value if language_code else None
            )

            # Detect language from transcribed text if auto_detect
            detected_language = language_code
            confidence = 1.0

            if auto_detect and audio_result.text:
                detection_result = self.localization_manager.detect_and_set_language(
                    audio_result.text
                )
                detected_language = detection_result.detected_language
                confidence = detection_result.confidence

            # Get cultural adaptations for detected language
            cultural_adaptations = self.localization_manager.get_cultural_adaptations(
                detected_language
            )
            voice_characteristics = self.localization_manager.get_voice_characteristics(
                detected_language
            )

            processing_time = (time.time() - start_time) * 1000

            return MultiLanguageAudioResult(
                audio_result=audio_result,
                detected_language=detected_language or LanguageCode.ENGLISH,
                confidence=confidence,
                cultural_adaptations=cultural_adaptations,
                voice_characteristics=voice_characteristics,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise

    async def get_supported_languages(self) -> List[LanguageCode]:
        """Get list of supported languages."""
        if not self.wav2vec2_processor:
            return []

        # Get supported languages from Wav2Vec2
        wav2vec2_languages = await self.wav2vec2_processor.get_supported_languages()

        # Map to our LanguageCode enum
        supported_languages = []
        for lang in wav2vec2_languages:
            try:
                # Map common language codes
                lang_mapping = {
                    "en": LanguageCode.ENGLISH,
                    "es": LanguageCode.SPANISH,
                    "fr": LanguageCode.FRENCH,
                    "de": LanguageCode.GERMAN,
                    "it": LanguageCode.ITALIAN,
                    "pt": LanguageCode.PORTUGUESE,
                    "ru": LanguageCode.RUSSIAN,
                    "zh": LanguageCode.CHINESE,
                    "ja": LanguageCode.JAPANESE,
                    "ko": LanguageCode.KOREAN,
                    "ar": LanguageCode.ARABIC,
                    "hi": LanguageCode.HINDI,
                }

                if lang in lang_mapping:
                    supported_languages.append(lang_mapping[lang])
            except (ValueError, KeyError):
                continue

        return supported_languages


class MultiLanguageAudioService:
    """Unified multi-language audio processing manager."""

    def __init__(self, config: Config):
        self.config = config
        self.tts_manager: Optional[MultiLanguageTTSService] = None
        self.stt_manager: Optional[MultiLanguageSTTService] = None
        self.localization_manager: Any = create_localization_manager()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the multi-language audio manager."""
        try:
            logger.info("Initializing multi-language audio manager")

            # Initialize TTS and STT managers
            self.tts_manager = MultiLanguageTTSService(self.config)
            self.stt_manager = MultiLanguageSTTService(self.config)

            await self.tts_manager.initialize()
            await self.stt_manager.initialize()

            self._initialized = True
            logger.info("Multi-language audio manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize multi-language audio manager: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the multi-language audio manager."""
        try:
            if self.tts_manager:
                await self.tts_manager.shutdown()
            if self.stt_manager:
                await self.stt_manager.shutdown()

            self._initialized = False
            logger.info("Multi-language audio manager shutdown complete")

        except Exception as e:
            logger.error(f"Error during multi-language audio manager shutdown: {e}")

    async def synthesize_speech(
        self,
        text: str,
        language_code: Optional[LanguageCode] = None,
        voice_style: Optional[str] = None,
        cultural_adaptations: Optional[Dict[str, Any]] = None,
    ) -> MultiLanguageAudioResult:
        """Synthesize speech with language-specific adaptations."""
        if not self._initialized:
            raise RuntimeError("Multi-language audio manager not initialized")

        if self.tts_manager is None:
            raise RuntimeError("TTS manager not initialized")
        return await self.tts_manager.synthesize_speech(
            text=text,
            language_code=language_code,
            voice_style=voice_style,
            cultural_adaptations=cultural_adaptations,
        )

    async def transcribe_audio(
        self,
        audio: AudioData,
        language_code: Optional[LanguageCode] = None,
        auto_detect: bool = True,
    ) -> MultiLanguageAudioResult:
        """Transcribe audio with language detection."""
        if not self._initialized:
            raise RuntimeError("Multi-language audio manager not initialized")

        if self.stt_manager is None:
            raise RuntimeError("STT manager not initialized")
        return await self.stt_manager.transcribe_audio(
            audio=audio, language_code=language_code, auto_detect=auto_detect
        )

    async def get_supported_languages(self) -> List[LanguageCode]:
        """Get list of supported languages."""
        if not self._initialized:
            return []

        # Get intersection of TTS and STT supported languages
        if self.tts_manager is None or self.stt_manager is None:
            return []
        tts_languages = await self.tts_manager.get_supported_languages()
        stt_languages = await self.stt_manager.get_supported_languages()

        # Return languages supported by both
        return list(set(tts_languages) & set(stt_languages))

    async def get_language_capabilities(
        self, language_code: LanguageCode
    ) -> Dict[str, Any]:
        """Get capabilities for a specific language."""
        if not self._initialized:
            return {}

        if self.tts_manager is None or self.stt_manager is None:
            return {}
        tts_capabilities = await self.tts_manager.get_language_capabilities(
            language_code
        )
        stt_capabilities = await self.stt_manager.get_supported_languages()

        return {
            "language_code": language_code.value,
            "tts_supported": language_code
            in await self.tts_manager.get_supported_languages(),
            "stt_supported": language_code in stt_capabilities,
            "tts_capabilities": tts_capabilities,
            "cultural_adaptations": self.localization_manager.get_cultural_adaptations(
                language_code
            ),
            "voice_characteristics": self.localization_manager.get_voice_characteristics(
                language_code
            ),
        }


def get_multilingual_audio_manager(
    config: Optional[Config] = None,
) -> MultiLanguageAudioService:
    """Factory function to create a multi-language audio manager instance."""
    if config is None:
        from ...core.config import Config

        config = Config()
    return MultiLanguageAudioService(config)
