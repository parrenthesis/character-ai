"""
Multi-language support system for character AI.

Provides language detection, localization, and cultural adaptation capabilities.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LanguageCode(Enum):
    """Supported language codes (ISO 639-1)."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"


class CulturalRegion(Enum):
    """Cultural regions for adaptation."""

    NORTH_AMERICA = "na"
    EUROPE = "eu"
    ASIA = "as"
    LATIN_AMERICA = "la"
    MIDDLE_EAST = "me"
    AFRICA = "af"


@dataclass
class LanguageDetectionResult:
    """Result of language detection."""

    detected_language: LanguageCode
    confidence: float
    text: str


@dataclass
class LanguagePack:
    """Language pack configuration."""

    language_code: LanguageCode
    cultural_region: CulturalRegion
    display_name: str
    native_name: str
    greeting_patterns: List[str] = field(default_factory=list)
    farewell_patterns: List[str] = field(default_factory=list)
    formality_levels: Dict[str, str] = field(default_factory=dict)
    cultural_adaptations: Dict[str, Any] = field(default_factory=dict)
    voice_characteristics: Dict[str, Any] = field(default_factory=dict)
    safety_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "language_code": self.language_code.value,
            "cultural_region": self.cultural_region.value,
            "display_name": self.display_name,
            "native_name": self.native_name,
            "greeting_patterns": self.greeting_patterns,
            "farewell_patterns": self.farewell_patterns,
            "formality_levels": self.formality_levels,
            "cultural_adaptations": self.cultural_adaptations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LanguagePack":
        """Create from dictionary."""
        return cls(
            language_code=LanguageCode(data["language_code"]),
            cultural_region=CulturalRegion(data["cultural_region"]),
            display_name=data["display_name"],
            native_name=data["native_name"],
            greeting_patterns=data.get("greeting_patterns", []),
            farewell_patterns=data.get("farewell_patterns", []),
            formality_levels=data.get("formality_levels", {}),
            cultural_adaptations=data.get("cultural_adaptations", {}),
        )


class LocalizationService:
    """Manages language detection, localization, and cultural adaptation."""

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        """Initialize localization manager."""
        self.config_dir = config_dir or Path.cwd() / "configs" / "language_packs"
        self.language_packs: Dict[LanguageCode, LanguagePack] = {}
        self._load_language_packs()

    def _load_language_packs(self) -> None:
        """Load language packs from configuration files."""
        try:
            if self.config_dir.exists():
                for yaml_file in self.config_dir.glob("*.yaml"):
                    try:
                        from ...core.config.yaml_loader import YAMLConfigLoader

                        data = YAMLConfigLoader.load_yaml(yaml_file)
                        if data and "language_code" in data:
                            language_pack = LanguagePack.from_dict(data)
                            self.language_packs[
                                language_pack.language_code
                            ] = language_pack
                    except Exception as e:
                        logger.warning(f"Error loading language pack {yaml_file}: {e}")

            # Add default English pack if none loaded
            if LanguageCode.ENGLISH not in self.language_packs:
                self.language_packs[LanguageCode.ENGLISH] = LanguagePack(
                    language_code=LanguageCode.ENGLISH,
                    cultural_region=CulturalRegion.NORTH_AMERICA,
                    display_name="English",
                    native_name="English",
                    greeting_patterns=[
                        "hello",
                        "hi",
                        "hey",
                        "good morning",
                        "good afternoon",
                    ],
                    farewell_patterns=["goodbye", "bye", "see you later", "take care"],
                    formality_levels={
                        "formal": "formal",
                        "casual": "casual",
                        "friendly": "friendly",
                    },
                    safety_patterns=[
                        "violence",
                        "hate",
                        "discrimination",
                        "harassment",
                        "threats",
                        "inappropriate",
                        "explicit",
                        "dangerous",
                    ],
                    voice_characteristics={
                        "pitch_range": "medium",
                        "speech_rate": "normal",
                        "emphasis_style": "moderate",
                    },
                )

            logger.info(f"Loaded {len(self.language_packs)} language packs")
        except Exception as e:
            logger.error(f"Error loading language packs: {e}")

    def detect_language(self, text: str) -> Tuple[LanguageCode, float]:
        """Detect language from text."""
        if not text or not text.strip():
            return LanguageCode.ENGLISH, 0.0

        # Simple keyword-based detection
        text_lower = text.lower()

        # English patterns - more comprehensive
        english_patterns = [
            "the",
            "and",
            "is",
            "are",
            "was",
            "were",
            "have",
            "has",
            "had",
            "hello",
            "how",
            "you",
            "are",
            "what",
            "where",
            "when",
            "why",
            "who",
        ]
        english_score = sum(1 for pattern in english_patterns if pattern in text_lower)

        # Spanish patterns
        spanish_patterns = [
            "el",
            "la",
            "de",
            "que",
            "en",
            "un",
            "es",
            "se",
            "no",
            "hola",
            "como",
            "estas",
            "donde",
            "cuando",
            "por",
            "que",
            "con",
            "para",
            "por",
            "muy",
            "mas",
            "todo",
            "esta",
            "esta",
            "pero",
            "como",
            "bien",
            "hacer",
            "tiempo",
            "vida",
            "casa",
            "mundo",
            "hombre",
            "mujer",
            "niño",
            "niña",
            "familia",
            "trabajo",
            "amigo",
            "amiga",
            "gracias",
            "por favor",
            "lo siento",
            "buenos días",
            "buenas tardes",
            "buenas noches",
        ]
        spanish_score = sum(1 for pattern in spanish_patterns if pattern in text_lower)

        # French patterns
        french_patterns = [
            "le",
            "la",
            "de",
            "et",
            "à",
            "un",
            "une",
            "est",
            "que",
            "ne",
            "bonjour",
            "comment",
            "vous",
            "allez",
            "quoi",
            "où",
            "quand",
            "pourquoi",
        ]
        french_score = sum(1 for pattern in french_patterns if pattern in text_lower)

        # German patterns
        german_patterns = [
            "der",
            "die",
            "das",
            "und",
            "ist",
            "in",
            "zu",
            "den",
            "von",
            "mit",
            "hallo",
            "wie",
            "geht",
            "es",
            "was",
            "wo",
            "wann",
            "warum",
        ]
        german_score = sum(1 for pattern in german_patterns if pattern in text_lower)

        # Chinese patterns (check for Chinese characters)
        chinese_score = 0
        for char in text:
            if "\u4e00" <= char <= "\u9fff":  # Chinese character range
                chinese_score += 1

        scores = {
            LanguageCode.ENGLISH: english_score,
            LanguageCode.SPANISH: spanish_score,
            LanguageCode.FRENCH: french_score,
            LanguageCode.GERMAN: german_score,
            LanguageCode.CHINESE: chinese_score,
        }

        # If no patterns found, default to English
        if not any(scores.values()):
            return LanguageCode.ENGLISH, 0.1

        detected_language = max(scores.items(), key=lambda x: x[1])
        confidence = min(detected_language[1] / len(text.split()) * 10, 1.0)

        return detected_language[0], confidence

    def get_language_pack(self, language_code: LanguageCode) -> Optional[LanguagePack]:
        """Get language pack for language code."""
        return self.language_packs.get(language_code)

    def get_supported_languages(self) -> List[LanguageCode]:
        """Get list of supported languages."""
        return list(self.language_packs.keys())

    def adapt_text_for_culture(
        self,
        text: str,
        language_code: LanguageCode,
        cultural_context: Optional[str] = None,
    ) -> str:
        """Adapt text for cultural context."""
        language_pack = self.get_language_pack(language_code)
        if not language_pack:
            return text

        # Simple cultural adaptations
        adapted_text = text

        # Apply cultural adaptations if available
        if cultural_context and cultural_context in language_pack.cultural_adaptations:
            adaptations = language_pack.cultural_adaptations[cultural_context]
            for original, adapted in adaptations.items():
                adapted_text = adapted_text.replace(original, adapted)

        return adapted_text

    def get_greeting(
        self, language_code: LanguageCode, formality: str = "casual"
    ) -> str:
        """Get appropriate greeting for language and formality."""
        language_pack = self.get_language_pack(language_code)
        if not language_pack or not language_pack.greeting_patterns:
            return "Hello"

        # Simple formality mapping
        if formality == "formal" and "formal" in language_pack.formality_levels:
            # Use more formal greeting
            return (
                language_pack.greeting_patterns[0]
                if language_pack.greeting_patterns
                else "Hello"
            )
        else:
            # Use casual greeting
            return (
                language_pack.greeting_patterns[-1]
                if language_pack.greeting_patterns
                else "Hello"
            )

    def get_farewell(
        self, language_code: LanguageCode, formality: str = "casual"
    ) -> str:
        """Get appropriate farewell for language and formality."""
        language_pack = self.get_language_pack(language_code)
        if not language_pack or not language_pack.farewell_patterns:
            return "Goodbye"

        # Simple formality mapping
        if formality == "formal" and "formal" in language_pack.formality_levels:
            # Use more formal farewell
            return (
                language_pack.farewell_patterns[0]
                if language_pack.farewell_patterns
                else "Goodbye"
            )
        else:
            # Use casual farewell
            return (
                language_pack.farewell_patterns[-1]
                if language_pack.farewell_patterns
                else "Goodbye"
            )

    def detect_and_set_language(self, text: str) -> "LanguageDetectionResult":
        """Detect language and set as current language."""
        detected_language, confidence = self.detect_language(text)
        self.current_language = detected_language
        return LanguageDetectionResult(
            detected_language=detected_language, confidence=confidence, text=text
        )

    def get_safety_patterns(self, language_code: LanguageCode) -> List[str]:
        """Get safety patterns for language."""
        language_pack = self.get_language_pack(language_code)
        if not language_pack:
            return []
        return getattr(language_pack, "safety_patterns", [])

    def get_cultural_adaptations(self, language_code: LanguageCode) -> Dict[str, Any]:
        """Get cultural adaptations for language."""
        language_pack = self.get_language_pack(language_code)
        if not language_pack:
            return {}
        return getattr(language_pack, "cultural_adaptations", {})

    def get_voice_characteristics(self, language_code: LanguageCode) -> Dict[str, Any]:
        """Get voice characteristics for language."""
        language_pack = self.get_language_pack(language_code)
        if not language_pack:
            return {}
        return getattr(language_pack, "voice_characteristics", {})

    def is_rtl_language(self, language_code: LanguageCode) -> bool:
        """Check if language is right-to-left."""
        rtl_languages = {LanguageCode.ARABIC, LanguageCode.HINDI}
        return language_code in rtl_languages

    def get_available_languages(self) -> List[LanguageCode]:
        """Get list of available languages."""
        return self.get_supported_languages()

    def create_language_pack_template(
        self, language_code: LanguageCode
    ) -> LanguagePack:
        """Create a template language pack."""
        # Map language codes to display names
        display_names = {
            LanguageCode.ENGLISH: "ENGLISH",
            LanguageCode.SPANISH: "SPANISH",
            LanguageCode.FRENCH: "FRENCH",
            LanguageCode.GERMAN: "GERMAN",
            LanguageCode.CHINESE: "CHINESE",
            LanguageCode.JAPANESE: "JAPANESE",
            LanguageCode.KOREAN: "KOREAN",
            LanguageCode.ARABIC: "ARABIC",
        }

        return LanguagePack(
            language_code=language_code,
            cultural_region=CulturalRegion.NORTH_AMERICA,
            display_name=display_names.get(language_code, language_code.value.upper()),
            native_name=display_names.get(language_code, language_code.value.upper()),
            greeting_patterns=["hello"],
            farewell_patterns=["goodbye"],
            formality_levels={"casual": "casual"},
            safety_patterns=[],
            voice_characteristics={},
        )

    @property
    def current_language(self) -> LanguageCode:
        """Get current language."""
        return getattr(self, "_current_language", LanguageCode.ENGLISH)

    @current_language.setter
    def current_language(self, language_code: LanguageCode) -> None:
        """Set current language."""
        self._current_language = language_code

    def set_language(self, language_code: LanguageCode) -> bool:
        """Set the current language."""
        if language_code in self.language_packs:
            self.current_language = language_code
            return True
        return False


def create_localization_manager() -> LocalizationService:
    """Create a new localization manager instance."""
    return LocalizationService()
