"""
Multi-language support system for the character.ai.

Provides language detection, localization, and cultural adaptation capabilities.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

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
class LanguagePack:
    """Language pack configuration."""

    language_code: LanguageCode
    cultural_region: CulturalRegion
    display_name: str
    native_name: str

    # Localization data
    safety_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cultural_adaptations: Dict[str, Any] = field(default_factory=dict)
    voice_characteristics: Dict[str, Any] = field(default_factory=dict)

    # Language-specific settings
    text_direction: str = "ltr"  # ltr, rtl
    character_encoding: str = "utf-8"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"

    # Safety and content filtering
    age_appropriate_content: bool = True
    cultural_sensitivity: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LanguageDetectionResult:
    """Result of language detection."""

    detected_language: LanguageCode
    confidence: float
    alternative_languages: List[Tuple[LanguageCode, float]] = field(
        default_factory=list
    )
    processing_time_ms: float = 0.0


class LanguageDetector:
    """Multi-language text detection using pattern matching and heuristics."""

    def __init__(self) -> None:
        self.language_patterns = self._load_language_patterns()
        self.character_sets = self._load_character_sets()

    def _load_language_patterns(self) -> Dict[LanguageCode, List[str]]:
        """Load language-specific patterns for detection."""
        return {
            LanguageCode.ENGLISH: [
                r"\b(the|and|or|but|in|on|at|to|for|of|with|by)\b",
                r"\b(is|are|was|were|be|been|being)\b",
                r"\b(this|that|these|those)\b",
            ],
            LanguageCode.SPANISH: [
                r"\b(el|la|los|las|un|una|de|del|en|con|por|para)\b",
                r"\b(es|son|era|eran|ser|estar)\b",
                r"\b(que|qué|como|cómo|donde|dónde)\b",
            ],
            LanguageCode.FRENCH: [
                r"\b(le|la|les|un|une|de|du|des|en|avec|pour|par)\b",
                r"\b(est|sont|était|étaient|être|avoir)\b",
                r"\b(que|qui|quoi|comment|où)\b",
            ],
            LanguageCode.GERMAN: [
                r"\b(der|die|das|ein|eine|und|oder|mit|von|zu|für)\b",
                r"\b(ist|sind|war|waren|sein|haben)\b",
                r"\b(was|wie|wo|wann|warum)\b",
            ],
            LanguageCode.CHINESE: [
                r"[\u4e00-\u9fff]",  # Chinese characters
                r"\b(的|是|在|有|和|与|或|但|因为|所以)\b",
            ],
            LanguageCode.JAPANESE: [
                r"[\u3040-\u309f]",  # Hiragana
                r"[\u30a0-\u30ff]",  # Katakana
                r"[\u4e00-\u9fff]",  # Kanji
                r"\b(です|である|いる|ある|する|なる)\b",
            ],
            LanguageCode.KOREAN: [
                r"[\uac00-\ud7af]",  # Korean characters
                r"\b(이다|있다|없다|하다|되다|가다|오다)\b",
            ],
            LanguageCode.ARABIC: [
                r"[\u0600-\u06ff]",  # Arabic characters
                r"\b(هذا|هذه|الذي|التي|في|على|من|إلى|مع|لـ)\b",
            ],
        }

    def _load_character_sets(self) -> Dict[LanguageCode, str]:
        """Load character set information for each language."""
        return {
            LanguageCode.ENGLISH: "latin",
            LanguageCode.SPANISH: "latin",
            LanguageCode.FRENCH: "latin",
            LanguageCode.GERMAN: "latin",
            LanguageCode.ITALIAN: "latin",
            LanguageCode.PORTUGUESE: "latin",
            LanguageCode.RUSSIAN: "cyrillic",
            LanguageCode.CHINESE: "cjk",
            LanguageCode.JAPANESE: "cjk",
            LanguageCode.KOREAN: "cjk",
            LanguageCode.ARABIC: "arabic",
            LanguageCode.HINDI: "devanagari",
        }

    def detect_language(self, text: str) -> LanguageDetectionResult:
        """Detect the primary language of the input text."""
        import time

        start_time = time.time()

        if not text or not text.strip():
            return LanguageDetectionResult(
                detected_language=LanguageCode.ENGLISH,
                confidence=0.0,
                processing_time_ms=0.0,
            )

        text_lower = text.lower().strip()
        language_scores = {}

        # Score each language based on pattern matches
        for lang, patterns in self.language_patterns.items():
            score = 0.0
            total_patterns = len(patterns)

            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    score += 1.0

            # Normalize score
            if total_patterns > 0:
                language_scores[lang] = score / total_patterns

        # Character set analysis
        char_set_scores = self._analyze_character_sets(text)
        for lang, score in char_set_scores.items():
            if lang in language_scores:
                language_scores[lang] = (language_scores[lang] + score) / 2
            else:
                language_scores[lang] = score

        # Find the best match
        if not language_scores:
            detected_language = LanguageCode.ENGLISH
            confidence = 0.0
        else:
            detected_language = max(language_scores.items(), key=lambda x: x[1])[0]
            confidence = language_scores[detected_language]

        # Get alternative languages
        alternatives = sorted(
            [
                (lang, score)
                for lang, score in language_scores.items()
                if lang != detected_language
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        processing_time = (time.time() - start_time) * 1000

        return LanguageDetectionResult(
            detected_language=detected_language,
            confidence=confidence,
            alternative_languages=alternatives,
            processing_time_ms=processing_time,
        )

    def _analyze_character_sets(self, text: str) -> Dict[LanguageCode, float]:
        """Analyze character sets to help with language detection."""
        scores = {}

        # Check for specific character sets
        if re.search(r"[\u4e00-\u9fff]", text):  # Chinese characters
            scores[LanguageCode.CHINESE] = 0.8
        if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text):  # Japanese characters
            scores[LanguageCode.JAPANESE] = 0.8
        if re.search(r"[\uac00-\ud7af]", text):  # Korean characters
            scores[LanguageCode.KOREAN] = 0.8
        if re.search(r"[\u0600-\u06ff]", text):  # Arabic characters
            scores[LanguageCode.ARABIC] = 0.8
        if re.search(r"[\u0400-\u04ff]", text):  # Cyrillic characters
            scores[LanguageCode.RUSSIAN] = 0.8

        return scores


class LocalizationManager:
    """Manages language packs and localization."""

    def __init__(self, language_packs_dir: Path = Path.cwd() / "configs/language_packs"):
        self.language_packs_dir = language_packs_dir
        self.language_packs: Dict[LanguageCode, LanguagePack] = {}
        self.detector = LanguageDetector()
        self.current_language = LanguageCode.ENGLISH
        self.fallback_language = LanguageCode.ENGLISH

        # Load available language packs
        self._load_language_packs()

    def _load_language_packs(self) -> None:
        """Load language packs from the configuration directory."""
        if not self.language_packs_dir.exists():
            logger.warning(
                f"Language packs directory not found: {self.language_packs_dir}"
            )
            return

        for pack_file in self.language_packs_dir.glob("*.yaml"):
            try:
                with open(pack_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                language_code = LanguageCode(data["language_code"])
                cultural_region = CulturalRegion(data["cultural_region"])

                pack = LanguagePack(
                    language_code=language_code,
                    cultural_region=cultural_region,
                    display_name=data["display_name"],
                    native_name=data["native_name"],
                    safety_patterns=data.get("safety_patterns", {}),
                    cultural_adaptations=data.get("cultural_adaptations", {}),
                    voice_characteristics=data.get("voice_characteristics", {}),
                    text_direction=data.get("text_direction", "ltr"),
                    character_encoding=data.get("character_encoding", "utf-8"),
                    date_format=data.get("date_format", "%Y-%m-%d"),
                    time_format=data.get("time_format", "%H:%M:%S"),
                    age_appropriate_content=data.get("age_appropriate_content", True),
                    cultural_sensitivity=data.get("cultural_sensitivity", {}),
                )

                self.language_packs[language_code] = pack
                logger.info(
                    f"Loaded language pack: {pack.display_name} ({pack.language_code.value})"
                )

            except Exception as e:
                logger.error(f"Failed to load language pack {pack_file}: {e}")

    def get_language_pack(self, language_code: LanguageCode) -> Optional[LanguagePack]:
        """Get a language pack by language code."""
        return self.language_packs.get(language_code)

    def detect_and_set_language(self, text: str) -> LanguageDetectionResult:
        """Detect language from text and set as current language."""
        result = self.detector.detect_language(text)

        if result.confidence > 0.3:  # Minimum confidence threshold
            self.current_language = result.detected_language
            logger.info(
                f"Language detected: {result.detected_language.value} (confidence: {result.confidence:.2f})"
            )

        return result

    def get_current_language_pack(self) -> Optional[LanguagePack]:
        """Get the current language pack."""
        return self.get_language_pack(self.current_language)

    def get_safety_patterns(
        self, language_code: Optional[LanguageCode] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get safety patterns for a specific language."""
        if language_code is None:
            language_code = self.current_language

        pack = self.get_language_pack(language_code)
        if pack and pack.safety_patterns:
            return pack.safety_patterns

        # Fallback to English patterns
        return self._get_default_safety_patterns()

    def _get_default_safety_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get default English safety patterns."""
        return {
            "violence": {
                "patterns": [
                    r"\b(kill|hurt|harm|violence|fight|attack|destroy)\b",
                    r"\b(weapon|gun|knife|bomb|explode)\b",
                    r"\b(die|death|dead|murder)\b",
                ],
                "severity": "high",
                "weight": 3.0,
            },
            "harassment": {
                "patterns": [
                    r"\b(stupid|idiot|dumb|hate|ugly|fat)\b",
                    r"\b(shut up|shutup|shutup)\b",
                ],
                "severity": "medium",
                "weight": 2.0,
            },
            "inappropriate": {
                "patterns": [
                    r"\b(sex|sexual|naked|nude|private parts)\b",
                    r"\b(drugs|alcohol|smoke|drunk)\b",
                ],
                "severity": "high",
                "weight": 3.0,
            },
        }

    def get_cultural_adaptations(
        self, language_code: Optional[LanguageCode] = None
    ) -> Dict[str, Any]:
        """Get cultural adaptations for a specific language."""
        if language_code is None:
            language_code = self.current_language

        pack = self.get_language_pack(language_code)
        if pack:
            return pack.cultural_adaptations

        return {}

    def get_voice_characteristics(
        self, language_code: Optional[LanguageCode] = None
    ) -> Dict[str, Any]:
        """Get voice characteristics for a specific language."""
        if language_code is None:
            language_code = self.current_language

        pack = self.get_language_pack(language_code)
        if pack:
            return pack.voice_characteristics

        return {}

    def is_rtl_language(self, language_code: Optional[LanguageCode] = None) -> bool:
        """Check if the language is right-to-left."""
        if language_code is None:
            language_code = self.current_language

        pack = self.get_language_pack(language_code)
        if pack:
            return pack.text_direction == "rtl"

        return False

    def get_available_languages(self) -> List[LanguageCode]:
        """Get list of available language codes."""
        return list(self.language_packs.keys())

    def create_language_pack_template(
        self, language_code: LanguageCode, cultural_region: CulturalRegion
    ) -> Dict[str, Any]:
        """Create a template for a new language pack."""
        return {
            "language_code": language_code.value,
            "cultural_region": cultural_region.value,
            "display_name": f"{language_code.name} Language Pack",
            "native_name": f"Native {language_code.name}",
            "safety_patterns": {
                "violence": {"patterns": [], "severity": "high", "weight": 3.0},
                "harassment": {"patterns": [], "severity": "medium", "weight": 2.0},
                "inappropriate": {"patterns": [], "severity": "high", "weight": 3.0},
            },
            "cultural_adaptations": {
                "greeting_style": "formal",
                "conversation_style": "polite",
                "age_appropriate": True,
                "cultural_taboos": [],
                "preferred_topics": [],
            },
            "voice_characteristics": {
                "preferred_pitch": "medium",
                "speaking_rate": "normal",
                "emotional_range": "moderate",
                "formality_level": "neutral",
            },
            "text_direction": "ltr",
            "character_encoding": "utf-8",
            "date_format": "%Y-%m-%d",
            "time_format": "%H:%M:%S",
            "age_appropriate_content": True,
            "cultural_sensitivity": {
                "avoid_topics": [],
                "preferred_approaches": [],
                "cultural_norms": [],
            },
        }


# Global instance
_localization_manager: Optional[LocalizationManager] = None


def get_localization_manager() -> LocalizationManager:
    """Get the global localization manager instance."""
    global _localization_manager
    if _localization_manager is None:
        _localization_manager = LocalizationManager()
    return _localization_manager


def detect_language(text: str) -> LanguageDetectionResult:
    """Convenience function to detect language."""
    return get_localization_manager().detect_and_set_language(text)


def get_current_language() -> LanguageCode:
    """Get the current language."""
    return get_localization_manager().current_language


def set_language(language_code: LanguageCode) -> bool:
    """Set the current language."""
    manager = get_localization_manager()
    if language_code in manager.language_packs:
        manager.current_language = language_code
        return True
    return False
