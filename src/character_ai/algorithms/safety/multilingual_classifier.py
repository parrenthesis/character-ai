"""
Multi-language safety classifier for toxicity and PII detection.

Extends the base safety classifier with language-specific patterns and
cultural adaptations.
"""

import logging
import re
import time
from typing import Any, Dict, Optional

from ...core.language_support import LanguageCode, get_localization_manager
from ..safety.classifier import (
    SafetyClassifier,
    SafetyLevel,
    SafetyResult,
    ToxicityClassifier,
)

logger = logging.getLogger(__name__)


class MultilingualToxicityClassifier(ToxicityClassifier):
    """Multi-language toxicity classifier with cultural adaptations."""

    def __init__(self) -> None:
        super().__init__()
        self.localization_manager = get_localization_manager()
        self.language_specific_patterns: Dict[
            LanguageCode, Dict[str, Dict[str, Any]]
        ] = {}
        self._load_language_specific_patterns()

    def _load_language_specific_patterns(self) -> None:
        """Load language-specific toxicity patterns."""
        # Spanish patterns
        self.language_specific_patterns[LanguageCode.SPANISH] = {
            "violence": {
                "patterns": [
                    r"\b(matar|herir|dañar|violencia|pelear|atacar|destruir)\b",
                    r"\b(arma|pistola|cuchillo|bomba|explotar)\b",
                    r"\b(morir|muerte|muerto|asesinar)\b",
                ],
                "severity": "high",
                "weight": 3.0,
            },
            "harassment": {
                "patterns": [
                    r"\b(estúpido|idiota|tonto|odio|feo|gordo)\b",
                    r"\b(cállate|callate|callate)\b",
                ],
                "severity": "medium",
                "weight": 2.0,
            },
            "inappropriate": {
                "patterns": [
                    r"\b(sexo|sexual|desnudo|desnuda|partes privadas)\b",
                    r"\b(drogas|alcohol|fumar|borracho)\b",
                ],
                "severity": "high",
                "weight": 3.0,
            },
        }

        # French patterns
        self.language_specific_patterns[LanguageCode.FRENCH] = {
            "violence": {
                "patterns": [
                    r"\b(tuer|blesser|nuire|violence|se battre|attaquer|détruire)\b",
                    r"\b(arme|pistolet|couteau|bombe|exploser)\b",
                    r"\b(mourir|mort|mort|assassiner)\b",
                ],
                "severity": "high",
                "weight": 3.0,
            },
            "harassment": {
                "patterns": [
                    r"\b(stupide|idiot|bête|haine|laid|gros)\b",
                    r"\b(tais-toi|tais toi|ferme)\b",
                ],
                "severity": "medium",
                "weight": 2.0,
            },
            "inappropriate": {
                "patterns": [
                    r"\b(sexe|sexuel|nu|nue|parties privées)\b",
                    r"\b(drogues|alcool|fumer|ivre)\b",
                ],
                "severity": "high",
                "weight": 3.0,
            },
        }

        # German patterns
        self.language_specific_patterns[LanguageCode.GERMAN] = {
            "violence": {
                "patterns": [
                    r"\b(töten|verletzen|schaden|gewalt|kämpfen|angreifen|zerstören)\b",

                    r"\b(waffe|pistole|messer|bombe|explodieren)\b",
                    r"\b(sterben|tod|tot|morden)\b",
                ],
                "severity": "high",
                "weight": 3.0,
            },
            "harassment": {
                "patterns": [
                    r"\b(dumm|idiot|dumm|hass|hässlich|fett)\b",
                    r"\b(halte den mund|halt den mund|ruhe)\b",
                ],
                "severity": "medium",
                "weight": 2.0,
            },
            "inappropriate": {
                "patterns": [
                    r"\b(sex|sexuell|nackt|nackt|private teile)\b",
                    r"\b(drogen|alkohol|rauchen|betrunken)\b",
                ],
                "severity": "high",
                "weight": 3.0,
            },
        }

        # Chinese patterns (simplified)
        self.language_specific_patterns[LanguageCode.CHINESE] = {
            "violence": {
                "patterns": [
                    r"\b(杀|伤害|暴力|打架|攻击|破坏)\b",
                    r"\b(武器|枪|刀|炸弹|爆炸)\b",
                    r"\b(死|死亡|死亡|谋杀)\b",
                ],
                "severity": "high",
                "weight": 3.0,
            },
            "harassment": {
                "patterns": [
                    r"\b(愚蠢|白痴|愚蠢|仇恨|丑陋|肥胖)\b",
                    r"\b(闭嘴|闭嘴|安静)\b",
                ],
                "severity": "medium",
                "weight": 2.0,
            },
            "inappropriate": {
                "patterns": [
                    r"\b(性|性|裸体|裸体|私处)\b",
                    r"\b(毒品|酒精|吸烟|醉酒)\b",
                ],
                "severity": "high",
                "weight": 3.0,
            },
        }

        # Japanese patterns
        self.language_specific_patterns[LanguageCode.JAPANESE] = {
            "violence": {
                "patterns": [
                    r"\b(殺す|傷つける|害|暴力|戦う|攻撃|破壊)\b",
                    r"\b(武器|銃|ナイフ|爆弾|爆発)\b",
                    r"\b(死ぬ|死|死|殺人)\b",
                ],
                "severity": "high",
                "weight": 3.0,
            },
            "harassment": {
                "patterns": [
                    r"\b(愚か|バカ|愚か|憎しみ|醜い|太った)\b",
                    r"\b(黙れ|黙れ|静か)\b",
                ],
                "severity": "medium",
                "weight": 2.0,
            },
            "inappropriate": {
                "patterns": [
                    r"\b(性|性的|裸|裸|プライベート部分)\b",
                    r"\b(薬物|アルコール|喫煙|酔っ払い)\b",
                ],
                "severity": "high",
                "weight": 3.0,
            },
        }

        # Korean patterns
        self.language_specific_patterns[LanguageCode.KOREAN] = {
            "violence": {
                "patterns": [
                    r"\b(죽이다|다치게하다|해치다|폭력|싸우다|공격하다|파괴하다)\b",
                    r"\b(무기|총|칼|폭탄|폭발)\b",
                    r"\b(죽다|죽음|죽은|살인)\b",
                ],
                "severity": "high",
                "weight": 3.0,
            },
            "harassment": {
                "patterns": [
                    r"\b(바보|멍청이|바보|증오|못생긴|뚱뚱한)\b",
                    r"\b(닥쳐|닥쳐|조용히)\b",
                ],
                "severity": "medium",
                "weight": 2.0,
            },
            "inappropriate": {
                "patterns": [
                    r"\b(성|성적|벌거벗은|벌거벗은|사생활 부분)\b",
                    r"\b(마약|알코올|흡연|술취한)\b",
                ],
                "severity": "high",
                "weight": 3.0,
            },
        }

        # Arabic patterns
        self.language_specific_patterns[LanguageCode.ARABIC] = {
            "violence": {
                "patterns": [
                    r"\b(قتل|إيذاء|ضرر|عنف|قتال|هجوم|تدمير)\b",
                    r"\b(سلاح|مسدس|سكين|قنبلة|انفجار)\b",
                    r"\b(موت|موت|ميت|قتل)\b",
                ],
                "severity": "high",
                "weight": 3.0,
            },
            "harassment": {
                "patterns": [
                    r"\b(غبي|أحمق|غبي|كراهية|قبيح|سمين)\b",
                    r"\b(اصمت|اصمت|هدوء)\b",
                ],
                "severity": "medium",
                "weight": 2.0,
            },
            "inappropriate": {
                "patterns": [
                    r"\b(جنس|جنسي|عار|عار|أجزاء خاصة)\b",
                    r"\b(مخدرات|كحول|تدخين|سكران)\b",
                ],
                "severity": "high",
                "weight": 3.0,
            },
        }

    def classify_toxicity(
        self, text: str, language_code: Optional[LanguageCode] = None
    ) -> SafetyResult:
        """Classify text for toxicity with language-specific patterns."""
        start_time = time.time()
        text_lower = text.lower()

        # Get language-specific patterns
        if language_code is None:
            language_code = self.localization_manager.current_language

        # Combine base patterns with language-specific patterns
        all_patterns = self.toxicity_patterns.copy()
        if language_code in self.language_specific_patterns:
            lang_patterns = self.language_specific_patterns[language_code]
            for category, config in lang_patterns.items():
                if category in all_patterns:
                    # Merge patterns
                    all_patterns[category]["patterns"].extend(config["patterns"])
                else:
                    all_patterns[category] = config

        detected_categories = []
        total_score = 0.0
        details = {}

        for category, config in all_patterns.items():
            category_score = 0.0
            matches = []

            for pattern in config["patterns"]:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    matches.append(pattern)
                    category_score += config["weight"]

            if matches:
                detected_categories.append(category)
                details[category] = {
                    "matches": matches,
                    "score": category_score,
                    "severity": config["severity"],
                }
                total_score += category_score

        # Determine safety level with cultural sensitivity
        cultural_threshold = self._get_cultural_threshold(language_code)

        if total_score >= cultural_threshold["unsafe"]:
            level = SafetyLevel.UNSAFE
            confidence = min(0.95, total_score / 5.0)
        elif total_score >= cultural_threshold["warning"]:
            level = SafetyLevel.WARNING
            confidence = min(0.8, total_score / 4.0)
        else:
            level = SafetyLevel.SAFE
            confidence = max(0.1, 1.0 - (total_score / 3.0))

        processing_time = (time.time() - start_time) * 1000

        return SafetyResult(
            level=level,
            confidence=confidence,
            categories=detected_categories,
            details=details,
            processing_time_ms=processing_time,
        )

    def _get_cultural_threshold(self, language_code: LanguageCode) -> Dict[str, float]:
        """Get cultural sensitivity thresholds for different languages."""
        # Different cultures may have different sensitivity levels
        thresholds = {
            LanguageCode.ENGLISH: {"unsafe": 3.0, "warning": 1.5},
            LanguageCode.SPANISH: {"unsafe": 2.5, "warning": 1.2},
            LanguageCode.FRENCH: {"unsafe": 2.8, "warning": 1.4},
            LanguageCode.GERMAN: {"unsafe": 3.2, "warning": 1.6},
            LanguageCode.CHINESE: {"unsafe": 2.0, "warning": 1.0},
            LanguageCode.JAPANESE: {"unsafe": 1.8, "warning": 0.9},
            LanguageCode.KOREAN: {"unsafe": 2.2, "warning": 1.1},
            LanguageCode.ARABIC: {"unsafe": 2.0, "warning": 1.0},
        }

        return thresholds.get(language_code, {"unsafe": 3.0, "warning": 1.5})


class MultilingualPIIClassifier:
    """Multi-language PII detection with cultural adaptations."""

    def __init__(self) -> None:
        self.localization_manager = get_localization_manager()
        self.language_specific_patterns: Dict[
            LanguageCode, Dict[str, Dict[str, Any]]
        ] = {}
        self._load_language_specific_pii_patterns()

    def _load_language_specific_pii_patterns(self) -> None:
        """Load language-specific PII patterns."""
        # Spanish PII patterns
        self.language_specific_patterns[LanguageCode.SPANISH] = {
            "phone": {
                "patterns": [
                    r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # US format
                    r"\b\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b",  # Spanish format
                    r"\b\+34\s?\d{2}\s?\d{3}\s?\d{3}\b",  # Spanish international
                ],
                "severity": "high",
                "weight": 3.0,
            },
            "email": {
                "patterns": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
                "severity": "high",
                "weight": 3.0,
            },
            "address": {
                "patterns": [
                    r"\b\d+\s+[A-Za-z\s]+(?:Calle|C\/|Avenida|Av|Carretera|Ctra|Plaza|Pza)\b",
                    r"\b[A-Za-z\s]+,\s*\d{5}\b",  # Spanish postal code
                ],
                "severity": "medium",
                "weight": 2.0,
            },
        }

        # Add more language-specific patterns as needed
        # For now, we'll use the base patterns for other languages

    def classify_pii(
        self, text: str, language_code: Optional[LanguageCode] = None
    ) -> SafetyResult:
        """Classify text for PII with language-specific patterns."""
        start_time = time.time()

        if language_code is None:
            language_code = self.localization_manager.current_language

        # Get language-specific patterns
        all_patterns = self._get_base_pii_patterns()
        if language_code in self.language_specific_patterns:
            lang_patterns = self.language_specific_patterns[language_code]
            for category, config in lang_patterns.items():
                if category in all_patterns:
                    all_patterns[category]["patterns"].extend(config["patterns"])
                else:
                    all_patterns[category] = config

        detected_categories = []
        total_score = 0.0
        details = {}

        for category, config in all_patterns.items():
            category_score = 0.0
            matches = []

            for pattern in config["patterns"]:
                found_matches = re.findall(pattern, text, re.IGNORECASE)
                if found_matches:
                    matches.extend(found_matches)
                    category_score += config["weight"]

            if matches:
                detected_categories.append(category)
                details[category] = {
                    "matches": matches[:3],  # Limit to first 3 matches
                    "count": len(matches),
                    "score": category_score,
                    "severity": config["severity"],
                }
                total_score += category_score

        # Determine safety level
        if total_score >= 3.0:
            level = SafetyLevel.UNSAFE
            confidence = min(0.95, total_score / 5.0)
        elif total_score >= 1.5:
            level = SafetyLevel.WARNING
            confidence = min(0.8, total_score / 4.0)
        else:
            level = SafetyLevel.SAFE
            confidence = max(0.1, 1.0 - (total_score / 3.0))

        processing_time = (time.time() - start_time) * 1000

        return SafetyResult(
            level=level,
            confidence=confidence,
            categories=detected_categories,
            details=details,
            processing_time_ms=processing_time,
        )

    def _get_base_pii_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get base PII patterns."""
        return {
            "phone": {
                "patterns": [
                    r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
                    r"\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b",
                ],
                "severity": "high",
                "weight": 3.0,
            },
            "email": {
                "patterns": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
                "severity": "high",
                "weight": 3.0,
            },
            "address": {
                "patterns": [
                    r"\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)\b",
                    r"\b[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}\b",
                ],
                "severity": "medium",
                "weight": 2.0,
            },
            "ssn": {
                "patterns": [r"\b\d{3}-\d{2}-\d{4}\b", r"\b\d{9}\b"],
                "severity": "high",
                "weight": 3.0,
            },
            "credit_card": {
                "patterns": [r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"],
                "severity": "high",
                "weight": 3.0,
            },
        }


class MultilingualSafetyClassifier(SafetyClassifier):
    """Multi-language safety classifier with cultural adaptations."""

    def __init__(self) -> None:
        super().__init__()
        self.localization_manager = get_localization_manager()
        self.multilingual_toxicity = MultilingualToxicityClassifier()
        self.multilingual_pii = MultilingualPIIClassifier()

    def classify(
        self, text: str, language_code: Optional[LanguageCode] = None
    ) -> SafetyResult:
        """Classify text for safety concerns with language detection."""
        if not self.enabled:
            return SafetyResult(
                level=SafetyLevel.SAFE,
                confidence=1.0,
                categories=[],
                details={"disabled": True},
                processing_time_ms=0.0,
            )

        # Auto-detect language if not provided
        if language_code is None:
            detection_result = self.localization_manager.detect_and_set_language(text)
            language_code = detection_result.detected_language

        # Get toxicity and PII results
        toxicity_result = self.multilingual_toxicity.classify_toxicity(
            text, language_code
        )
        pii_result = self.multilingual_pii.classify_pii(text, language_code)

        # Determine overall safety level
        if (
            toxicity_result.level == SafetyLevel.UNSAFE
            or pii_result.level == SafetyLevel.UNSAFE
        ):
            overall_level = SafetyLevel.UNSAFE
        elif (
            toxicity_result.level == SafetyLevel.WARNING
            or pii_result.level == SafetyLevel.WARNING
        ):
            overall_level = SafetyLevel.WARNING
        else:
            overall_level = SafetyLevel.SAFE

        # Combine confidence scores
        overall_confidence = max(toxicity_result.confidence, pii_result.confidence)

        # Combine categories
        all_categories = toxicity_result.categories + pii_result.categories

        # Combine details
        combined_details = {
            "toxicity": toxicity_result.details,
            "pii": pii_result.details,
            "language": language_code.value,
            "cultural_adaptations": self.localization_manager.get_cultural_adaptations(
                language_code
            ),
        }

        # Total processing time
        total_time = toxicity_result.processing_time_ms + pii_result.processing_time_ms

        return SafetyResult(
            level=overall_level,
            confidence=overall_confidence,
            categories=all_categories,
            details=combined_details,
            processing_time_ms=total_time,
        )

    def classify_detailed(
        self, text: str, language_code: Optional[LanguageCode] = None
    ) -> Dict[str, SafetyResult]:
        """Get detailed classification results with language detection."""
        if not self.enabled:
            return {
                "overall": SafetyResult(
                    level=SafetyLevel.SAFE,
                    confidence=1.0,
                    categories=[],
                    details={"disabled": True},
                    processing_time_ms=0.0,
                ),
                "toxicity": SafetyResult(
                    level=SafetyLevel.SAFE,
                    confidence=1.0,
                    categories=[],
                    details={"disabled": True},
                    processing_time_ms=0.0,
                ),
                "pii": SafetyResult(
                    level=SafetyLevel.SAFE,
                    confidence=1.0,
                    categories=[],
                    details={"disabled": True},
                    processing_time_ms=0.0,
                ),
            }

        # Auto-detect language if not provided
        if language_code is None:
            detection_result = self.localization_manager.detect_and_set_language(text)
            language_code = detection_result.detected_language

        # Get detailed results
        toxicity_result = self.multilingual_toxicity.classify_toxicity(
            text, language_code
        )
        pii_result = self.multilingual_pii.classify_pii(text, language_code)

        # Overall result
        overall_result = self.classify(text, language_code)

        return {
            "overall": overall_result,
            "toxicity": toxicity_result,
            "pii": pii_result,
        }

    def get_safety_summary(
        self, text: str, language_code: Optional[LanguageCode] = None
    ) -> Dict[str, Any]:
        """Get a summary of safety assessment with language information."""
        result = self.classify(text, language_code)

        return {
            "safe": result.level == SafetyLevel.SAFE,
            "level": result.level.value,
            "confidence": result.confidence,
            "categories": result.categories,
            "processing_time_ms": result.processing_time_ms,
            "language": (
                language_code.value
                if language_code
                else self.localization_manager.current_language.value
            ),
            "cultural_adaptations": self.localization_manager.get_cultural_adaptations(
                language_code
            ),
        }
