"""
Lightweight safety classifier for toxicity and PII detection.

Provides on-device content moderation with minimal dependencies and fast inference.
"""

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety classification levels."""

    SAFE = "safe"
    WARNING = "warning"
    UNSAFE = "unsafe"


@dataclass
class SafetyResult:
    """Result of safety classification."""

    level: SafetyLevel
    confidence: float
    categories: List[str]
    details: Dict[str, Any]
    processing_time_ms: float


class ToxicityClassifier:
    """Lightweight toxicity classifier using rule-based and pattern matching."""

    def __init__(self) -> None:
        self.toxicity_patterns = self._load_toxicity_patterns()
        self.pii_patterns = self._load_pii_patterns()
        self.severity_weights = {"high": 3.0, "medium": 2.0, "low": 1.0}

    def _load_toxicity_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load toxicity detection patterns."""
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
            "threats": {
                "patterns": [
                    r"\b(threat|threaten|revenge|get you|come after)\b",
                    r"\b(payback|consequences|you'll see)\b",
                ],
                "severity": "high",
                "weight": 3.0,
            },
        }

    def _load_pii_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load PII detection patterns."""
        return {
            "phone": {
                "patterns": [
                    r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # US phone numbers
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
                    r"\b[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}\b",  # City, State ZIP
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

    def classify_toxicity(self, text: str) -> SafetyResult:
        """Classify text for toxicity."""
        start_time = time.time()
        text_lower = text.lower()

        detected_categories = []
        total_score = 0.0
        details = {}

        for category, config in self.toxicity_patterns.items():
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

        # Determine safety level (more sensitive thresholds)
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

    def classify_pii(self, text: str) -> SafetyResult:
        """Classify text for PII detection."""
        start_time = time.time()

        detected_categories = []
        total_score = 0.0
        details = {}

        for category, config in self.pii_patterns.items():
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

        # Determine safety level (more sensitive thresholds)
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

    def classify_content(self, text: str) -> Dict[str, SafetyResult]:
        """Classify content for both toxicity and PII."""
        toxicity_result = self.classify_toxicity(text)
        pii_result = self.classify_pii(text)

        return {"toxicity": toxicity_result, "pii": pii_result}

    def get_overall_safety(self, text: str) -> SafetyResult:
        """Get overall safety assessment."""
        results = self.classify_content(text)

        # Use the most severe result
        toxicity_level = results["toxicity"].level
        pii_level = results["pii"].level

        # Determine overall level
        if toxicity_level == SafetyLevel.UNSAFE or pii_level == SafetyLevel.UNSAFE:
            overall_level = SafetyLevel.UNSAFE
        elif toxicity_level == SafetyLevel.WARNING or pii_level == SafetyLevel.WARNING:
            overall_level = SafetyLevel.WARNING
        else:
            overall_level = SafetyLevel.SAFE

        # Combine confidence scores
        overall_confidence = max(
            results["toxicity"].confidence, results["pii"].confidence
        )

        # Combine categories
        all_categories = results["toxicity"].categories + results["pii"].categories

        # Combine details
        combined_details = {
            "toxicity": results["toxicity"].details,
            "pii": results["pii"].details,
        }

        # Total processing time
        total_time = (
            results["toxicity"].processing_time_ms + results["pii"].processing_time_ms
        )

        return SafetyResult(
            level=overall_level,
            confidence=overall_confidence,
            categories=all_categories,
            details=combined_details,
            processing_time_ms=total_time,
        )


class SafetyClassifier:
    """Main safety classifier interface."""

    def __init__(self) -> None:
        self.toxicity_classifier = ToxicityClassifier()
        self.enabled = True

    def classify(self, text: str) -> SafetyResult:
        """Classify text for safety concerns."""
        if not self.enabled:
            return SafetyResult(
                level=SafetyLevel.SAFE,
                confidence=1.0,
                categories=[],
                details={"disabled": True},
                processing_time_ms=0.0,
            )

        return self.toxicity_classifier.get_overall_safety(text)

    def classify_detailed(self, text: str) -> Dict[str, SafetyResult]:
        """Get detailed classification results."""
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

        results = self.toxicity_classifier.classify_content(text)
        results["overall"] = self.toxicity_classifier.get_overall_safety(text)
        return results

    def is_safe(self, text: str) -> bool:
        """Quick safety check - returns True if content is safe."""
        result = self.classify(text)
        return result.level == SafetyLevel.SAFE

    def get_safety_summary(self, text: str) -> Dict[str, Any]:
        """Get a summary of safety assessment."""
        result = self.classify(text)

        return {
            "safe": result.level == SafetyLevel.SAFE,
            "level": result.level.value,
            "confidence": result.confidence,
            "categories": result.categories,
            "processing_time_ms": result.processing_time_ms,
        }
