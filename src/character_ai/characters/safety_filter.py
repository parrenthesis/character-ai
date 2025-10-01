from __future__ import annotations

import logging

from ..algorithms.safety.classifier import SafetyClassifier, SafetyLevel

logger = logging.getLogger(__name__)


class ChildSafetyFilter:
    """Enhanced child safety filter with toxicity and PII detection."""

    SAFE_WORDS = {"please", "friend", "share", "kind"}
    BANNED_SUBSTRINGS = {"kill", "hurt", "die", "blood"}

    def __init__(self, enable_classifier: bool = True):
        """Initialize safety filter with optional classifier."""
        self.classifier = SafetyClassifier() if enable_classifier else None
        self.classifier_enabled = enable_classifier

    async def filter_response(self, text: str) -> str:
        """Filter response text for safety concerns."""
        # First, run safety classifier if enabled
        if self.classifier and self.classifier_enabled:
            safety_result = self.classifier.classify(text)

            if safety_result.level == SafetyLevel.UNSAFE:
                logger.warning(f"Unsafe content detected: {safety_result.categories}")
                return "I can't say that. Let's talk about something fun instead!"
            elif safety_result.level == SafetyLevel.WARNING:
                logger.info(f"Warning content detected: {safety_result.categories}")
                # Continue with basic filtering but log the warning

        # Apply basic text filtering
        filtered_text = self._apply_basic_filtering(text)

        # Add positive reinforcement if needed
        if not any(word in filtered_text.lower() for word in self.SAFE_WORDS):
            filtered_text += " Be kind and have fun!"

        return filtered_text

    def _apply_basic_filtering(self, text: str) -> str:
        """Apply basic text filtering rules."""
        lowered = text.lower()
        for banned in self.BANNED_SUBSTRINGS:
            if banned in lowered:
                lowered = lowered.replace(banned, "*")
        return lowered

    def check_safety(self, text: str) -> dict:
        """Check safety of text and return detailed results."""
        if not self.classifier or not self.classifier_enabled:
            return {
                "safe": True,
                "level": "safe",
                "confidence": 1.0,
                "categories": [],
                "processing_time_ms": 0.0,
            }

        return self.classifier.get_safety_summary(text)

    def get_detailed_safety(self, text: str) -> dict:
        """Get detailed safety analysis."""
        if not self.classifier or not self.classifier_enabled:
            return {
                "overall": {
                    "safe": True,
                    "level": "safe",
                    "confidence": 1.0,
                    "categories": [],
                    "processing_time_ms": 0.0,
                },
                "toxicity": {
                    "safe": True,
                    "level": "safe",
                    "confidence": 1.0,
                    "categories": [],
                    "processing_time_ms": 0.0,
                },
                "pii": {
                    "safe": True,
                    "level": "safe",
                    "confidence": 1.0,
                    "categories": [],
                    "processing_time_ms": 0.0,
                },
            }

        results = self.classifier.classify_detailed(text)
        return {
            "overall": {
                "safe": results["overall"].level == SafetyLevel.SAFE,
                "level": results["overall"].level.value,
                "confidence": results["overall"].confidence,
                "categories": results["overall"].categories,
                "processing_time_ms": results["overall"].processing_time_ms,
            },
            "toxicity": {
                "safe": results["toxicity"].level == SafetyLevel.SAFE,
                "level": results["toxicity"].level.value,
                "confidence": results["toxicity"].confidence,
                "categories": results["toxicity"].categories,
                "processing_time_ms": results["toxicity"].processing_time_ms,
            },
            "pii": {
                "safe": results["pii"].level == SafetyLevel.SAFE,
                "level": results["pii"].level.value,
                "confidence": results["pii"].confidence,
                "categories": results["pii"].categories,
                "processing_time_ms": results["pii"].processing_time_ms,
            },
        }
