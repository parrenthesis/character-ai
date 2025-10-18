"""Character-specific response filtering (post-LLM, pre-TextNormalizer)."""

import logging
import os
import re
from typing import Any, Dict, List, Optional

from ...algorithms.conversational_ai.session_memory import ConversationTurn
from ..management.types import Character

logger = logging.getLogger(__name__)


class CharacterResponseFilter:
    """
    Applies character-specific post-processing to LLM responses.

    Separate from TextNormalizer (which handles generic cleanup) to keep
    character personality logic data-driven and modular.
    """

    def __init__(
        self,
        character: Character,
        franchise: Optional[str] = None,
    ):
        self.character = character
        self.franchise = franchise or getattr(
            character, "franchise", character.name.lower()
        )
        self.filter_config = self._load_filter_config()
        self.recent_responses: List[str] = []  # Track for repetition prevention

    def _load_filter_config(self) -> Dict[str, Any]:
        """Load character-specific filter rules from filters.yaml"""
        franchise = self.franchise if self.franchise else self.character.name.lower()
        possible_paths = [
            os.path.join(
                "configs",
                "characters",
                franchise,
                self.character.name.lower(),
                "filters.yaml",
            ),
            os.path.join("configs", "characters", franchise, "data", "filters.yaml"),
            os.path.join("configs", "templates", "filters", "generic_filters.yaml"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    from pathlib import Path

                    from ...core.config.yaml_loader import YAMLConfigLoader

                    config = YAMLConfigLoader.load_yaml(Path(path))
                    logger.info(f"Loaded character filters from {path}")
                    return config
                except Exception as e:
                    logger.warning(f"Failed to load filters from {path}: {e}")
                    continue

        # Return default config if no file found
        logger.info("No character-specific filters found, using defaults")
        return {"remove_phrases": [], "simplify_patterns": {}, "max_words": 30}

    def filter_response(
        self,
        response: str,
        conversation_history: Optional[List[ConversationTurn]] = None,
    ) -> str:
        """Apply character-specific filtering to response."""
        if not response:
            return ""

        filtered = response.strip()

        # Step 1: Remove character-specific phrases
        filtered = self._remove_unwanted_phrases(filtered)

        # Step 2: Apply simplification patterns
        filtered = self._apply_simplification_patterns(filtered)

        # Step 3: Prevent repetitive phrases across conversation
        if conversation_history:
            filtered = self._prevent_repetition(filtered, conversation_history)

        # Step 4: Enforce character-specific word limit
        filtered = self._enforce_word_limit(filtered)

        # Track this response for future repetition checking
        self.recent_responses.append(filtered)
        if len(self.recent_responses) > 5:  # Keep last 5 responses
            self.recent_responses.pop(0)

        return filtered

    def _remove_unwanted_phrases(self, text: str) -> str:
        """Remove phrases that don't fit character personality."""
        phrases = self.filter_config.get("remove_phrases", [])
        for phrase in phrases:
            text = re.sub(phrase, "", text, flags=re.IGNORECASE)

        # Clean up fragments and orphaned punctuation left by removal
        # Remove patterns like "i am ." or "i am , " (orphaned sentence starters)
        text = re.sub(
            r"\b(i am|i\'m|i have|i\'ve|i will|i\'ll)\s*[.,;:!?]\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Remove double punctuation (e.g., ". ," or ", .")
        text = re.sub(r"[.,;:]\s*[.,;:]", ".", text)

        # Remove spaces before punctuation
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)

        # Clean up extra whitespace left by removal
        text = re.sub(r"\s+", " ", text).strip()

        # Capitalize first letter if it's lowercase after cleanup
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        return text

    def _apply_simplification_patterns(self, text: str) -> str:
        """Apply regex patterns to simplify overly complex responses."""
        patterns = self.filter_config.get("simplify_patterns", {})
        for pattern, replacement in patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Additional cleanup after pattern simplification
        # Remove any remaining orphaned fragments
        text = re.sub(r"\b(i am|i\'m)\s+\.", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\.\s*\.", ".", text)  # Remove double periods

        return text.strip()

    def _prevent_repetition(
        self, text: str, conversation_history: List[ConversationTurn]
    ) -> str:
        """Prevent repetitive phrases that appeared recently in conversation."""
        repetition_config = self.filter_config.get("prevent_repetition", {})
        if not repetition_config.get("enabled", False):
            return text

        window_turns = repetition_config.get("window_turns", 3)
        phrases_to_track = repetition_config.get("phrases_to_track", [])

        # Get recent character responses
        recent_turns = (
            conversation_history[-window_turns:]
            if len(conversation_history) > window_turns
            else conversation_history
        )
        recent_character_text = " ".join(
            turn.character_response.lower() for turn in recent_turns
        )

        # Check if tracked phrases appear too frequently
        text_lower = text.lower()
        for phrase in phrases_to_track:
            phrase_lower = phrase.lower()
            # If phrase appears in recent history AND current response, try to vary
            if phrase_lower in recent_character_text and phrase_lower in text_lower:
                # Simple variation: if "fascinating" used recently, try to remove it
                # More sophisticated: could use synonyms, but that requires LLM
                logger.debug(f"Detected repetitive phrase '{phrase}' - keeping for now")
                # TODO: Could implement synonym replacement or prompt LLM for variation

        return text

    def _enforce_word_limit(self, text: str) -> str:
        """Truncate response to character-specific word limit."""
        max_words = self.filter_config.get("max_words", 30)
        words = text.split()

        if len(words) <= max_words:
            return text

        # Truncate at last complete sentence within limit
        truncated = " ".join(words[:max_words])
        last_period = max(
            truncated.rfind("."),
            truncated.rfind("!"),
            truncated.rfind("?"),
        )

        if last_period > 0:
            result = truncated[: last_period + 1]
        else:
            # No sentence ending found, add period
            result = truncated + "."

        # Final cleanup: remove any trailing fragments
        result = re.sub(r"\s+[.,;:!?]+$", ".", result)

        return result
