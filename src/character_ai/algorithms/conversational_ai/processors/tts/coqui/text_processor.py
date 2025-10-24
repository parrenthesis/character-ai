"""
Text processing utilities for Coqui TTS.
"""

import re
from typing import List


class CoquiTextProcessor:
    """Text processing utilities for Coqui TTS."""

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """
        Split text into sentences for better TTS processing.

        Args:
            text: Input text to split

        Returns:
            List of sentences
        """
        if not text or not text.strip():
            return []

        # Clean up the text
        text = text.strip()

        # Handle common abbreviations that shouldn't end sentences
        abbreviations = [
            r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|Inc|Ltd|Corp|Co)\.",
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.",
            r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\.",
            r"\b(?:St|Ave|Rd|Blvd|Pkwy)\.",
        ]

        # Replace abbreviations with placeholders
        placeholder_map: dict[str, str] = {}
        for i, pattern in enumerate(abbreviations):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                placeholder = f"__ABBREV_{i}_{len(placeholder_map)}__"
                placeholder_map[placeholder] = match.group()
                text = text.replace(match.group(), placeholder, 1)

        # Split on sentence endings
        sentences = re.split(r"[.!?]+", text)

        # Restore abbreviations
        for placeholder, original in placeholder_map.items():
            for i, sentence in enumerate(sentences):
                sentences[i] = sentence.replace(placeholder, original)

        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        # If no sentences found, return the original text
        if not sentences:
            return [text]

        return sentences
