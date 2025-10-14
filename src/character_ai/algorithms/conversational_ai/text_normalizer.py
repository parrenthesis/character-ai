"""Text normalization utilities for LLM and TTS processing."""

import re
from typing import Optional

from ...core.config import Config


class TextNormalizer:
    """Centralized text normalization for LLM and TTS processing."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def clean_llm_response(self, response: str) -> str:
        """Clean LLM response - consolidates logic from real_time_engine.py and providers.py"""
        if not response:
            return ""

        # Start with the response
        cleaned = response.strip()

        # Remove role labels and stage directions (from providers.py)
        cleaned = re.sub(
            r"^(user|assistant|data|character):\s*", "", cleaned, flags=re.IGNORECASE
        )
        cleaned = re.sub(
            r"\[.*?\]", "", cleaned
        )  # Remove stage directions like [smiles]
        cleaned = re.sub(
            r"\(.*?\)", "", cleaned
        )  # Remove parentheticals like (chuckles)

        # Remove TTS artifacts and malformed output (from real_time_engine.py)
        # Remove common TTS artifacts
        tts_artifacts = [
            r"<speak>.*?</speak>",
            r"<voice.*?>.*?</voice>",
            r"<prosody.*?>.*?</prosody>",
            r"<break.*?/>",
            r"<mark.*?/>",
            r"<audio.*?/>",
            r"<emphasis.*?>.*?</emphasis>",
            r"<sub.*?>.*?</sub>",
            r"<say-as.*?>.*?</say-as>",
            r"<phoneme.*?>.*?</phoneme>",
            r"<lang.*?>.*?</lang>",
            r"<p>.*?</p>",
            r"<s>.*?</s>",
            r"<w>.*?</w>",
            r"<mstts:express-as.*?>.*?</mstts:express-as>",
            r"<mstts:silence.*?/>",
            r"<mstts:backgroundaudio.*?/>",
            r"<mstts:viseme.*?/>",
            r"<mstts:bookmark.*?/>",
            r"<mstts:param.*?/>",
            r"<mstts:express-as.*?/>",
            r"<mstts:silence.*?>.*?</mstts:silence>",
            r"<mstts:backgroundaudio.*?>.*?</mstts:backgroundaudio>",
            r"<mstts:viseme.*?>.*?</mstts:viseme>",
            r"<mstts:bookmark.*?>.*?</mstts:bookmark>",
            r"<mstts:param.*?>.*?</mstts:param>",
        ]

        for pattern in tts_artifacts:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

        # Remove HTML/XML tags
        cleaned = re.sub(r"<[^>]+>", "", cleaned)

        # Remove markdown formatting
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)  # Bold
        cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)  # Italic
        cleaned = re.sub(r"`(.*?)`", r"\1", cleaned)  # Code
        cleaned = re.sub(r"#+\s*", "", cleaned)  # Headers
        cleaned = re.sub(r"^\s*[-*+]\s*", "", cleaned, flags=re.MULTILINE)  # Lists

        # Remove excessive whitespace and normalize
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip()

        # Remove empty lines and keep only the first non-empty line (single-line enforcement)
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        if lines:
            cleaned = lines[0]

        # NOTE: Character-specific phrase filtering moved to CharacterResponseFilter
        # This keeps TextNormalizer generic and reusable across all characters

        # Truncate if response is too long (>25 words for more natural brevity)
        words = cleaned.split()
        if len(words) > 25:
            # Try to find the last complete sentence within 25 words
            truncated = " ".join(words[:25])
            # Find last sentence-ending punctuation
            last_period = max(
                truncated.rfind("."),
                truncated.rfind("!"),
                truncated.rfind("?"),
            )
            if last_period > 0:
                cleaned = truncated[: last_period + 1]
            else:
                # No sentence ending found, just truncate and add period
                cleaned = truncated + "."

        # Final cleanup
        cleaned = cleaned.strip()

        return cleaned

    def prepare_for_tts(self, text: str) -> str:
        """Prepare text for TTS synthesis - from coqui_processor.py lines 148-160"""
        if not text:
            return ""

        # Prevent TTS sentence splitting by replacing periods with commas
        # This prevents the TTS from creating multiple audio chunks
        tts_text = text.replace(".", ",")

        # Remove sentence-ending punctuation that causes multi-chunk synthesis
        tts_text = re.sub(r"[.!?]+$", "", tts_text)

        # Ensure the text ends with appropriate punctuation for TTS
        if tts_text and not tts_text.endswith((",", ";", ":", "!", "?")):
            tts_text += "."

        return tts_text.strip()
