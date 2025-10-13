"""Audio configuration constants and utilities.

Centralizes audio-related constants and configuration to eliminate
hardcoded values throughout the codebase.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AudioConfig:
    """Audio configuration constants."""

    # Sample rates
    DEFAULT_SAMPLE_RATE: int = 44100
    FALLBACK_SAMPLE_RATES: List[int] = field(default_factory=lambda: [48000, 16000])
    TTS_SAMPLE_RATE: int = 22050
    STT_SAMPLE_RATE: int = 16000

    # Capture settings
    DEFAULT_CHUNK_SIZE: int = 512
    DEFAULT_CHANNELS: int = 1

    # Test paths
    TEST_SAMPLES_BASE: str = "tests_dev/audio_samples"

    @classmethod
    def get_test_input_path(cls, franchise: str, character: str) -> str:
        """Get test input path for a character."""
        return f"{cls.TEST_SAMPLES_BASE}/{franchise}/{character}/input"

    @classmethod
    def get_test_output_path(cls, franchise: str, character: str) -> str:
        """Get test output path for a character."""
        return f"{cls.TEST_SAMPLES_BASE}/{franchise}/{character}/output"
