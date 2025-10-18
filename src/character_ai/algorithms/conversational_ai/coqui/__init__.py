"""
Coqui TTS processor for text-to-speech synthesis and voice cloning.

This module provides high-quality text-to-speech conversion with
native voice cloning capabilities using Coqui TTS.
"""

from .config import CoquiConfig
from .text_processor import CoquiTextProcessor
from .tts_processor import CoquiTTSProcessor
from .voice_cloner import CoquiVoiceCloner

__all__ = [
    "CoquiConfig",
    "CoquiTextProcessor",
    "CoquiTTSProcessor",
    "CoquiVoiceCloner",
]
