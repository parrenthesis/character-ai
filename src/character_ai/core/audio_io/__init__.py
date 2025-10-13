"""Audio I/O infrastructure for real-time microphone and speaker interaction.

This module provides interfaces and implementations for:
- Audio device discovery and management
- Real-time audio capture from microphones
- Real-time audio output to speakers
- Mock implementations for testing without hardware
"""

from .audio_config import AudioConfig
from .audio_utils import (
    decode_wav_bytes,
    prepare_audio_for_playback,
    resample_audio,
    validate_and_fix_audio,
)
from .factory import AudioComponentFactory
from .interfaces import AudioCapture, AudioDevice, AudioDeviceManager, AudioOutput
from .mock_audio import FileAudioCapture, FileAudioOutput, MockAudioDeviceManager
from .real_audio import RealAudioCapture, RealAudioDeviceManager, RealAudioOutput

__all__ = [
    # Configuration
    "AudioConfig",
    # Utilities
    "decode_wav_bytes",
    "prepare_audio_for_playback",
    "resample_audio",
    "validate_and_fix_audio",
    # Factory
    "AudioComponentFactory",
    # Interfaces
    "AudioCapture",
    "AudioDevice",
    "AudioDeviceManager",
    "AudioOutput",
    # Real implementations
    "RealAudioCapture",
    "RealAudioDeviceManager",
    "RealAudioOutput",
    # Mock implementations
    "FileAudioCapture",
    "FileAudioOutput",
    "MockAudioDeviceManager",
]
