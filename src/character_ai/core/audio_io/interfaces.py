"""Audio I/O interfaces for dependency injection and testing.

These interfaces allow swapping between real hardware and mock implementations
without changing the calling code.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional, Protocol

import numpy as np


class AudioDevice(Protocol):
    """Represents an audio input or output device."""

    name: str
    index: int
    channels: int
    sample_rate: int
    is_input: bool
    is_output: bool


class AudioDeviceManager(ABC):
    """Manages audio device discovery and selection."""

    @abstractmethod
    async def list_input_devices(self) -> List[AudioDevice]:
        """List all available input devices (microphones)."""
        pass

    @abstractmethod
    async def list_output_devices(self) -> List[AudioDevice]:
        """List all available output devices (speakers)."""
        pass

    @abstractmethod
    async def get_default_input_device(self) -> Optional[AudioDevice]:
        """Get the default input device."""
        pass

    @abstractmethod
    async def get_default_output_device(self) -> Optional[AudioDevice]:
        """Get the default output device."""
        pass

    @abstractmethod
    async def get_device_by_name(self, name: str) -> Optional[AudioDevice]:
        """Get a device by name."""
        pass


class AudioCapture(ABC):
    """Captures audio from an input device (microphone)."""

    @abstractmethod
    async def start_capture(
        self,
        device: AudioDevice,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
    ) -> None:
        """Start capturing audio from the device."""
        pass

    @abstractmethod
    async def stop_capture(self) -> None:
        """Stop capturing audio."""
        pass

    @abstractmethod
    async def read_audio_chunk(self) -> Optional[np.ndarray]:
        """Read the next chunk of audio data."""
        pass

    @abstractmethod
    async def capture_stream(
        self, sample_rate: int = 22050, channels: int = 1
    ) -> AsyncGenerator[np.ndarray, None]:
        """Stream audio chunks as they become available."""
        pass

    @abstractmethod
    def is_capturing(self) -> bool:
        """Check if currently capturing audio."""
        pass


class AudioOutput(ABC):
    """Plays audio to an output device (speaker)."""

    @abstractmethod
    async def start_playback(
        self, device: AudioDevice, sample_rate: int = 22050, channels: int = 1
    ) -> None:
        """Start audio playback to the device."""
        pass

    @abstractmethod
    async def stop_playback(self) -> None:
        """Stop audio playback."""
        pass

    @abstractmethod
    async def write_audio_chunk(self, audio_data: np.ndarray) -> None:
        """Write a chunk of audio data for playback."""
        pass

    @abstractmethod
    async def play_audio_data(self, audio_data: np.ndarray) -> None:
        """Play complete audio data."""
        pass

    @abstractmethod
    def is_playing(self) -> bool:
        """Check if currently playing audio."""
        pass
