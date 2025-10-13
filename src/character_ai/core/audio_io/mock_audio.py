"""Mock audio implementations for testing without hardware.

These implementations use files instead of real audio devices,
allowing the voice pipeline to be tested without microphones or speakers.
"""

import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import numpy as np
import soundfile as sf

from .interfaces import AudioCapture, AudioDevice, AudioDeviceManager, AudioOutput

logger = logging.getLogger(__name__)


class MockAudioDevice:
    """Mock audio device for testing."""

    def __init__(
        self, name: str, index: int, is_input: bool = True, is_output: bool = False
    ):
        self.name = name
        self.index = index
        self.channels = 1
        self.sample_rate = 16000 if is_input else 22050
        self.is_input = is_input
        self.is_output = is_output


class MockAudioDeviceManager(AudioDeviceManager):
    """Mock device manager that provides fake devices."""

    async def list_input_devices(self) -> List[AudioDevice]:
        """Return mock input devices."""
        return [
            MockAudioDevice("Mock Microphone", 0, is_input=True),
            MockAudioDevice("Mock USB Mic", 1, is_input=True),
        ]

    async def list_output_devices(self) -> List[AudioDevice]:
        """Return mock output devices."""
        return [
            MockAudioDevice("Mock Speaker", 0, is_input=False, is_output=True),
            MockAudioDevice("Mock Headphones", 1, is_input=False, is_output=True),
        ]

    async def get_default_input_device(self) -> Optional[AudioDevice]:
        """Return default mock input device."""
        devices = await self.list_input_devices()
        return devices[0] if devices else None

    async def get_default_output_device(self) -> Optional[AudioDevice]:
        """Return default mock output device."""
        devices = await self.list_output_devices()
        return devices[0] if devices else None

    async def get_device_by_name(self, name: str) -> Optional[AudioDevice]:
        """Find device by name."""
        all_devices = await self.list_input_devices() + await self.list_output_devices()
        for device in all_devices:
            if device.name == name:
                return device
        return None


class FileAudioCapture(AudioCapture):
    """Mock audio capture that reads from a file."""

    def __init__(self, input_file: Optional[str] = None):
        self.input_file = input_file
        self._is_capturing = False
        self._audio_data: Optional[np.ndarray] = None
        self._chunk_size = 1024
        self._current_pos = 0
        self._sample_rate = 16000
        self._channels = 1

    async def start_capture(
        self,
        device: AudioDevice,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
    ) -> None:
        """Load audio data from file."""
        if not self.input_file or not Path(self.input_file).exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        # Load audio file
        self._audio_data, self._sample_rate = sf.read(self.input_file)
        if len(self._audio_data.shape) > 1:
            self._audio_data = self._audio_data[:, 0]  # Convert to mono

        # Resample if needed
        if self._sample_rate != sample_rate:
            from .audio_utils import resample_audio

            self._audio_data = resample_audio(
                self._audio_data, orig_sr=self._sample_rate, target_sr=sample_rate
            )
            self._sample_rate = sample_rate

        self._channels = channels
        self._chunk_size = chunk_size
        self._current_pos = 0
        self._is_capturing = True

        logger.info(
            f"Started mock capture from {self.input_file}, "
            f"sample_rate={sample_rate}, channels={channels}"
        )

    async def stop_capture(self) -> None:
        """Stop capturing."""
        self._is_capturing = False
        logger.info("Stopped mock capture")

    async def read_audio_chunk(self) -> Optional[np.ndarray]:
        """Read next chunk from loaded audio data."""
        if not self._is_capturing or self._audio_data is None:
            return None

        if self._current_pos >= len(self._audio_data):
            return None  # End of file

        end_pos = min(self._current_pos + self._chunk_size, len(self._audio_data))
        chunk = self._audio_data[self._current_pos : end_pos]
        self._current_pos = end_pos

        return chunk

    async def capture_stream(  # type: ignore[override]
        self, sample_rate: int = 22050, channels: int = 1
    ) -> AsyncGenerator[np.ndarray, None]:
        """Stream audio chunks from file."""
        while self._is_capturing:
            chunk = await self.read_audio_chunk()
            if chunk is None:
                break
            yield chunk
            await asyncio.sleep(0.01)  # Simulate real-time

    def is_capturing(self) -> bool:
        """Check if capturing."""
        return self._is_capturing


class FileAudioOutput(AudioOutput):
    """Mock audio output that writes to a file."""

    def __init__(self, output_file: Optional[str] = None):
        self.output_file = output_file
        self._is_playing = False
        self._audio_buffer: List[np.ndarray] = []
        self._sample_rate = 22050
        self._channels = 1

    async def start_playback(
        self, device: AudioDevice, sample_rate: int = 22050, channels: int = 1
    ) -> None:
        """Initialize playback parameters."""
        self._sample_rate = sample_rate
        self._channels = channels
        self._audio_buffer = []
        self._is_playing = True

        logger.info(
            f"Started mock playback to {self.output_file}, "
            f"sample_rate={sample_rate}, channels={channels}"
        )

    async def stop_playback(self) -> None:
        """Stop playback and save to file."""
        if self._audio_buffer and self.output_file:
            # Concatenate all audio chunks
            full_audio = np.concatenate(self._audio_buffer)

            # Ensure output directory exists
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to file
            sf.write(self.output_file, full_audio, self._sample_rate)
            logger.info(f"Saved mock audio output to {self.output_file}")

        self._is_playing = False
        self._audio_buffer = []
        logger.info("Stopped mock playback")

    async def write_audio_chunk(self, audio_data: np.ndarray) -> None:
        """Add audio chunk to buffer."""
        if self._is_playing:
            self._audio_buffer.append(audio_data.copy())

    async def play_audio_data(self, audio_data: np.ndarray) -> None:
        """Play complete audio data."""
        if self.output_file:
            # Ensure output directory exists
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save directly to file
            sf.write(self.output_file, audio_data, self._sample_rate)
            logger.info(f"Saved complete audio to {self.output_file}")

    def is_playing(self) -> bool:
        """Check if playing."""
        return self._is_playing
