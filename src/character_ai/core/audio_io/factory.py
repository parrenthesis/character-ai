"""Factory for creating audio components based on configuration.

This allows switching between mock and real audio implementations
via environment variables or configuration files.
"""

import logging
import os
from typing import Optional

from .interfaces import AudioCapture, AudioDeviceManager, AudioOutput
from .mock_audio import FileAudioCapture, FileAudioOutput, MockAudioDeviceManager
from .real_audio import RealAudioCapture, RealAudioDeviceManager, RealAudioOutput

logger = logging.getLogger(__name__)


class AudioComponentFactory:
    """Factory for creating audio components."""

    @staticmethod
    def create_device_manager(use_mocks: Optional[bool] = None) -> AudioDeviceManager:
        """Create audio device manager.

        Args:
            use_mocks: If True, use mock implementations. If None, check environment.
        """
        if use_mocks is None:
            use_mocks = os.getenv("USE_MOCK_AUDIO", "false").lower() == "true"

        if use_mocks:
            logger.info("Using mock audio device manager")
            return MockAudioDeviceManager()
        else:
            logger.info("Using real audio device manager")
            return RealAudioDeviceManager()

    @staticmethod
    def create_audio_capture(
        use_mocks: Optional[bool] = None, input_file: Optional[str] = None
    ) -> AudioCapture:
        """Create audio capture component.

        Args:
            use_mocks: If True, use mock implementations. If None, check environment.
            input_file: Input file path for mock capture.
        """
        if use_mocks is None:
            use_mocks = os.getenv("USE_MOCK_AUDIO", "false").lower() == "true"

        if use_mocks:
            logger.info(f"Using mock audio capture with input file: {input_file}")
            return FileAudioCapture(input_file)
        else:
            logger.info("Using real audio capture")
            return RealAudioCapture()

    @staticmethod
    def create_audio_output(
        use_mocks: Optional[bool] = None, output_file: Optional[str] = None
    ) -> AudioOutput:
        """Create audio output component.

        Args:
            use_mocks: If True, use mock implementations. If None, check environment.
            output_file: Output file path for mock output.
        """
        if use_mocks is None:
            use_mocks = os.getenv("USE_MOCK_AUDIO", "false").lower() == "true"

        if use_mocks:
            logger.info(f"Using mock audio output with output file: {output_file}")
            return FileAudioOutput(output_file)
        else:
            logger.info("Using real audio output")
            return RealAudioOutput()
