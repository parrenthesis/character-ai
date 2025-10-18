"""Audio system management for real-time interaction."""

import logging
from typing import Any, Dict, Optional

from ...core.audio_io.device_selector import AudioDeviceSelector
from ...core.audio_io.factory import AudioComponentFactory
from ...core.audio_io.vad_session import VADSessionManager
from ...hardware.toy_hardware_manager import ToyHardwareManager

logger = logging.getLogger(__name__)


class AudioSystem:
    """Manages audio components and device selection."""

    def __init__(self, hardware_manager: Optional[ToyHardwareManager] = None):
        self.hardware_manager = hardware_manager

        # Audio components
        self.audio_device_selector = AudioDeviceSelector()
        self.audio_factory = AudioComponentFactory()
        self.vad_session_manager = VADSessionManager()

    async def initialize(self) -> None:
        """Initialize audio system components."""
        try:
            logger.info("Initializing audio system...")

            # AudioDeviceSelector and AudioComponentFactory don't need explicit initialization
            # VAD session manager doesn't need explicit initialization

            logger.info("Audio system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize audio system: {e}")
            raise

    def get_audio_devices(self) -> Dict[str, Any]:
        """Get available audio devices."""
        input_devices = self.audio_device_selector.list_available_devices(is_input=True)
        output_devices = self.audio_device_selector.list_available_devices(
            is_input=False
        )
        return {
            "input_devices": [
                {"name": d.name, "index": d.index} for d in input_devices
            ],
            "output_devices": [
                {"name": d.name, "index": d.index} for d in output_devices
            ],
        }

    def select_audio_device(
        self, device_type: str, device_name: Optional[str] = None
    ) -> Optional[str]:
        """Select audio device for input or output."""
        is_input = device_type.lower() == "input"
        devices = self.audio_device_selector.list_available_devices(is_input=is_input)
        if device_name:
            for device in devices:
                if device.name == device_name:
                    return device.name
        return devices[0].name if devices else None

    def create_audio_component(self, component_type: str, **kwargs: Any) -> Any:
        """Create audio component using factory."""
        if component_type == "capture":
            return self.audio_factory.create_audio_capture(**kwargs)
        elif component_type == "output":
            return self.audio_factory.create_audio_output(**kwargs)
        elif component_type == "device_manager":
            return self.audio_factory.create_device_manager(**kwargs)
        else:
            raise ValueError(f"Unknown component type: {component_type}")

    def get_vad_session(self, session_id: str) -> Any:
        """Get or create VAD session."""
        # VADSessionManager doesn't have get_session method, return the manager itself
        return self.vad_session_manager

    async def shutdown(self) -> None:
        """Shutdown audio system components."""
        try:
            logger.info("Shutting down audio system...")

            # Clean up VAD sessions
            if self.vad_session_manager:
                self.vad_session_manager.reset_session()

            logger.info("Audio system shutdown complete")
        except Exception as e:
            logger.error(f"Error during audio system shutdown: {e}")
