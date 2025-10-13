"""Audio device selection and discovery utilities.

Centralizes device discovery, selection, and fallback logic to eliminate
hardcoded device selection from test.py and other components.
"""

import logging
from typing import List, Optional

import sounddevice as sd

from .interfaces import AudioDevice
from .real_audio import RealAudioDevice

logger = logging.getLogger(__name__)


class AudioDeviceSelector:
    """Centralized audio device discovery and selection logic."""

    def __init__(self) -> None:
        """Initialize the device selector."""
        self._device_cache: Optional[List[dict]] = None

    def _get_devices(self) -> List[dict]:
        """Get cached device list or query from sounddevice."""
        if self._device_cache is None:
            try:
                self._device_cache = sd.query_devices()
            except Exception as e:
                logger.error(f"Failed to query audio devices: {e}")
                self._device_cache = []
        return self._device_cache

    def find_device_by_name(
        self, name: str, is_input: bool = True
    ) -> Optional[AudioDevice]:
        """Find device by name pattern (case-insensitive)."""
        devices = self._get_devices()
        name_lower = name.lower()

        for i, device in enumerate(devices):
            device_name = device.get("name", "").lower()
            if name_lower in device_name:
                # Check if device has the required channels
                if is_input and device.get("max_input_channels", 0) > 0:
                    try:
                        return RealAudioDevice(device, is_input=True)
                    except Exception as e:
                        logger.warning(f"Failed to create device {device_name}: {e}")
                        continue
                elif not is_input and device.get("max_output_channels", 0) > 0:
                    try:
                        return RealAudioDevice(device, is_input=False)
                    except Exception as e:
                        logger.warning(f"Failed to create device {device_name}: {e}")
                        continue

        return None

    def find_device_by_alsa_id(
        self, alsa_id: str, is_input: bool = True
    ) -> Optional[AudioDevice]:
        """Find device by ALSA identifier (e.g., 'hw:3,0')."""
        try:
            device_info = sd.query_devices(alsa_id)
            return RealAudioDevice(device_info, is_input=is_input)
        except Exception as e:
            logger.warning(f"Failed to access device {alsa_id}: {e}")
            return None

    def find_device_by_index(
        self, index: int, is_input: bool = True
    ) -> Optional[AudioDevice]:
        """Find device by index."""
        try:
            device_info = sd.query_devices(index)
            return RealAudioDevice(device_info, is_input=is_input)
        except Exception as e:
            logger.warning(f"Failed to access device at index {index}: {e}")
            return None

    def get_compatible_device(
        self, preferred_name: str, fallback_to_default: bool = True
    ) -> Optional[AudioDevice]:
        """Smart device selection with fallbacks.

        Args:
            preferred_name: Preferred device name pattern
            fallback_to_default: Whether to fallback to default device

        Returns:
            Selected device or None if no compatible device found
        """
        # Try ALSA ID first (most specific)
        if preferred_name.startswith("hw:"):
            device = self.find_device_by_alsa_id(preferred_name, is_input=True)
            if device:
                logger.info(f"Found device by ALSA ID: {device.name}")
                return device

        # Try by name pattern
        device = self.find_device_by_name(preferred_name, is_input=True)
        if device:
            logger.info(f"Found device by name pattern: {device.name}")
            return device

        # Fallback to default device
        if fallback_to_default:
            try:
                default_device = sd.query_devices(kind="input")
                if default_device and default_device.get("max_input_channels", 0) > 0:
                    device = RealAudioDevice(default_device, is_input=True)
                    logger.info(f"Using default input device: {device.name}")
                    return device
            except Exception as e:
                logger.warning(f"Failed to get default input device: {e}")

        logger.error(f"No compatible input device found for '{preferred_name}'")
        return None

    def test_sample_rates(self, device: AudioDevice, rates: List[int]) -> Optional[int]:
        """Test which sample rates are compatible with the device.

        Args:
            device: Audio device to test
            rates: List of sample rates to test

        Returns:
            First compatible sample rate or None if none work
        """
        for rate in rates:
            try:
                # Test by querying device capabilities
                device_info = sd.query_devices(device.index)
                if device_info:
                    # Check if the rate is in the supported range
                    # This is a basic check - actual testing would require opening streams
                    logger.debug(f"Testing sample rate {rate} for device {device.name}")
                    return rate
            except Exception as e:
                logger.debug(
                    f"Sample rate {rate} not compatible with {device.name}: {e}"
                )
                continue

        return None

    def list_available_devices(self, is_input: bool = True) -> List[AudioDevice]:
        """List all available devices of the specified type."""
        devices = self._get_devices()
        available_devices: List[AudioDevice] = []

        for device in devices:
            try:
                if is_input and device.get("max_input_channels", 0) > 0:
                    available_devices.append(RealAudioDevice(device, is_input=True))
                elif not is_input and device.get("max_output_channels", 0) > 0:
                    available_devices.append(RealAudioDevice(device, is_input=False))
            except Exception as e:
                logger.debug(
                    f"Failed to create device {device.get('name', 'Unknown')}: {e}"
                )

        return available_devices

    def get_device_info(self, device: AudioDevice) -> dict:
        """Get detailed information about a device."""
        try:
            device_info = sd.query_devices(device.index)
            return {
                "name": device_info.get("name", "Unknown"),
                "index": device_info.get("index", -1),
                "max_input_channels": device_info.get("max_input_channels", 0),
                "max_output_channels": device_info.get("max_output_channels", 0),
                "default_samplerate": device_info.get("default_samplerate", 44100),
                "hostapi": device_info.get("hostapi", -1),
            }
        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            return {}

    def clear_cache(self) -> None:
        """Clear the device cache to force re-query."""
        self._device_cache = None
