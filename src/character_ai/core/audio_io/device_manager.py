"""Audio device management with configurable device selection and sample rate handling."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import sounddevice as sd

logger = logging.getLogger(__name__)


@dataclass
class AudioDevice:
    """Audio device information."""

    index: int
    name: str
    input_channels: int
    output_channels: int
    default_sample_rate: float
    supported_sample_rates: List[float]


@dataclass
class AudioDeviceConfig:
    """Audio device configuration."""

    preferred_input: str
    preferred_output: str
    fallback_to_default: bool
    audiobox_config: Dict[str, Any]
    default_config: Dict[str, Any]


class AudioDeviceManager:
    """Manages audio device selection and configuration."""

    def __init__(self, config: Dict[str, Any]):
        # Provide default values for missing config fields
        default_config = {
            "preferred_input": "audiobox",
            "preferred_output": "audiobox",
            "fallback_to_default": True,
            "audiobox_config": {
                "device_name_pattern": "AudioBox USB",
                "input_channels": 2,
                "output_channels": 2,
                "supported_sample_rates": [44100, 48000, 96000],
                "preferred_sample_rate": 48000,
                "input_channel": 1,
            },
            "default_config": {"sample_rate": 22050, "channels": 1, "chunk_size": 1024},
        }

        # Merge provided config with defaults
        merged_config = {**default_config, **config}
        self.config = AudioDeviceConfig(**merged_config)
        self._devices = self._scan_devices()
        self._input_device: Optional[AudioDevice] = None
        self._output_device: Optional[AudioDevice] = None

    def _scan_devices(self) -> List[AudioDevice]:
        """Scan available audio devices."""
        devices = []
        try:
            device_list = sd.query_devices()
            for i, device_info in enumerate(device_list):
                device = AudioDevice(
                    index=i,
                    name=device_info["name"],
                    input_channels=device_info["max_input_channels"],
                    output_channels=device_info["max_output_channels"],
                    default_sample_rate=device_info["default_samplerate"],
                    supported_sample_rates=self._get_supported_sample_rates(i),
                )
                devices.append(device)
        except Exception as e:
            logger.error(f"Failed to scan audio devices: {e}")

        return devices

    def _get_supported_sample_rates(self, device_index: int) -> List[float]:
        """Get supported sample rates for a device."""
        # Common sample rates to test
        test_rates = [8000, 16000, 22050, 44100, 48000, 96000]
        supported: List[float] = []

        for rate in test_rates:
            try:
                # Test if device supports this sample rate
                sd.check_device(device_index, samplerate=rate)
                supported.append(float(rate))
            except Exception:
                continue

        return supported if supported else [44100.0]  # Fallback to 44.1kHz

    def find_audiobox_device(self) -> Optional[AudioDevice]:
        """Find AudioBox device by name pattern."""
        pattern = self.config.audiobox_config.get("device_name_pattern", "AudioBox USB")

        for device in self._devices:
            if pattern.lower() in device.name.lower():
                logger.info(
                    f"Found AudioBox device: {device.name} (index: {device.index})"
                )
                return device

        logger.warning(f"AudioBox device not found (pattern: {pattern})")
        return None

    def get_compatible_sample_rate(
        self, device: AudioDevice, target_rate: float
    ) -> float:
        """Get a compatible sample rate for the device."""
        if target_rate in device.supported_sample_rates:
            return target_rate

        # Find the closest supported rate
        supported = device.supported_sample_rates
        if not supported:
            return target_rate

        # Find closest rate
        closest = min(supported, key=lambda x: abs(x - target_rate))
        logger.info(
            f"Using compatible sample rate {closest} instead of {target_rate} for {device.name}"
        )
        return closest

    def setup_input_device(self) -> Tuple[AudioDevice, float]:
        """Setup input device with compatible sample rate."""
        if self.config.preferred_input == "audiobox":
            device = self.find_audiobox_device()
            if device and device.input_channels > 0:
                self._input_device = device
                target_rate = self.config.audiobox_config.get(
                    "preferred_sample_rate", 48000
                )
                sample_rate = self.get_compatible_sample_rate(device, target_rate)
                logger.info(f"Using AudioBox input: {device.name} at {sample_rate}Hz")
                return device, sample_rate

        # Fallback to default
        if self.config.fallback_to_default:
            default_device = self._get_default_input_device()
            if default_device:
                self._input_device = default_device
                target_rate = self.config.default_config.get("sample_rate", 22050)
                sample_rate = self.get_compatible_sample_rate(
                    default_device, target_rate
                )
                logger.info(
                    f"Using default input: {default_device.name} at {sample_rate}Hz"
                )
                return default_device, sample_rate

        raise RuntimeError("No compatible input device found")

    def setup_output_device(self) -> Tuple[AudioDevice, float]:
        """Setup output device with compatible sample rate."""
        if self.config.preferred_output == "audiobox":
            device = self.find_audiobox_device()
            if device and device.output_channels > 0:
                self._output_device = device
                target_rate = self.config.audiobox_config.get(
                    "preferred_sample_rate", 48000
                )
                sample_rate = self.get_compatible_sample_rate(device, target_rate)
                logger.info(f"Using AudioBox output: {device.name} at {sample_rate}Hz")
                return device, sample_rate

        # Fallback to default
        if self.config.fallback_to_default:
            default_device = self._get_default_output_device()
            if default_device:
                self._output_device = default_device
                target_rate = self.config.default_config.get("sample_rate", 22050)
                sample_rate = self.get_compatible_sample_rate(
                    default_device, target_rate
                )
                logger.info(
                    f"Using default output: {default_device.name} at {sample_rate}Hz"
                )
                return default_device, sample_rate

        raise RuntimeError("No compatible output device found")

    def _get_default_input_device(self) -> Optional[AudioDevice]:
        """Get default input device."""
        try:
            default_index = sd.default.device[0]
            for device in self._devices:
                if device.index == default_index and device.input_channels > 0:
                    return device
        except Exception:
            pass

        # Find any input device
        for device in self._devices:
            if device.input_channels > 0:
                return device

        return None

    def _get_default_output_device(self) -> Optional[AudioDevice]:
        """Get default output device."""
        try:
            default_index = sd.default.device[1]
            for device in self._devices:
                if device.index == default_index and device.output_channels > 0:
                    return device
        except Exception:
            pass

        # Find any output device
        for device in self._devices:
            if device.output_channels > 0:
                return device

        return None

    def get_input_device(self) -> Optional[AudioDevice]:
        """Get current input device."""
        return self._input_device

    def get_output_device(self) -> Optional[AudioDevice]:
        """Get current output device."""
        return self._output_device

    def list_devices(self) -> List[AudioDevice]:
        """List all available devices."""
        return self._devices.copy()

    def get_audiobox_config(self) -> Dict[str, Any]:
        """Get AudioBox specific configuration."""
        return self.config.audiobox_config.copy()

    def get_default_config(self) -> Dict[str, Any]:
        """Get default device configuration."""
        return self.config.default_config.copy()
