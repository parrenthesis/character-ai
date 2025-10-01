"""
Mock hardware for testing character.ai.

Provides mock implementations when no physical hardware is available.
"""

import logging
from typing import Any, Dict, Optional

from character_ai.hardware.toy_hardware_manager import (
    HardwareConstraints,
    ToyHardwareManager,
)

logger = logging.getLogger(__name__)


class MockHardwareManager(ToyHardwareManager):
    """Mock hardware manager for testing without physical hardware."""

    def __init__(self, constraints: Optional[HardwareConstraints] = None):
        super().__init__(constraints)
        self.mock_microphone_data = b""
        self.mock_button_states = {
            "button_1": False,
            "button_2": False,
            "button_3": False,
        }
        self.mock_led_states = {"led_1": False, "led_2": False, "led_3": False}
        self.mock_battery_level = 85.0

    async def initialize(self) -> None:
        """Initialize mock hardware."""
        try:
            # Initialize mock hardware interfaces
            self.microphone = await self._init_mock_microphone()
            self.speaker = await self._init_mock_speaker()
            self.buttons = await self._init_mock_buttons()
            self.leds = await self._init_mock_leds()
            self.battery = await self._init_mock_battery()

            logger.info("Mock hardware initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize mock hardware: {e}")
            raise

    async def _init_mock_microphone(self) -> Dict[str, Any]:
        """Initialize mock microphone."""
        logger.info("Mock microphone initialized")
        return {"status": "initialized", "type": "mock_microphone"}

    async def _init_mock_speaker(self) -> Dict[str, Any]:
        """Initialize mock speaker."""
        logger.info("Mock speaker initialized")
        return {"status": "initialized", "type": "mock_speaker"}

    async def _init_mock_buttons(self) -> Dict[str, Any]:
        """Initialize mock buttons."""
        logger.info("Mock buttons initialized")
        return {"status": "initialized", "type": "mock_buttons"}

    async def _init_mock_leds(self) -> Dict[str, Any]:
        """Initialize mock LEDs."""
        logger.info("Mock LEDs initialized")
        return {"status": "initialized", "type": "mock_leds"}

    async def _init_mock_battery(self) -> Dict[str, Any]:
        """Initialize mock battery monitoring."""
        logger.info("Mock battery monitoring initialized")
        return {"status": "initialized", "type": "mock_battery"}

    async def set_mock_audio_input(self, audio_data: bytes) -> None:
        """Set mock audio input for testing."""
        self.mock_microphone_data = audio_data
        logger.info(f"Set mock audio input ({len(audio_data)} bytes)")

    async def get_mock_audio_output(self) -> bytes:
        """Get mock audio output for testing."""
        # In a real implementation, this would return the actual audio output
        return b"mock_audio_output"

    async def set_mock_button_state(self, button_name: str, pressed: bool) -> None:
        """Set mock button state for testing."""
        if button_name in self.mock_button_states:
            self.mock_button_states[button_name] = pressed
            logger.info(
                f"Mock button {button_name} {'pressed' if pressed else 'released'}"
            )

    async def get_mock_button_states(self) -> Dict[str, bool]:
        """Get mock button states."""
        return self.mock_button_states.copy()

    async def set_mock_led_state(self, led_name: str, on: bool) -> None:
        """Set mock LED state for testing."""
        if led_name in self.mock_led_states:
            self.mock_led_states[led_name] = on
            logger.info(f"Mock LED {led_name} {'on' if on else 'off'}")

    async def get_mock_led_states(self) -> Dict[str, bool]:
        """Get mock LED states."""
        return self.mock_led_states.copy()

    async def set_mock_battery_level(self, level: float) -> None:
        """Set mock battery level for testing."""
        self.mock_battery_level = max(0.0, min(100.0, level))
        logger.info(f"Mock battery level set to {self.mock_battery_level}%")

    async def get_mock_battery_level(self) -> float:
        """Get mock battery level."""
        return self.mock_battery_level

    async def simulate_battery_drain(self, drain_rate: float = 0.1) -> None:
        """Simulate battery drain over time."""
        self.mock_battery_level = max(0.0, self.mock_battery_level - drain_rate)
        logger.info(f"Simulated battery drain: {self.mock_battery_level}% remaining")

    async def get_mock_hardware_status(self) -> Dict[str, Any]:
        """Get comprehensive mock hardware status."""
        return {
            "microphone": self.microphone,
            "speaker": self.speaker,
            "buttons": self.buttons,
            "leds": self.leds,
            "battery": self.battery,
            "mock_data": {
                "button_states": self.mock_button_states,
                "led_states": self.mock_led_states,
                "battery_level": self.mock_battery_level,
                "audio_input_size": len(self.mock_microphone_data),
            },
        }
