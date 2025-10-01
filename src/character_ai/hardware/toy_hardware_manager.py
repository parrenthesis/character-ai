"""
Hardware management for interactive character toys.

Manages toy hardware constraints, sensors, and optimization for edge deployment.
"""

import logging
import resource
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class HardwareConstraints:
    """Hardware constraints for toy deployment."""

    max_memory_gb: float = 4.0
    max_cpu_cores: int = 4
    battery_life_hours: float = 8.0
    target_latency_ms: int = 500


class ToyHardwareManager:
    """Manages toy hardware constraints and sensors."""

    def __init__(self, constraints: Optional[HardwareConstraints] = None) -> None:
        self.constraints = constraints or HardwareConstraints()
        self.microphone: Optional[Dict[str, Any]] = None
        self.speaker: Optional[Dict[str, Any]] = None
        self.buttons: Optional[Dict[str, Any]] = None
        self.leds: Optional[Dict[str, Any]] = None
        self.battery: Optional[Dict[str, Any]] = None

    async def initialize(self) -> None:
        """Initialize hardware interfaces. Continues on per-component errors."""
        # Initialize microphone
        try:
            self.microphone = await self._init_microphone()
        except Exception as e:
            logger.error(f"Failed to initialize microphone: {e}")

        # Initialize speaker
        try:
            self.speaker = await self._init_speaker()
        except Exception as e:
            logger.error(f"Failed to initialize speaker: {e}")

        # Initialize buttons/LEDs
        try:
            self.buttons = await self._init_buttons()
        except Exception as e:
            logger.error(f"Failed to initialize buttons: {e}")
        try:
            self.leds = await self._init_leds()
        except Exception as e:
            logger.error(f"Failed to initialize LEDs: {e}")

        # Initialize battery monitoring
        try:
            self.battery = await self._init_battery()
        except Exception as e:
            logger.error(f"Failed to initialize battery monitoring: {e}")

        logger.info("Toy hardware initialization attempted for all components")

    async def optimize_for_toy(self) -> Dict[str, Any]:
        """Optimize system for toy hardware constraints."""
        optimizations = {
            "memory_limit": self.constraints.max_memory_gb,
            "cpu_cores": self.constraints.max_cpu_cores,
            "cpu_limit": self.constraints.max_cpu_cores,
            "target_latency": self.constraints.target_latency_ms,
            "battery_optimization": True,
        }

        # Set system limits
        memory_limit = int(self.constraints.max_memory_gb * 1024 * 1024 * 1024)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        except Exception as e:
            logger.warning(f"Could not set memory limit: {e}")

        return optimizations

    async def get_hardware_status(self) -> Dict[str, Any]:
        """Return current hardware component status or error if not initialized."""
        if any(
            x is None
            for x in [
                self.microphone,
                self.speaker,
                self.buttons,
                self.leds,
                self.battery,
            ]
        ):
            return {"error": "Hardware not initialized"}
        return {
            "microphone": self.microphone,
            "speaker": self.speaker,
            "buttons": self.buttons,
            "leds": self.leds,
            "battery": self.battery,
        }

    async def get_memory_usage(self) -> Dict[str, Any]:
        """Return memory usage info from system metrics."""
        if psutil:
            vm = psutil.virtual_memory()
            used_gb = float(vm.used) / (1024**3)
            total_gb = float(vm.total) / (1024**3)
        else:
            used_gb = 0.0
            total_gb = 0.0
        return {
            "used_gb": round(used_gb, 2),
            "total_gb": round(total_gb, 2),
            "percentage": float(vm.percent) if psutil and vm else 0.0,
        }

    async def get_cpu_usage(self) -> float:
        """Return CPU usage percentage."""
        if psutil:
            return float(psutil.cpu_percent(interval=None))
        else:
            return 0.0

    async def get_battery_status(self) -> Dict[str, Any]:
        """Return battery status if available."""
        if psutil:
            batt = getattr(psutil, "sensors_battery", lambda: None)()
        else:
            batt = None
        if batt is None:
            return {"error": "no battery available"}
        return {"percentage": float(batt.percent), "plugged": bool(batt.power_plugged)}

    async def _init_microphone(self) -> Dict[str, Any]:
        """Initialize microphone interface."""
        try:
            # In real implementation, this would initialize actual microphone hardware
            logger.info("Microphone interface initialized")
            return {"status": "initialized", "type": "toy_microphone"}
        except Exception as e:
            logger.error(f"Failed to initialize microphone: {e}")
            return {"status": "error", "type": "toy_microphone", "error": str(e)}

    async def _init_speaker(self) -> Dict[str, Any]:
        """Initialize speaker interface."""
        try:
            # In real implementation, this would initialize actual speaker hardware
            logger.info("Speaker interface initialized")
            return {"status": "initialized", "type": "toy_speaker"}
        except Exception as e:
            logger.error(f"Failed to initialize speaker: {e}")
            return {"status": "error", "type": "toy_speaker", "error": str(e)}

    async def _init_buttons(self) -> Dict[str, Any]:
        """Initialize button interface."""
        try:
            # In real implementation, this would initialize actual button hardware
            logger.info("Button interface initialized")
            return {"status": "initialized", "type": "toy_buttons"}
        except Exception as e:
            logger.error(f"Failed to initialize buttons: {e}")
            return {"status": "error", "type": "toy_buttons", "error": str(e)}

    async def _init_leds(self) -> Dict[str, Any]:
        """Initialize LED interface."""
        try:
            # In real implementation, this would initialize actual LED hardware
            logger.info("LED interface initialized")
            return {"status": "initialized", "type": "toy_leds"}
        except Exception as e:
            logger.error(f"Failed to initialize LEDs: {e}")
            return {"status": "error", "type": "toy_leds", "error": str(e)}

    async def _init_battery(self) -> Dict[str, Any]:
        """Initialize battery monitoring."""
        try:
            # In real implementation, this would initialize actual battery monitoring
            logger.info("Battery monitoring initialized")
            return {"status": "initialized", "type": "toy_battery"}
        except Exception as e:
            logger.error(f"Failed to initialize battery monitoring: {e}")
            return {"status": "error", "type": "toy_battery", "error": str(e)}

    async def shutdown(self) -> None:
        """Shutdown hardware interfaces and clean up resources."""
        try:
            logger.info("Shutting down toy hardware...")

            # Clear hardware references
            self.microphone = None
            self.speaker = None
            self.buttons = None
            self.leds = None
            self.battery = None

            logger.info("Toy hardware shutdown complete")

        except Exception as e:
            logger.error(f"Error during hardware shutdown: {e}")
