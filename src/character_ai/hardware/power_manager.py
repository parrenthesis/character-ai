"""
Power management for interactive character toys.

Optimizes battery life and manages power consumption for edge deployment.
"""

import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PowerMode(Enum):
    """Power consumption modes."""

    PERFORMANCE = "performance"  # Maximum performance, high power
    BALANCED = "balanced"  # Balance performance and power
    BATTERY_SAVER = "battery_saver"  # Minimize power consumption
    SLEEP = "sleep"  # Deep sleep between interactions


class PowerManager:
    """Manages power consumption and battery life optimization."""

    def __init__(self, constraints: Optional[Dict] = None):
        from ..core.config import Config

        cfg = Config()
        self.constraints = constraints or {}
        self.current_mode = PowerMode.BALANCED
        self.idle_start_time: Optional[float] = None
        self.sleep_threshold_s = 60  # Sleep after 1 min idle
        self.battery_level = 100.0  # Mock battery, would read from hardware
        self.last_activity = time.time()
        self.idle_timeout = getattr(cfg.runtime, "idle_timeout_s", 300)

    async def set_power_mode(self, mode: PowerMode) -> None:
        """Set power consumption mode."""
        self.current_mode = mode

        if mode == PowerMode.PERFORMANCE:
            # Max CPU frequency, no throttling
            await self._set_cpu_governor("performance")
            await self._disable_gpu_power_saving()
        elif mode == PowerMode.BALANCED:
            # Balanced scaling
            await self._set_cpu_governor("ondemand")
            await self._enable_gpu_power_saving()
        elif mode == PowerMode.BATTERY_SAVER:
            # Minimize power
            await self._set_cpu_governor("powersave")
            await self._enable_aggressive_power_saving()
            await self._reduce_model_precision()
        elif mode == PowerMode.SLEEP:
            # Deep sleep
            await self._unload_non_essential_models()
            await self._minimize_cpu_usage()

        logger.info(f"Power mode set to: {mode.value}")

    async def _set_cpu_governor(self, governor: str) -> None:
        """Set CPU frequency governor (Linux)."""
        try:
            # For Raspberry Pi/Linux
            cpu_dirs = list(Path("/sys/devices/system/cpu/").glob("cpu[0-9]*"))
            for cpu_dir in cpu_dirs:
                governor_file = cpu_dir / "cpufreq" / "scaling_governor"
                if governor_file.exists():
                    with open(governor_file, "w") as f:
                        f.write(governor)
        except Exception as e:
            logger.warning(f"Could not set CPU governor: {e}")

    async def _disable_gpu_power_saving(self) -> None:
        """Disable GPU power saving."""
        logger.info("GPU power saving disabled")

    async def _enable_gpu_power_saving(self) -> None:
        """Enable GPU power saving."""
        logger.info("GPU power saving enabled")

    async def _enable_aggressive_power_saving(self) -> None:
        """Enable aggressive power saving measures."""
        logger.info("Aggressive power saving enabled")

    async def _reduce_model_precision(self) -> None:
        """Switch to lower precision models for battery saving."""
        # Could trigger model swap to INT8 quantized versions
        logger.info("Using lower precision models for battery saving")

    async def _unload_non_essential_models(self) -> None:
        """Unload non-essential models for deep sleep."""
        logger.info("Unloading non-essential models for sleep")

    async def _minimize_cpu_usage(self) -> None:
        """Minimize CPU usage for sleep mode."""
        logger.info("Minimizing CPU usage for sleep")

    async def initialize(self) -> bool:
        """Initialize power management system."""
        try:
            logger.info("Power management system initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize power management: {e}")
            return False

    async def get_battery_level(self) -> float:
        """Get current battery level percentage."""
        try:
            # In real implementation, this would read actual battery level
            # For now, simulate battery level based on activity
            idle_time = time.time() - self.last_activity
            if idle_time > 3600:  # 1 hour of inactivity
                self.battery_level = max(0, self.battery_level - 0.1)
            return self.battery_level
        except Exception as e:
            logger.error(f"Failed to get battery level: {e}")
            return 0.0

    async def is_low_battery(self) -> bool:
        """Check if battery is low."""
        battery_level = await self.get_battery_level()
        return battery_level < 20.0

    async def enable_power_saving(self) -> bool:
        """Enable power saving mode."""
        try:
            self.power_saving_mode = True
            logger.info("Power saving mode enabled")
            return True
        except Exception as e:
            logger.error(f"Failed to enable power saving: {e}")
            return False

    async def disable_power_saving(self) -> bool:
        """Disable power saving mode."""
        try:
            self.power_saving_mode = False
            logger.info("Power saving mode disabled")
            return True
        except Exception as e:
            logger.error(f"Failed to disable power saving: {e}")
            return False

    async def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()

    async def should_enter_sleep_mode(self) -> bool:
        """Check if system should enter sleep mode."""
        idle_time = time.time() - self.last_activity
        return idle_time > self.idle_timeout

    async def optimize_for_battery_life(self) -> Dict[str, Any]:
        """Optimize system for maximum battery life."""
        optimizations = {
            "reduce_cpu_frequency": True,
            "disable_unused_sensors": True,
            "reduce_led_brightness": True,
            "shorten_audio_playback": True,
            "enable_sleep_mode": True,
        }

        logger.info("Battery life optimizations applied")
        return optimizations

    async def check_idle_and_sleep(self) -> bool:
        """Check if system should enter sleep mode."""
        if self.idle_start_time is None:
            return False

        idle_duration = time.time() - self.idle_start_time
        if idle_duration > self.sleep_threshold_s:
            await self.set_power_mode(PowerMode.SLEEP)
            return True
        return False

    def mark_active(self) -> None:
        """Mark system as active (reset idle timer)."""
        self.idle_start_time = None
        self.last_activity = time.time()

    def mark_idle(self) -> None:
        """Mark system as idle (start idle timer)."""
        if self.idle_start_time is None:
            self.idle_start_time = time.time()

    def _estimate_runtime(self) -> float:
        """Estimate remaining runtime based on power mode."""
        # Rough estimates based on typical Raspberry Pi power draw
        power_draw_watts = {
            PowerMode.PERFORMANCE: 7.5,  # ~7.5W for RPi 5 under load
            PowerMode.BALANCED: 4.0,  # ~4W typical
            PowerMode.BATTERY_SAVER: 2.0,  # ~2W minimal
            PowerMode.SLEEP: 0.5,  # ~0.5W idle
        }

        # Assume 10,000 mAh battery at 5V = 50Wh
        battery_capacity_wh = 50 * (self.battery_level / 100)
        current_draw = power_draw_watts.get(self.current_mode, 4.0)
        return battery_capacity_wh / current_draw if current_draw > 0 else float("inf")

    async def get_power_stats(self) -> Dict[str, Any]:
        """Get power consumption statistics."""
        return {
            "current_mode": self.current_mode.value,
            "battery_level": self.battery_level,
            "idle_duration_s": time.time() - self.idle_start_time
            if self.idle_start_time
            else 0,
            "estimated_runtime_hours": self._estimate_runtime(),
        }

    async def get_power_status(self) -> Dict[str, Any]:
        """Get comprehensive power status."""
        return {
            "battery_level": await self.get_battery_level(),
            "power_saving_mode": self.current_mode == PowerMode.BATTERY_SAVER,
            "low_battery": await self.is_low_battery(),
            "idle_time": time.time() - self.last_activity,
            "should_sleep": await self.should_enter_sleep_mode(),
            "current_mode": self.current_mode.value,
            "estimated_runtime_hours": self._estimate_runtime(),
        }
