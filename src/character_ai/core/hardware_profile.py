"""Hardware profile management for platform-specific optimizations."""

import copy
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from .config import Config

logger = logging.getLogger(__name__)


class HardwareProfileManager:
    """Load and apply hardware-specific configurations."""

    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """Load hardware profile from configs/hardware/{profile_name}.yaml"""
        path = Path(f"configs/hardware/{profile_name}.yaml")
        if not path.exists():
            raise FileNotFoundError(f"Hardware profile not found: {profile_name}")
        with open(path) as f:
            result = yaml.safe_load(f)
            return result if result is not None else {}

    def detect_hardware(self) -> str:
        """Auto-detect hardware platform."""
        # Check for Raspberry Pi
        if Path("/proc/device-tree/model").exists():
            with open("/proc/device-tree/model") as f:
                model = f.read().lower()
                if "raspberry pi" in model:
                    return "raspberry_pi"
                if "orange pi" in model:
                    return "orange_pi"

        # Check for ARM-based systems
        try:
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read().lower()
                if "arm" in cpuinfo or "aarch64" in cpuinfo:
                    # Default to raspberry_pi for ARM systems
                    return "raspberry_pi"
        except Exception:
            pass

        # Default to desktop
        return "desktop"

    def merge_with_config(self, hardware_config: Dict, base_config: Config) -> Config:
        """Merge hardware config with base config."""
        # Hardware config takes precedence for model selection
        # Base config provides defaults for everything else
        merged = copy.deepcopy(base_config)

        # Override models if specified
        if "models" in hardware_config:
            if hasattr(merged.runtime, "models"):
                setattr(merged.runtime, "models", hardware_config["models"])

        # Override runtime settings
        if "runtime" in hardware_config:
            for key, value in hardware_config["runtime"].items():
                setattr(merged.runtime, key, value)

        # Override audio settings
        if "audio" in hardware_config:
            for key, value in hardware_config["audio"].items():
                setattr(merged.interaction, key, value)

        # Override VAD settings
        if "vad" in hardware_config:
            # VAD settings would need to be added to Config class
            # For now, we'll store them in a custom section
            if not hasattr(merged, "vad"):
                merged.vad = type("VADConfig", (), {})()  # type: ignore[attr-defined]
            for key, value in hardware_config["vad"].items():
                setattr(merged.vad, key, value)  # type: ignore[attr-defined]

        # Override power settings
        if "power" in hardware_config:
            if not hasattr(merged, "power"):
                merged.power = type("PowerConfig", (), {})()  # type: ignore[attr-defined]
            for key, value in hardware_config["power"].items():
                setattr(merged.power, key, value)  # type: ignore[attr-defined]

        # Override optimization settings
        if "optimizations" in hardware_config:
            if not hasattr(merged, "optimizations"):
                merged.optimizations = type("OptimizationConfig", (), {})()  # type: ignore[attr-defined]
            for key, value in hardware_config["optimizations"].items():
                setattr(merged.optimizations, key, value)  # type: ignore[attr-defined]

        return merged

    def get_hardware_constraints(self, profile_name: str) -> Dict[str, Any]:
        """Get hardware constraints from profile."""
        profile = self.load_profile(profile_name)
        constraints = profile.get("constraints", {})
        return constraints if constraints is not None else {}

    def list_available_profiles(self) -> list[str]:
        """List all available hardware profiles."""
        hardware_dir = Path("configs/hardware")
        if not hardware_dir.exists():
            return []

        profiles = []
        for yaml_file in hardware_dir.glob("*.yaml"):
            profiles.append(yaml_file.stem)

        return sorted(profiles)
