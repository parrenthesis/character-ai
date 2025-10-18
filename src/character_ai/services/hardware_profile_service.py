"""Hardware profile service with unified auto-detection.

Consolidates hardware profile management and auto-detection logic.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.config import Config

logger = logging.getLogger(__name__)


class HardwareProfileManager:
    """Load and apply hardware-specific configurations."""

    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """Load hardware profile from configs/hardware/{profile_name}.yaml"""
        path = Path(f"configs/hardware/{profile_name}.yaml")
        if not path.exists():
            raise FileNotFoundError(f"Hardware profile not found: {profile_name}")
        from ..core.config.yaml_loader import YAMLConfigLoader

        return YAMLConfigLoader.load_yaml(path)

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
        except (OSError, IOError) as e:
            logger.debug(f"Could not read /proc/cpuinfo: {e}")

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

    def list_available_profiles(self) -> List[str]:
        """List all available hardware profiles."""
        hardware_dir = Path("configs/hardware")
        if not hardware_dir.exists():
            return []

        profiles = []
        for yaml_file in hardware_dir.glob("*.yaml"):
            profiles.append(yaml_file.stem)

        return sorted(profiles)


class HardwareProfileService:
    """Hardware profile management service with auto-detection."""

    def __init__(self) -> None:
        self.profile_manager = HardwareProfileManager()

    def load_or_detect(
        self, profile: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Load specified profile or auto-detect hardware.

        Args:
            profile: Profile name ("desktop", "raspberry_pi", "orange_pi") or "auto"/None for detection

        Returns:
            Tuple of (profile_name, hardware_config dict)
        """
        if profile and profile != "auto":
            config = self.profile_manager.load_profile(profile)
            logger.info(f"Loaded hardware profile: {profile}")
            return (profile, config)

        # Auto-detect
        detected = self.auto_detect()
        config = self.profile_manager.load_profile(detected)
        logger.info(f"Auto-detected hardware profile: {detected}")
        return (detected, config)

    def auto_detect(self) -> str:
        """Auto-detect hardware profile based on system characteristics.

        Uses a combination of:
        1. Device tree detection (for ARM devices like Raspberry Pi, Orange Pi)
        2. psutil-based heuristics (CPU count, memory, GPU availability)

        Returns:
            Profile name: "desktop", "raspberry_pi", or "orange_pi"
        """
        # First try device tree detection (more reliable for ARM devices)
        device_tree_profile = self.profile_manager.detect_hardware()
        if device_tree_profile != "desktop":
            # Device tree found a specific ARM device
            return device_tree_profile

        # Fall back to psutil-based detection
        try:
            import psutil

            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)

            # Check for GPU
            has_gpu = False
            try:
                import torch

                has_gpu = torch.cuda.is_available()
            except ImportError as e:
                logger.debug(f"PyTorch not available for GPU detection: {e}")

            # Heuristics for hardware detection
            if cpu_count >= 8 and memory_gb >= 16 and has_gpu:
                return "desktop"
            elif cpu_count <= 4 and memory_gb <= 4:
                return "raspberry_pi"
            elif cpu_count <= 6 and memory_gb <= 8:
                return "orange_pi"
            else:
                # Default to desktop for unknown configurations
                return "desktop"

        except Exception as e:
            logger.warning(
                f"Hardware auto-detection failed: {e}, defaulting to desktop"
            )
            return "desktop"
