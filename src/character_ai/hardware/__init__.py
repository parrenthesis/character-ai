"""
Hardware management package.

Provides hardware-specific functionality for toy devices and power management.
"""

from .power_manager import PowerManager
from .toy_hardware_manager import ToyHardwareManager

__all__ = [
    "PowerManager",
    "ToyHardwareManager",
]
