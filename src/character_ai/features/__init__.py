"""
Features package for optional character AI functionality.

This package contains feature modules that are not part of the core
STT-LLM-TTS pipeline but provide additional capabilities like
personalization, parental controls, security, and localization.
"""

from .localization import LocalizationService, create_localization_manager
from .parental_controls import ParentalControlService, create_parental_control_manager
from .security import DeviceIdentityService, SecurityMiddleware

__all__ = [
    "ParentalControlService",
    "create_parental_control_manager",
    "DeviceIdentityService",
    "SecurityMiddleware",
    "LocalizationService",
    "create_localization_manager",
]
