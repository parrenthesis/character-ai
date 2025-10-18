"""
Parental controls feature package.

Provides content filtering, time limits, and safety monitoring for child users.
"""

from .manager import ParentalControlService, create_parental_control_manager
from .profiles import ParentalControlProfile
from .types import (
    AlertType,
    ChildAgeGroup,
    ContentFilter,
    ContentFilterLevel,
    SafetyAlert,
    TimeLimit,
    TimeLimitType,
    UsageStats,
)

__all__ = [
    # Manager
    "ParentalControlService",
    "create_parental_control_manager",
    # Profiles
    "ParentalControlProfile",
    # Types
    "AlertType",
    "ChildAgeGroup",
    "ContentFilter",
    "ContentFilterLevel",
    "SafetyAlert",
    "TimeLimit",
    "TimeLimitType",
    "UsageStats",
]
