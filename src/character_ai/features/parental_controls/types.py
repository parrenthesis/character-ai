"""
Parental controls types and data structures.

Contains enums, data classes, and type definitions for the parental controls system.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class ContentFilterLevel(Enum):
    """Content filtering levels for parental controls."""

    STRICT = "strict"  # Most restrictive - blocks most content
    MODERATE = "moderate"  # Balanced filtering
    LENIENT = "lenient"  # Minimal filtering
    DISABLED = "disabled"  # No filtering


class TimeLimitType(Enum):
    """Types of time limits for parental controls."""

    DAILY = "daily"
    WEEKLY = "weekly"
    SESSION = "session"
    CONTINUOUS = "continuous"  # No breaks allowed


class AlertType(Enum):
    """Types of parental control alerts."""

    CONTENT_BLOCKED = "content_blocked"
    TIME_LIMIT_REACHED = "time_limit_reached"
    SAFETY_VIOLATION = "safety_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    USAGE_ANOMALY = "usage_anomaly"


class ChildAgeGroup(Enum):
    """Age groups for child-specific parental controls."""

    TODDLER = "toddler"  # 2-4 years
    PRESCHOOL = "preschool"  # 4-6 years
    SCHOOL_AGE = "school_age"  # 6-12 years
    TEEN = "teen"  # 13-17 years


@dataclass
class ContentFilter:
    """Content filtering configuration."""

    level: ContentFilterLevel = ContentFilterLevel.MODERATE
    blocked_categories: Set[str] = field(default_factory=set)
    allowed_topics: Set[str] = field(default_factory=set)
    blocked_keywords: Set[str] = field(default_factory=set)
    age_appropriate: bool = True
    language_restrictions: Set[str] = field(default_factory=set)
    custom_rules: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TimeLimit:
    """Time limit configuration."""

    limit_type: TimeLimitType
    duration_minutes: int
    break_duration_minutes: int = 0
    start_time: Optional[str] = None  # HH:MM format
    end_time: Optional[str] = None  # HH:MM format
    days_of_week: Set[int] = field(
        default_factory=lambda: {0, 1, 2, 3, 4, 5, 6}
    )  # 0=Monday
    enabled: bool = True


@dataclass
class SafetyAlert:
    """Safety alert configuration and data."""

    alert_type: AlertType
    severity: str = "medium"  # low, medium, high, critical
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    child_id: Optional[str] = None
    device_id: Optional[str] = None
    alert_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "acknowledged": self.acknowledged,
            "child_id": self.child_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyAlert":
        """Create from dictionary."""
        return cls(
            alert_type=AlertType(data["alert_type"]),
            severity=data.get("severity", "medium"),
            message=data.get("message", ""),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
            acknowledged=data.get("acknowledged", False),
            child_id=data.get("child_id"),
        )


@dataclass
class UsageStats:
    """Usage statistics for monitoring."""

    child_id: str
    device_id: str
    date: str
    total_time_minutes: int = 0
    session_count: int = 0
    character_interactions: int = 0
    content_blocks: int = 0
    safety_violations: int = 0
    topics_discussed: Set[str] = field(default_factory=set)
    languages_used: Set[str] = field(default_factory=set)
    peak_usage_hour: int = 0
    last_activity: float = field(default_factory=time.time)
    daily_usage: Dict[str, int] = field(default_factory=dict)  # date -> minutes
    weekly_usage: Dict[str, int] = field(default_factory=dict)  # week -> minutes
    monthly_usage: Dict[str, int] = field(default_factory=dict)  # month -> minutes

    def update_daily_usage(self, date: str, minutes: int) -> None:
        """Update daily usage statistics."""
        self.daily_usage[date] = self.daily_usage.get(date, 0) + minutes

    def get_weekly_total(self) -> int:
        """Get total usage for current week."""
        return sum(self.weekly_usage.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "child_id": self.child_id,
            "device_id": self.device_id,
            "date": self.date,
            "total_time_minutes": self.total_time_minutes,
            "session_count": self.session_count,
            "character_interactions": self.character_interactions,
            "content_blocks": self.content_blocks,
            "safety_violations": self.safety_violations,
            "topics_discussed": list(self.topics_discussed),
            "languages_used": list(self.languages_used),
            "peak_usage_hour": self.peak_usage_hour,
            "last_activity": self.last_activity,
            "daily_usage": self.daily_usage,
            "weekly_usage": self.weekly_usage,
            "monthly_usage": self.monthly_usage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UsageStats":
        """Create from dictionary."""
        return cls(
            child_id=data["child_id"],
            device_id=data["device_id"],
            date=data["date"],
            total_time_minutes=data.get("total_time_minutes", 0),
            session_count=data.get("session_count", 0),
            character_interactions=data.get("character_interactions", 0),
            content_blocks=data.get("content_blocks", 0),
            safety_violations=data.get("safety_violations", 0),
            topics_discussed=set(data.get("topics_discussed", [])),
            languages_used=set(data.get("languages_used", [])),
            peak_usage_hour=data.get("peak_usage_hour", 0),
            last_activity=data.get("last_activity", time.time()),
            daily_usage=data.get("daily_usage", {}),
            weekly_usage=data.get("weekly_usage", {}),
            monthly_usage=data.get("monthly_usage", {}),
        )

    def get_monthly_total(self) -> int:
        """Get total usage for current month."""
        return sum(self.monthly_usage.values())
