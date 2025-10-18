"""
Parental control profiles and configuration management.

Contains the ParentalControlProfile class for managing child-specific parental control settings.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .types import (
    ChildAgeGroup,
    ContentFilter,
    ContentFilterLevel,
    TimeLimit,
    TimeLimitType,
)


@dataclass
class ParentalControlProfile:
    """Complete parental control profile for a child."""

    child_id: str
    parent_id: str
    child_name: str
    age_group: ChildAgeGroup
    created_at: float
    last_updated: float

    # Content filtering
    content_filter: ContentFilter = field(default_factory=ContentFilter)

    # Time limits
    time_limits: List[TimeLimit] = field(default_factory=list)

    # Monitoring settings
    monitoring_enabled: bool = True
    alert_parents: bool = True
    log_all_interactions: bool = True

    # Safety settings
    safety_level: str = "moderate"  # "strict", "moderate", "lenient"
    block_unsafe_content: bool = True
    require_approval_for_new_characters: bool = True

    # Privacy settings
    data_retention_days: int = 30
    share_usage_with_parents: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "child_id": self.child_id,
            "parent_id": self.parent_id,
            "child_name": self.child_name,
            "age_group": self.age_group.value,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "content_filter": {
                "level": self.content_filter.level.value,
                "blocked_categories": list(self.content_filter.blocked_categories),
                "allowed_topics": list(self.content_filter.allowed_topics),
                "blocked_keywords": list(self.content_filter.blocked_keywords),
                "age_appropriate": self.content_filter.age_appropriate,
                "language_restrictions": list(
                    self.content_filter.language_restrictions
                ),
                "custom_rules": self.content_filter.custom_rules,
            },
            "time_limits": [
                {
                    "limit_type": limit.limit_type.value,
                    "duration_minutes": limit.duration_minutes,
                    "break_duration_minutes": limit.break_duration_minutes,
                    "start_time": limit.start_time,
                    "end_time": limit.end_time,
                    "days_of_week": list(limit.days_of_week),
                    "enabled": limit.enabled,
                }
                for limit in self.time_limits
            ],
            "monitoring_enabled": self.monitoring_enabled,
            "alert_parents": self.alert_parents,
            "log_all_interactions": self.log_all_interactions,
            "safety_level": self.safety_level,
            "block_unsafe_content": self.block_unsafe_content,
            "require_approval_for_new_characters": self.require_approval_for_new_characters,
            "data_retention_days": self.data_retention_days,
            "share_usage_with_parents": self.share_usage_with_parents,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParentalControlProfile":
        """Create from dictionary."""
        content_filter_data = data.get("content_filter", {})
        content_filter = ContentFilter(
            level=ContentFilterLevel(content_filter_data.get("level", "moderate")),
            blocked_categories=set(content_filter_data.get("blocked_categories", [])),
            allowed_topics=set(content_filter_data.get("allowed_topics", [])),
            blocked_keywords=set(content_filter_data.get("blocked_keywords", [])),
            age_appropriate=content_filter_data.get("age_appropriate", True),
            language_restrictions=set(
                content_filter_data.get("language_restrictions", [])
            ),
            custom_rules=content_filter_data.get("custom_rules", []),
        )

        time_limits = []
        for limit_data in data.get("time_limits", []):
            time_limits.append(
                TimeLimit(
                    limit_type=TimeLimitType(limit_data["limit_type"]),
                    duration_minutes=limit_data["duration_minutes"],
                    break_duration_minutes=limit_data.get("break_duration_minutes", 0),
                    start_time=limit_data.get("start_time"),
                    end_time=limit_data.get("end_time"),
                    days_of_week=set(
                        limit_data.get("days_of_week", [0, 1, 2, 3, 4, 5, 6])
                    ),
                    enabled=limit_data.get("enabled", True),
                )
            )

        return cls(
            child_id=data["child_id"],
            parent_id=data["parent_id"],
            child_name=data["child_name"],
            age_group=ChildAgeGroup(data["age_group"]),
            created_at=data["created_at"],
            last_updated=data["last_updated"],
            content_filter=content_filter,
            time_limits=time_limits,
            monitoring_enabled=data.get("monitoring_enabled", True),
            alert_parents=data.get("alert_parents", True),
            log_all_interactions=data.get("log_all_interactions", True),
            safety_level=data.get("safety_level", "moderate"),
            block_unsafe_content=data.get("block_unsafe_content", True),
            require_approval_for_new_characters=data.get(
                "require_approval_for_new_characters", True
            ),
            data_retention_days=data.get("data_retention_days", 30),
            share_usage_with_parents=data.get("share_usage_with_parents", True),
        )

    def update_last_updated(self) -> None:
        """Update the last_updated timestamp."""
        self.last_updated = time.time()

    def is_time_limit_active(self) -> bool:
        """Check if any time limits are currently active."""
        if not self.time_limits:
            return False

        current_time = time.time()
        current_hour = time.localtime(current_time).tm_hour
        current_minute = time.localtime(current_time).tm_min
        current_weekday = time.localtime(current_time).tm_wday

        for limit in self.time_limits:
            if not limit.enabled:
                continue

            # Check if current day is in allowed days
            if current_weekday not in limit.days_of_week:
                continue

            # Check time window if specified
            if limit.start_time and limit.end_time:
                start_hour, start_min = map(int, limit.start_time.split(":"))
                end_hour, end_min = map(int, limit.end_time.split(":"))

                current_minutes = current_hour * 60 + current_minute
                start_minutes = start_hour * 60 + start_min
                end_minutes = end_hour * 60 + end_min

                if not (start_minutes <= current_minutes <= end_minutes):
                    continue

            return True

        return False

    def get_active_time_limit(self) -> Optional[TimeLimit]:
        """Get the currently active time limit, if any."""
        if not self.time_limits:
            return None

        current_time = time.time()
        current_hour = time.localtime(current_time).tm_hour
        current_minute = time.localtime(current_time).tm_min
        current_weekday = time.localtime(current_time).tm_wday

        for limit in self.time_limits:
            if not limit.enabled:
                continue

            # Check if current day is in allowed days
            if current_weekday not in limit.days_of_week:
                continue

            # Check time window if specified
            if limit.start_time and limit.end_time:
                start_hour, start_min = map(int, limit.start_time.split(":"))
                end_hour, end_min = map(int, limit.end_time.split(":"))

                current_minutes = current_hour * 60 + current_minute
                start_minutes = start_hour * 60 + start_min
                end_minutes = end_hour * 60 + end_min

                if not (start_minutes <= current_minutes <= end_minutes):
                    continue

            return limit

        return None
