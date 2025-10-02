"""
Parental controls and monitoring system for child safety.

Provides comprehensive parental controls including content filtering, usage monitoring,
time limits, and safety alerts for child-focused deployments.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..algorithms.safety.classifier import SafetyLevel, SafetyResult
from ..algorithms.safety.multilingual_classifier import MultilingualSafetyClassifier
from .log_aggregation import get_log_aggregator
from .metrics import get_metrics_collector

logger = logging.getLogger(__name__)


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
    USAGE_EXCEEDED = "usage_exceeded"


class ChildAgeGroup(Enum):
    """Age groups for child-appropriate content."""

    TODDLER = "toddler"  # 2-4 years
    PRESCHOOL = "preschool"  # 4-6 years
    ELEMENTARY = "elementary"  # 6-12 years
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
    """Safety alert for parental monitoring."""

    alert_id: str
    alert_type: AlertType
    severity: str  # "low", "medium", "high", "critical"
    message: str
    timestamp: float
    user_id: str
    device_id: str
    character_id: Optional[str] = None
    content: Optional[str] = None
    safety_result: Optional[SafetyResult] = None
    resolved: bool = False
    parent_notified: bool = False


@dataclass
class UsageStats:
    """Usage statistics for monitoring."""

    user_id: str
    device_id: str
    date: str  # YYYY-MM-DD format
    total_time_minutes: int = 0
    session_count: int = 0
    character_interactions: int = 0
    content_blocks: int = 0
    safety_violations: int = 0
    topics_discussed: Set[str] = field(default_factory=set)
    languages_used: Set[str] = field(default_factory=set)
    peak_usage_hour: Optional[int] = None


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


class ParentalControlManager:
    """Manages parental controls and child safety monitoring."""

    def __init__(self, storage_path: Path = Path.cwd() / "data/parental_controls"):
        self.storage_path = storage_path
        # Don't create directory during instantiation - create when actually needed

        # Core components
        self.safety_classifier = MultilingualSafetyClassifier()
        self.metrics_collector = get_metrics_collector()
        self.log_aggregator = get_log_aggregator()

        # Data storage
        self.profiles: Dict[str, ParentalControlProfile] = {}
        self.usage_stats: Dict[str, UsageStats] = {}  # child_id -> stats
        self.alerts: List[SafetyAlert] = []
        self.active_sessions: Dict[str, Dict[str, Any]] = {}  # child_id -> session data

        # Don't load data during instantiation - load when actually needed

        logger.info(
            "ParentalControlManager initialized",
            extra={"storage_path": str(storage_path)},
        )

    def _load_profiles(self) -> None:
        """Load parental control profiles from storage."""
        try:
            # Create directory if it doesn't exist
            if not self.storage_path.exists():
                self.storage_path.mkdir(parents=True, exist_ok=True)

            profiles_file = self.storage_path / "profiles.json"
            if profiles_file.exists():
                with open(profiles_file, "r") as f:
                    profiles_data = json.load(f)

                for child_id, profile_data in profiles_data.items():
                    self.profiles[child_id] = ParentalControlProfile.from_dict(
                        profile_data
                    )

                logger.info(f"Loaded {len(self.profiles)} parental control profiles")
        except Exception as e:
            logger.error(f"Failed to load parental control profiles: {e}")

    def _save_profiles(self) -> None:
        """Save parental control profiles to storage."""
        try:
            # Create directory if it doesn't exist
            if not self.storage_path.exists():
                self.storage_path.mkdir(parents=True, exist_ok=True)

            profiles_file = self.storage_path / "profiles.json"
            profiles_data = {
                child_id: profile.to_dict()
                for child_id, profile in self.profiles.items()
            }

            with open(profiles_file, "w") as f:
                json.dump(profiles_data, f, indent=2)

            logger.debug("Saved parental control profiles to storage")
        except Exception as e:
            logger.error(f"Failed to save parental control profiles: {e}")

    def _load_usage_stats(self) -> None:
        """Load usage statistics from storage."""
        try:
            stats_file = self.storage_path / "usage_stats.json"
            if stats_file.exists():
                with open(stats_file, "r") as f:
                    stats_data = json.load(f)

                for child_id, stats_data in stats_data.items():
                    self.usage_stats[child_id] = UsageStats(
                        user_id=stats_data["user_id"],
                        device_id=stats_data["device_id"],
                        date=stats_data["date"],
                        total_time_minutes=stats_data["total_time_minutes"],
                        session_count=stats_data["session_count"],
                        character_interactions=stats_data["character_interactions"],
                        content_blocks=stats_data["content_blocks"],
                        safety_violations=stats_data["safety_violations"],
                        topics_discussed=set(stats_data.get("topics_discussed", [])),
                        languages_used=set(stats_data.get("languages_used", [])),
                        peak_usage_hour=stats_data.get("peak_usage_hour"),
                    )

                logger.info(f"Loaded usage stats for {len(self.usage_stats)} children")
        except Exception as e:
            logger.error(f"Failed to load usage statistics: {e}")

    def _save_usage_stats(self) -> None:
        """Save usage statistics to storage."""
        try:
            stats_file = self.storage_path / "usage_stats.json"
            stats_data = {}
            for child_id, stats in self.usage_stats.items():
                stats_data[child_id] = {
                    "user_id": stats.user_id,
                    "device_id": stats.device_id,
                    "date": stats.date,
                    "total_time_minutes": stats.total_time_minutes,
                    "session_count": stats.session_count,
                    "character_interactions": stats.character_interactions,
                    "content_blocks": stats.content_blocks,
                    "safety_violations": stats.safety_violations,
                    "topics_discussed": list(stats.topics_discussed),
                    "languages_used": list(stats.languages_used),
                    "peak_usage_hour": stats.peak_usage_hour,
                }

            with open(stats_file, "w") as f:
                json.dump(stats_data, f, indent=2)

            logger.debug("Saved usage statistics to storage")
        except Exception as e:
            logger.error(f"Failed to save usage statistics: {e}")

    def _load_alerts(self) -> None:
        """Load safety alerts from storage."""
        try:
            alerts_file = self.storage_path / "alerts.json"
            if alerts_file.exists():
                with open(alerts_file, "r") as f:
                    alerts_data = json.load(f)

                for alert_data in alerts_data:
                    alert = SafetyAlert(
                        alert_id=alert_data["alert_id"],
                        alert_type=AlertType(alert_data["alert_type"]),
                        severity=alert_data["severity"],
                        message=alert_data["message"],
                        timestamp=alert_data["timestamp"],
                        user_id=alert_data["user_id"],
                        device_id=alert_data["device_id"],
                        character_id=alert_data.get("character_id"),
                        content=alert_data.get("content"),
                        resolved=alert_data.get("resolved", False),
                        parent_notified=alert_data.get("parent_notified", False),
                    )
                    self.alerts.append(alert)

                logger.info(f"Loaded {len(self.alerts)} safety alerts")
        except Exception as e:
            logger.error(f"Failed to load safety alerts: {e}")

    def _save_alerts(self) -> None:
        """Save safety alerts to storage."""
        try:
            alerts_file = self.storage_path / "alerts.json"
            alerts_data = []
            for alert in self.alerts:
                alerts_data.append(
                    {
                        "alert_id": alert.alert_id,
                        "alert_type": alert.alert_type.value,
                        "severity": alert.severity,
                        "message": alert.message,
                        "timestamp": alert.timestamp,
                        "user_id": alert.user_id,
                        "device_id": alert.device_id,
                        "character_id": alert.character_id,
                        "content": alert.content,
                        "resolved": alert.resolved,
                        "parent_notified": alert.parent_notified,
                    }
                )

            with open(alerts_file, "w") as f:
                json.dump(alerts_data, f, indent=2)

            logger.debug("Saved safety alerts to storage")
        except Exception as e:
            logger.error(f"Failed to save safety alerts: {e}")

    def create_profile(
        self, child_id: str, parent_id: str, child_name: str, age_group: ChildAgeGroup
    ) -> ParentalControlProfile:
        """Create a new parental control profile."""
        profile = ParentalControlProfile(
            child_id=child_id,
            parent_id=parent_id,
            child_name=child_name,
            age_group=age_group,
            created_at=time.time(),
            last_updated=time.time(),
        )

        # Set age-appropriate defaults
        self._set_age_appropriate_defaults(profile)

        self.profiles[child_id] = profile
        self._save_profiles()

        logger.info(
            f"Created parental control profile for child {child_name} ({age_group.value})"
        )
        return profile

    def _set_age_appropriate_defaults(self, profile: ParentalControlProfile) -> None:
        """Set age-appropriate default settings."""
        if profile.age_group == ChildAgeGroup.TODDLER:
            profile.content_filter.level = ContentFilterLevel.STRICT
            profile.content_filter.blocked_categories = {"violence", "scary", "adult"}
            profile.safety_level = "strict"
            profile.time_limits = [TimeLimit(TimeLimitType.DAILY, 30, 15)]
        elif profile.age_group == ChildAgeGroup.PRESCHOOL:
            profile.content_filter.level = ContentFilterLevel.STRICT
            profile.content_filter.blocked_categories = {"violence", "scary"}
            profile.safety_level = "strict"
            profile.time_limits = [TimeLimit(TimeLimitType.DAILY, 60, 30)]
        elif profile.age_group == ChildAgeGroup.ELEMENTARY:
            profile.content_filter.level = ContentFilterLevel.MODERATE
            profile.content_filter.blocked_categories = {"violence", "adult"}
            profile.safety_level = "moderate"
            profile.time_limits = [TimeLimit(TimeLimitType.DAILY, 120, 30)]
        elif profile.age_group == ChildAgeGroup.TEEN:
            profile.content_filter.level = ContentFilterLevel.LENIENT
            profile.content_filter.blocked_categories = {"adult"}
            profile.safety_level = "lenient"
            profile.time_limits = [TimeLimit(TimeLimitType.DAILY, 180, 60)]

    def get_profile(self, child_id: str) -> Optional[ParentalControlProfile]:
        """Get parental control profile for a child."""
        return self.profiles.get(child_id)

    def update_profile(self, child_id: str, updates: Dict[str, Any]) -> bool:
        """Update parental control profile."""
        if child_id not in self.profiles:
            return False

        profile = self.profiles[child_id]

        # Update fields
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        profile.last_updated = time.time()
        self._save_profiles()

        logger.info(f"Updated parental control profile for child {child_id}")
        return True

    def check_content_safety(
        self, child_id: str, content: str, character_id: Optional[str] = None
    ) -> Tuple[bool, SafetyResult, Optional[SafetyAlert]]:
        """Check if content is safe for a child."""
        profile = self.get_profile(child_id)
        if not profile:
            return True, SafetyResult(SafetyLevel.SAFE, 1.0, [], {}, 0.0), None

        # Get safety classification
        safety_result = self.safety_classifier.classify(content)

        # Apply content filtering
        is_safe = self._apply_content_filters(profile, content, safety_result)

        # Create alert if content is blocked
        alert = None
        if not is_safe:
            alert = self._create_safety_alert(
                child_id=child_id,
                alert_type=AlertType.CONTENT_BLOCKED,
                severity=(
                    "high" if safety_result.level == SafetyLevel.UNSAFE else "medium"
                ),
                message=f"Content blocked: {safety_result.categories}",
                content=content,
                character_id=character_id,
                safety_result=safety_result,
            )

            # Update usage stats
            self._update_usage_stats(child_id, content_blocked=True)

        return is_safe, safety_result, alert

    def _apply_content_filters(
        self, profile: ParentalControlProfile, content: str, safety_result: SafetyResult
    ) -> bool:
        """Apply content filtering rules."""
        filter_config = profile.content_filter

        # Check safety level
        if profile.block_unsafe_content and safety_result.level == SafetyLevel.UNSAFE:
            return False

        # Check blocked categories
        if safety_result.categories and any(
            cat in filter_config.blocked_categories for cat in safety_result.categories
        ):
            return False

        # Check blocked keywords
        content_lower = content.lower()
        for keyword in filter_config.blocked_keywords:
            if keyword.lower() in content_lower:
                return False

        # Check language restrictions
        if filter_config.language_restrictions:
            # Detect language and check against restrictions
            try:
                from .language_support import get_localization_manager

                localization_manager = get_localization_manager()
                detection_result = localization_manager.detect_and_set_language(content)

                # Check if detected language is in restricted languages
                if (
                    detection_result.detected_language.value
                    in filter_config.language_restrictions
                ):
                    logger.info(
                        f"Content blocked due to language restriction: {detection_result.detected_language.value}"
                    )
                    return False

            except Exception as e:
                logger.warning(f"Language detection failed for content filtering: {e}")
                # If language detection fails, allow content (fail open for safety)

        # Check custom rules
        for rule in filter_config.custom_rules:
            if self._evaluate_custom_rule(rule, content):
                return False

        return True

    def _evaluate_custom_rule(self, rule: Dict[str, Any], content: str) -> bool:
        """Evaluate a custom content filtering rule."""
        rule_type = rule.get("type", "keyword")

        if rule_type == "keyword":
            keywords = rule.get("keywords", [])
            return any(keyword.lower() in content.lower() for keyword in keywords)
        elif rule_type == "regex":
            import re

            pattern = rule.get("pattern", "")
            return re.search(pattern, content, re.IGNORECASE) is not None
        elif rule_type == "length":
            max_length = int(rule.get("max_length", 1000))
            return len(content) > max_length

        return False

    def check_time_limits(self, child_id: str) -> Tuple[bool, Optional[SafetyAlert]]:
        """Check if child has exceeded time limits."""
        profile = self.get_profile(child_id)
        if not profile:
            return True, None

        current_time = time.time()
        current_date = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d")

        # Get or create usage stats for today
        if child_id not in self.usage_stats:
            self.usage_stats[child_id] = UsageStats(
                user_id=child_id,
                device_id="",  # Will be set when session starts
                date=current_date,
            )

        stats = self.usage_stats[child_id]

        # Check each time limit
        for time_limit in profile.time_limits:
            if not time_limit.enabled:
                continue

            if time_limit.limit_type == TimeLimitType.DAILY:
                if stats.total_time_minutes >= time_limit.duration_minutes:
                    alert = self._create_safety_alert(
                        child_id=child_id,
                        alert_type=AlertType.TIME_LIMIT_REACHED,
                        severity="medium",
                        message=f"Daily time limit of {time_limit.duration_minutes} minutes reached",
                    )
                    return False, alert

            elif time_limit.limit_type == TimeLimitType.SESSION:
                # Check current session time
                if child_id in self.active_sessions:
                    session_start = self.active_sessions[child_id].get(
                        "start_time", current_time
                    )
                    session_duration = (current_time - session_start) / 60  # minutes
                    if session_duration >= time_limit.duration_minutes:
                        alert = self._create_safety_alert(
                            child_id=child_id,
                            alert_type=AlertType.TIME_LIMIT_REACHED,
                            severity="medium",
                            message=f"Session time limit of {time_limit.duration_minutes} minutes reached",
                        )
                        return False, alert

        return True, None

    def start_session(self, child_id: str, device_id: str) -> bool:
        """Start a monitoring session for a child."""
        # Check time limits first
        can_continue, alert = self.check_time_limits(child_id)
        if not can_continue:
            return False

        # Start session
        self.active_sessions[child_id] = {
            "start_time": time.time(),
            "device_id": device_id,
            "character_id": None,
            "interaction_count": 0,
        }

        logger.info(f"Started monitoring session for child {child_id}")
        return True

    def end_session(self, child_id: str) -> None:
        """End a monitoring session for a child."""
        if child_id not in self.active_sessions:
            return

        session_data = self.active_sessions[child_id]
        session_duration = (time.time() - session_data["start_time"]) / 60  # minutes

        # Update usage stats
        self._update_usage_stats(child_id, session_duration=session_duration)

        # Remove session
        del self.active_sessions[child_id]

        logger.info(
            f"Ended monitoring session for child {child_id}, duration: {session_duration:.1f} minutes"
        )

    def record_interaction(
        self, child_id: str, character_id: str, content: str
    ) -> Tuple[bool, Optional[SafetyAlert]]:
        """Record a child's interaction with a character."""
        # Check content safety
        is_safe, safety_result, alert = self.check_content_safety(
            child_id, content, character_id
        )

        if not is_safe:
            return False, alert

        # Update session data
        if child_id in self.active_sessions:
            self.active_sessions[child_id]["character_id"] = character_id
            self.active_sessions[child_id]["interaction_count"] += 1

        # Update usage stats
        self._update_usage_stats(child_id, interaction=True)

        # Check for suspicious activity
        if self._detect_suspicious_activity(child_id, content):
            alert = self._create_safety_alert(
                child_id=child_id,
                alert_type=AlertType.SUSPICIOUS_ACTIVITY,
                severity="medium",
                message="Suspicious activity detected",
                content=content,
                character_id=character_id,
            )
            return True, alert

        return True, None

    def _detect_suspicious_activity(self, child_id: str, content: str) -> bool:
        """Detect suspicious activity patterns."""
        # Simple heuristics - in production, this would be more sophisticated
        suspicious_patterns = [
            "personal information",
            "meet me",
            "don't tell",
            "secret",
            "private",
        ]

        content_lower = content.lower()
        return any(pattern in content_lower for pattern in suspicious_patterns)

    def _update_usage_stats(
        self,
        child_id: str,
        session_duration: float = 0,
        interaction: bool = False,
        content_blocked: bool = False,
    ) -> None:
        """Update usage statistics for a child."""
        if child_id not in self.usage_stats:
            current_date = datetime.now().strftime("%Y-%m-%d")
            self.usage_stats[child_id] = UsageStats(
                user_id=child_id, device_id="", date=current_date
            )

        stats = self.usage_stats[child_id]

        if session_duration > 0:
            stats.total_time_minutes += int(session_duration)
            stats.session_count += 1

        if interaction:
            stats.character_interactions += 1

        if content_blocked:
            stats.content_blocks += 1

        # Update peak usage hour
        current_hour = datetime.now().hour
        if stats.peak_usage_hour is None or current_hour > stats.peak_usage_hour:
            stats.peak_usage_hour = current_hour

        self._save_usage_stats()

    def _create_safety_alert(
        self,
        child_id: str,
        alert_type: AlertType,
        severity: str,
        message: str,
        content: Optional[str] = None,
        character_id: Optional[str] = None,
        safety_result: Optional[SafetyResult] = None,
    ) -> SafetyAlert:
        """Create a safety alert."""
        alert = SafetyAlert(
            alert_id=f"alert_{int(time.time() * 1000)}_{hashlib.sha256(child_id.encode()).hexdigest()[:8]}",
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=time.time(),
            user_id=child_id,
            device_id=self.active_sessions.get(child_id, {}).get("device_id", ""),
            character_id=character_id,
            content=content,
            safety_result=safety_result,
        )

        self.alerts.append(alert)
        self._save_alerts()

        logger.warning(f"Created safety alert for child {child_id}: {message}")
        return alert

    def get_usage_report(self, child_id: str, days: int = 7) -> Dict[str, Any]:
        """Get usage report for a child."""
        profile = self.get_profile(child_id)
        if not profile:
            return {"error": "Profile not found"}

        # Get usage stats
        stats = self.usage_stats.get(child_id)
        if not stats:
            return {"error": "No usage data available"}

        # Get recent alerts
        recent_alerts = [
            alert
            for alert in self.alerts
            if alert.user_id == child_id
            and alert.timestamp > time.time() - (days * 86400)
        ]

        return {
            "child_id": child_id,
            "child_name": profile.child_name,
            "age_group": profile.age_group.value,
            "usage_stats": {
                "total_time_minutes": stats.total_time_minutes,
                "session_count": stats.session_count,
                "character_interactions": stats.character_interactions,
                "content_blocks": stats.content_blocks,
                "safety_violations": stats.safety_violations,
                "topics_discussed": list(stats.topics_discussed),
                "languages_used": list(stats.languages_used),
                "peak_usage_hour": stats.peak_usage_hour,
            },
            "recent_alerts": [
                {
                    "alert_type": alert.alert_type.value,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                    "resolved": alert.resolved,
                }
                for alert in recent_alerts
            ],
            "profile_settings": {
                "content_filter_level": profile.content_filter.level.value,
                "safety_level": profile.safety_level,
                "monitoring_enabled": profile.monitoring_enabled,
                "time_limits": len(profile.time_limits),
            },
        }

    def get_parent_dashboard(self, parent_id: str) -> Dict[str, Any]:
        """Get parent dashboard with all children's data."""
        children_profiles = [
            profile
            for profile in self.profiles.values()
            if profile.parent_id == parent_id
        ]

        dashboard_data = {
            "parent_id": parent_id,
            "children": [],
            "total_alerts": 0,
            "active_sessions": 0,
        }

        for profile in children_profiles:
            child_data = self.get_usage_report(profile.child_id)
            dashboard_data["children"].append(child_data)  # type: ignore

            # Count alerts
            child_alerts = [
                alert
                for alert in self.alerts
                if alert.user_id == profile.child_id and not alert.resolved
            ]
            dashboard_data["total_alerts"] += len(child_alerts)  # type: ignore

            # Check active sessions
            if profile.child_id in self.active_sessions:
                dashboard_data["active_sessions"] += 1  # type: ignore

        return dashboard_data


# Global instance
_parental_control_manager: Optional[ParentalControlManager] = None


def get_parental_control_manager() -> ParentalControlManager:
    """Get the global parental control manager instance."""
    global _parental_control_manager
    if _parental_control_manager is None:
        _parental_control_manager = ParentalControlManager()
    return _parental_control_manager
