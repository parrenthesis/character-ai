"""
Parental control manager for child safety monitoring.

Manages parental controls, content filtering, time limits, and safety monitoring.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...algorithms.safety.classifier import SafetyLevel, SafetyResult
from ...algorithms.safety.multilingual_classifier import MultilingualSafetyClassifier
from ...core.persistence.base_manager import BaseDataManager
from ...core.persistence.json_manager import JSONRepository
from ...observability.log_aggregation import LogAggregator, create_log_aggregator
from ...observability.metrics import MetricsCollector, create_metrics_collector
from .profiles import ParentalControlProfile
from .types import (
    AlertType,
    ChildAgeGroup,
    ContentFilterLevel,
    SafetyAlert,
    TimeLimitType,
    UsageStats,
)

logger = logging.getLogger(__name__)


class ParentalControlService(BaseDataManager):
    """Manages parental controls and child safety monitoring."""

    def __init__(
        self,
        storage_path: Path = Path.cwd() / "data/parental_controls",
        metrics_collector: Optional[MetricsCollector] = None,
        log_aggregator: Optional[LogAggregator] = None,
    ):
        super().__init__(storage_path, "ParentalControlService")

        # Core components
        self.safety_classifier = MultilingualSafetyClassifier()
        self.metrics_collector = metrics_collector or create_metrics_collector()
        self.log_aggregator = log_aggregator or create_log_aggregator()

        # Data storage
        self.profiles: Dict[str, ParentalControlProfile] = {}
        self.usage_stats: Dict[str, UsageStats] = {}  # child_id -> stats
        self.alerts: List[SafetyAlert] = []
        self.active_sessions: Dict[str, Dict[str, Any]] = {}  # child_id -> session data

    async def _load_data(self) -> None:
        """Load data during initialization."""
        self._load_profiles()
        self._load_usage_stats()

    async def _save_data(self) -> None:
        """Save data during shutdown."""
        self._save_profiles()
        self._save_usage_stats()

    def _load_profiles(self) -> None:
        """Load parental control profiles from storage."""
        self.profiles = JSONRepository.load_json_objects(
            self.storage_path / "profiles.json", ParentalControlProfile.from_dict
        )
        logger.info(f"Loaded {len(self.profiles)} parental control profiles")

    def _save_profiles(self) -> None:
        """Save parental control profiles to storage."""
        success = JSONRepository.save_json_objects(
            self.storage_path / "profiles.json",
            self.profiles,
            lambda profile: profile.to_dict(),
        )
        if success:
            logger.info(f"Saved {len(self.profiles)} parental control profiles")
        else:
            logger.error("Failed to save parental control profiles")

    def _load_usage_stats(self) -> None:
        """Load usage statistics from storage."""
        self.usage_stats = JSONRepository.load_json_objects(
            self.storage_path / "usage_stats.json", UsageStats.from_dict
        )
        logger.info(f"Loaded usage stats for {len(self.usage_stats)} children")

    def _save_usage_stats(self) -> None:
        """Save usage statistics to storage."""
        success = JSONRepository.save_json_objects(
            self.storage_path / "usage_stats.json",
            self.usage_stats,
            lambda stats: stats.to_dict(),
        )
        if success:
            logger.info(f"Saved usage stats for {len(self.usage_stats)} children")
        else:
            logger.error("Failed to save usage stats")

    def start_session(self, child_id: str, device_id: str) -> bool:
        """Start a monitoring session for a child."""
        if child_id not in self.profiles:
            logger.warning(f"Cannot start session: no profile for child {child_id}")
            return False

        # Check time limits
        if not self._check_time_limits(child_id):
            logger.warning(f"Session blocked: time limit exceeded for child {child_id}")
            return False

        # Start session
        self.active_sessions[child_id] = {
            "device_id": device_id,
            "start_time": time.time(),
            "interactions": 0,
            "content_blocks": 0,
            "safety_violations": 0,
        }

        logger.info(f"Started session for child {child_id} on device {device_id}")
        return True

    def end_session(self, child_id: str) -> bool:
        """End a monitoring session for a child."""
        if child_id not in self.active_sessions:
            return False

        session_data = self.active_sessions.pop(child_id)
        session_duration = time.time() - session_data["start_time"]

        # Update usage stats
        if child_id not in self.usage_stats:
            from datetime import datetime

            self.usage_stats[child_id] = UsageStats(
                child_id=child_id,
                device_id=session_data["device_id"],
                date=datetime.now().strftime("%Y-%m-%d"),
            )

        stats = self.usage_stats[child_id]
        stats.total_time_minutes += int(session_duration / 60)
        stats.session_count += 1
        stats.character_interactions += session_data["interactions"]
        stats.content_blocks += session_data["content_blocks"]
        stats.safety_violations += session_data["safety_violations"]
        stats.last_activity = time.time()

        logger.info(
            f"Ended session for child {child_id}, duration: {session_duration:.1f}s"
        )
        return True

    def check_content_safety(
        self, child_id: str, text: str, character_id: str
    ) -> tuple[bool, SafetyResult, Optional[SafetyAlert]]:
        """Check if content is safe for a child."""
        if child_id not in self.profiles:
            return True, SafetyResult(SafetyLevel.SAFE, 1.0, [], {}, 0.0), None

        profile = self.profiles[child_id]

        # Use safety classifier
        safety_result = self.safety_classifier.classify(text)

        # Check against content filter
        is_safe = self._check_content_filter(profile, text, safety_result)

        alert = None
        if not is_safe:
            alert = SafetyAlert(
                alert_type=AlertType.CONTENT_BLOCKED,
                severity="high",
                message=f"Content blocked for child {child_id}: {text[:50]}...",
                child_id=child_id,
                metadata={
                    "character_id": character_id,
                    "content": text,
                    "safety_result": {
                        "level": safety_result.level.value if safety_result else None,
                        "confidence": safety_result.confidence
                        if safety_result
                        else 0.0,
                        "categories": safety_result.categories if safety_result else [],
                        "details": safety_result.details if safety_result else {},
                        "processing_time_ms": safety_result.processing_time_ms
                        if safety_result
                        else 0.0,
                    }
                    if safety_result
                    else {},
                },
            )

        return is_safe, safety_result, alert

    def _check_content_filter(
        self, profile: ParentalControlProfile, text: str, safety_result: SafetyResult
    ) -> bool:
        """Check content against profile's content filter."""
        # Check blocked keywords
        text_lower = text.lower()
        for keyword in profile.content_filter.blocked_keywords:
            if keyword.lower() in text_lower:
                return False

        # Check safety level
        if safety_result and safety_result.level in [
            SafetyLevel.UNSAFE,
            SafetyLevel.WARNING,
        ]:
            return False

        # Check age appropriateness
        if profile.content_filter.age_appropriate:
            # Simple age check - could be more sophisticated
            if profile.age_group == ChildAgeGroup.TODDLER:
                # Very strict for toddlers
                if any(
                    word in text_lower for word in ["fight", "scary", "monster", "dark"]
                ):
                    return False

        return True

    def _check_time_limits(self, child_id: str) -> bool:
        """Check if child has exceeded time limits."""
        if child_id not in self.profiles:
            return True

        profile = self.profiles[child_id]

        # Check daily time limit
        if child_id in self.usage_stats and profile.time_limits:
            stats = self.usage_stats[child_id]
            # Find the daily time limit
            for time_limit in profile.time_limits:
                if time_limit.limit_type == TimeLimitType.DAILY and time_limit.enabled:
                    if stats.total_time_minutes >= time_limit.duration_minutes:
                        return False
                    break

        return True

    def get_usage_report(self, child_id: str) -> Dict[str, Any]:
        """Get usage report for a child."""
        if child_id not in self.usage_stats:
            return {}

        stats = self.usage_stats[child_id]
        profile = self.get_profile(child_id)

        report = {
            "child_id": child_id,
            "total_time_minutes": stats.total_time_minutes,
            "session_count": stats.session_count,
            "character_interactions": stats.character_interactions,
            "content_blocks": stats.content_blocks,
            "safety_violations": stats.safety_violations,
            "topics_discussed": list(stats.topics_discussed),
            "languages_used": list(stats.languages_used),
            "peak_usage_hour": stats.peak_usage_hour,
            "last_activity": stats.last_activity,
        }

        # Add profile information if available
        if profile:
            report["child_name"] = profile.child_name
            report["age_group"] = profile.age_group.value
            report["usage_stats"] = {
                "total_time_minutes": stats.total_time_minutes,
                "character_interactions": stats.character_interactions,
                "topics_discussed": list(stats.topics_discussed),
            }

        return report

    def get_parent_dashboard(self, parent_id: str) -> Dict[str, Any]:
        """Get dashboard data for a parent."""
        children = [
            profile
            for profile in self.profiles.values()
            if profile.parent_id == parent_id
        ]

        dashboard: dict[str, Any] = {
            "parent_id": parent_id,
            "children": [],
            "total_alerts": len(
                [alert for alert in self.alerts if not alert.acknowledged]
            ),
            "summary": {
                "total_children": len(children),
                "active_sessions": len(
                    [
                        child
                        for child in children
                        if child.child_id in self.active_sessions
                    ]
                ),
                "total_usage_today": 0,
            },
        }

        for child in children:
            child_data = {
                "child_id": child.child_id,
                "child_name": child.child_name,
                "age_group": child.age_group.value,
                "is_active": child.child_id in self.active_sessions,
                "usage_stats": self.get_usage_report(child.child_id),
            }
            dashboard["children"].append(child_data)

            if child.child_id in self.usage_stats:
                dashboard["summary"]["total_usage_today"] += self.usage_stats[
                    child.child_id
                ].total_time_minutes

        return dashboard

    def _load_alerts(self) -> None:
        """Load safety alerts from storage."""
        try:
            alerts_file = self.storage_path / "alerts.json"
            if alerts_file.exists():
                with open(alerts_file, "r") as f:
                    alerts_data = json.load(f)

                self.alerts = [SafetyAlert.from_dict(alert) for alert in alerts_data]

                logger.info(f"Loaded {len(self.alerts)} safety alerts")
        except Exception as e:
            logger.error(f"Failed to load safety alerts: {e}")

    def _save_alerts(self) -> None:
        """Save safety alerts to storage."""
        try:
            alerts_file = self.storage_path / "alerts.json"
            alerts_data = [alert.to_dict() for alert in self.alerts]

            with open(alerts_file, "w") as f:
                json.dump(alerts_data, f, indent=2)

            logger.info(f"Saved {len(self.alerts)} safety alerts")
        except Exception as e:
            logger.error(f"Failed to save safety alerts: {e}")

    def create_profile(
        self,
        child_id: str,
        parent_id: str,
        child_name: str,
        age_group: ChildAgeGroup,
    ) -> ParentalControlProfile:
        """Create a new parental control profile for a child."""
        profile = ParentalControlProfile(
            child_id=child_id,
            parent_id=parent_id,
            child_name=child_name,
            age_group=age_group,
            created_at=time.time(),
            last_updated=time.time(),
        )

        # Set age-appropriate defaults
        self._set_age_appropriate_defaults(profile, age_group)

        self.profiles[child_id] = profile
        from datetime import datetime

        self.usage_stats[child_id] = UsageStats(
            child_id=child_id,
            device_id="default",  # Will be updated when session starts
            date=datetime.now().strftime("%Y-%m-%d"),
        )

        # Save to storage
        self._save_profiles()
        self._save_usage_stats()

        logger.info(
            f"Created parental control profile for child {child_name} ({child_id})"
        )

        return profile

    def _set_age_appropriate_defaults(
        self, profile: ParentalControlProfile, age_group: ChildAgeGroup
    ) -> None:
        """Set age-appropriate defaults for a profile."""
        from .types import TimeLimit, TimeLimitType

        if age_group == ChildAgeGroup.SCHOOL_AGE:
            # School age children: 2 hours daily limit
            time_limit = TimeLimit(
                limit_type=TimeLimitType.DAILY,
                duration_minutes=120,
                break_duration_minutes=30,
            )
            profile.time_limits = [time_limit]
            profile.content_filter.level = ContentFilterLevel.MODERATE
            profile.safety_level = "moderate"
        elif age_group == ChildAgeGroup.TODDLER:
            # Toddlers: 30 minutes daily limit, strict content filter
            time_limit = TimeLimit(
                limit_type=TimeLimitType.DAILY,
                duration_minutes=30,
                break_duration_minutes=15,
            )
            profile.time_limits = [time_limit]
            profile.content_filter.level = ContentFilterLevel.STRICT
            profile.safety_level = "strict"
        elif age_group == ChildAgeGroup.TEEN:
            # Teens: 3 hours daily limit, lenient content filter
            time_limit = TimeLimit(
                limit_type=TimeLimitType.DAILY,
                duration_minutes=180,
                break_duration_minutes=45,
            )
            profile.time_limits = [time_limit]
            profile.content_filter.level = ContentFilterLevel.LENIENT
            profile.safety_level = "lenient"
        else:
            # Default: 1 hour daily limit
            time_limit = TimeLimit(
                limit_type=TimeLimitType.DAILY,
                duration_minutes=60,
                break_duration_minutes=30,
            )
            profile.time_limits = [time_limit]
            profile.content_filter.level = ContentFilterLevel.MODERATE
            profile.safety_level = "moderate"

    def get_profile(self, child_id: str) -> Optional[ParentalControlProfile]:
        """Get parental control profile for a child."""
        if not self.profiles:
            self._load_profiles()

        return self.profiles.get(child_id)

    def update_profile(self, child_id: str, updates: Dict[str, Any]) -> bool:
        """Update parental control profile for a child."""
        profile = self.get_profile(child_id)
        if not profile:
            return False

        # Update fields
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        profile.update_last_updated()

        # Save to storage
        self._save_profiles()

        logger.info(f"Updated parental control profile for child {child_id}")

        return True

    def delete_profile(self, child_id: str) -> bool:
        """Delete parental control profile for a child."""
        if child_id not in self.profiles:
            return False

        del self.profiles[child_id]
        if child_id in self.usage_stats:
            del self.usage_stats[child_id]

        # Save to storage
        self._save_profiles()
        self._save_usage_stats()

        logger.info(f"Deleted parental control profile for child {child_id}")

        return True

    def check_time_limits(self, child_id: str) -> Tuple[bool, Optional[SafetyAlert]]:
        """Check if child has exceeded time limits."""
        profile = self.get_profile(child_id)
        if not profile:
            return True, None

        if not profile.time_limits:
            return True, None

        # Check if time limits are active
        if not profile.is_time_limit_active():
            return True, None

        # Get usage stats
        if not self.usage_stats:
            self._load_usage_stats()

        stats = self.usage_stats.get(child_id)
        if not stats:
            return True, None

        # Check daily limits
        active_limit = profile.get_active_time_limit()
        if active_limit and active_limit.limit_type == TimeLimitType.DAILY:
            if stats.total_time_minutes >= active_limit.duration_minutes:
                alert = SafetyAlert(
                    alert_type=AlertType.TIME_LIMIT_REACHED,
                    severity="medium",
                    message=f"Daily time limit exceeded ({active_limit.duration_minutes} minutes)",
                    child_id=child_id,
                    metadata={
                        "time_used": stats.total_time_minutes,
                        "time_limit": active_limit.duration_minutes,
                    },
                )
                return False, alert

        return True, None

    def record_interaction(
        self, child_id: str, character_id: str, content: str
    ) -> Tuple[bool, Optional[SafetyAlert]]:
        """Record a child interaction for monitoring."""
        if not self.usage_stats:
            self._load_usage_stats()

        if child_id not in self.usage_stats:
            from datetime import datetime

            self.usage_stats[child_id] = UsageStats(
                child_id=child_id,
                device_id="default",
                date=datetime.now().strftime("%Y-%m-%d"),
            )

        stats = self.usage_stats[child_id]
        stats.character_interactions += 1
        stats.last_activity = time.time()

        # Check content safety
        is_safe, safety_result, alert = self.check_content_safety(
            child_id=child_id, text=content, character_id=character_id
        )

        if not is_safe and alert:
            stats.safety_violations += 1
            self.alerts.append(alert)

        # Save stats
        self._save_usage_stats()

        # Log interaction if monitoring is enabled
        profile = self.get_profile(child_id)
        if profile and profile.log_all_interactions:
            logger.info(
                "Child interaction recorded",
                extra={
                    "child_id": child_id,
                    "character_id": character_id,
                    "content": content,
                },
            )

        return is_safe, alert

    def create_alert(
        self,
        child_id: str,
        alert_type: AlertType,
        message: str,
        severity: str = "medium",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SafetyAlert:
        """Create a new safety alert."""
        alert = SafetyAlert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            child_id=child_id,
            metadata=metadata or {},
        )

        self.alerts.append(alert)

        # Save alerts
        self._save_alerts()

        # Log alert
        logger.warning(
            "Safety alert created",
            extra={
                "child_id": child_id,
                "alert_type": alert_type.value,
                "severity": severity,
                "message": message,
            },
        )

        return alert

    def get_alerts(
        self, child_id: Optional[str] = None, acknowledged: Optional[bool] = None
    ) -> List[SafetyAlert]:
        """Get safety alerts, optionally filtered by child or acknowledgment status."""
        if not self.alerts:
            self._load_alerts()

        filtered_alerts = self.alerts

        if child_id:
            filtered_alerts = [a for a in filtered_alerts if a.child_id == child_id]

        if acknowledged is not None:
            filtered_alerts = [
                a for a in filtered_alerts if a.acknowledged == acknowledged
            ]

        return filtered_alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a safety alert."""
        if not self.alerts:
            self._load_alerts()

        for alert in self.alerts:
            if alert.timestamp == float(alert_id):  # Using timestamp as ID
                alert.acknowledged = True
                self._save_alerts()
                return True

        return False

    def get_usage_stats(self, child_id: str) -> Optional[UsageStats]:
        """Get usage statistics for a child."""
        if not self.usage_stats:
            self._load_usage_stats()

        return self.usage_stats.get(child_id)

    def get_all_profiles(self) -> Dict[str, ParentalControlProfile]:
        """Get all parental control profiles."""
        if not self.profiles:
            self._load_profiles()

        return self.profiles.copy()

    def cleanup_old_data(self, days: int = 30) -> None:
        """Clean up old data based on retention policies."""
        cutoff_time = time.time() - (days * 24 * 60 * 60)

        # Clean up old alerts
        if not self.alerts:
            self._load_alerts()

        original_count = len(self.alerts)
        self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        removed_count = original_count - len(self.alerts)

        if removed_count > 0:
            self._save_alerts()
            logger.info(f"Cleaned up {removed_count} old alerts")

        # Clean up old usage data
        if not self.usage_stats:
            self._load_usage_stats()

        for child_id, stats in self.usage_stats.items():
            # Clean up old daily usage data
            old_dates = [
                date
                for date in stats.daily_usage.keys()
                if time.mktime(time.strptime(date, "%Y-%m-%d")) < cutoff_time
            ]
            for date in old_dates:
                del stats.daily_usage[date]

        self._save_usage_stats()
        logger.info(f"Cleaned up old usage data for {len(self.usage_stats)} children")


def create_parental_control_manager(
    storage_path: Optional[Path] = None,
    metrics_collector: Optional[MetricsCollector] = None,
    log_aggregator: Optional[LogAggregator] = None,
) -> ParentalControlService:
    """Create a new parental control manager instance."""
    return ParentalControlService(
        storage_path=storage_path or Path.cwd() / "data/parental_controls",
        metrics_collector=metrics_collector,
        log_aggregator=log_aggregator,
    )
