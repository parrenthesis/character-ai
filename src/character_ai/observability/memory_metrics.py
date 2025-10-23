# mypy: ignore-errors
"""
Prometheus metrics for hybrid memory system monitoring.

Tracks memory system performance, usage patterns, and health metrics
for operational monitoring and alerting.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Try to import prometheus_client, fallback to mock classes if not available
try:
    from prometheus_client import Counter, Gauge, Histogram, Info  # type: ignore

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Mock classes for environments without prometheus_client
    class Counter:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            pass

        def inc(self, *args, **kwargs) -> None:
            pass

        def labels(self, *args, **kwargs) -> "Counter":
            return self

    class Histogram:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            pass

        def observe(self, *args, **kwargs) -> None:
            pass

        def time(self) -> "Histogram":
            return self

        def __enter__(self) -> "Histogram":
            return self

        def __exit__(self, *args) -> None:
            pass

    class Gauge:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set(self, *args, **kwargs) -> None:
            pass

        def inc(self, *args, **kwargs) -> None:
            pass

        def dec(self, *args, **kwargs) -> None:
            pass

        def labels(self, *args, **kwargs) -> "Gauge":
            return self

    class Info:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            pass

        def info(self, *args, **kwargs) -> None:
            pass


class MemoryMetrics:
    """Prometheus metrics for hybrid memory system."""

    def __init__(self) -> None:
        """Initialize memory system metrics."""
        # Preference extraction metrics
        self.preferences_extracted = Counter(
            "memory_preferences_extracted_total",
            "Total number of preferences extracted",
            ["preference_type", "extraction_method"],
        )

        # Conversation storage metrics
        self.turns_stored = Counter(
            "memory_turns_stored_total",
            "Total number of conversation turns stored",
            ["character_name"],
        )

        self.sessions_created = Counter(
            "memory_sessions_created_total",
            "Total number of sessions created",
            ["character_name"],
        )

        # Database operation metrics
        self.db_operations = Counter(
            "memory_db_operations_total",
            "Total database operations",
            ["operation_type", "status"],
        )

        self.db_operation_duration = Histogram(
            "memory_db_operation_duration_seconds",
            "Database operation duration",
            ["operation_type"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        )

        # Summarization metrics
        self.summaries_created = Counter(
            "memory_summaries_created_total",
            "Total number of summaries created",
            ["character_name"],
        )

        self.summarization_duration = Histogram(
            "memory_summarization_duration_seconds",
            "Summarization duration",
            ["character_name"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        )

        # Context building metrics
        self.context_builds = Counter(
            "memory_context_builds_total", "Total context builds", ["character_name"]
        )

        self.context_build_duration = Histogram(
            "memory_context_build_duration_seconds",
            "Context build duration",
            ["character_name"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        )

        # Memory system health metrics
        self.active_sessions = Gauge(
            "memory_active_sessions", "Number of active sessions", ["character_name"]
        )

        self.total_turns = Gauge(
            "memory_total_turns", "Total number of stored turns", ["character_name"]
        )

        self.database_size_bytes = Gauge(
            "memory_database_size_bytes", "Database size in bytes"
        )

        # System info
        self.memory_system_info = Info(
            "memory_system_info", "Memory system information"
        )

        # Set system info
        self.memory_system_info.info(
            {"version": "1.0.0", "features": "preferences,storage,summarization"}
        )

    def record_preference_extraction(
        self, preference_type: str, method: str = "pattern"
    ) -> None:
        """Record a preference extraction event."""
        self.preferences_extracted.labels(
            preference_type=preference_type, extraction_method=method
        ).inc()

    def record_turn_stored(self, character_name: str) -> None:
        """Record a turn storage event."""
        self.turns_stored.labels(character_name=character_name).inc()

    def record_session_created(self, character_name: str) -> None:
        """Record a session creation event."""
        self.sessions_created.labels(character_name=character_name).inc()

    def record_db_operation(
        self, operation_type: str, status: str, duration: float
    ) -> None:
        """Record a database operation."""
        self.db_operations.labels(operation_type=operation_type, status=status).inc()
        self.db_operation_duration.labels(operation_type=operation_type).observe(
            duration
        )

    def record_summarization(self, character_name: str, duration: float) -> None:
        """Record a summarization event."""
        self.summaries_created.labels(character_name=character_name).inc()
        self.summarization_duration.labels(character_name=character_name).observe(
            duration
        )

    def record_context_build(self, character_name: str, duration: float) -> None:
        """Record a context build event."""
        self.context_builds.labels(character_name=character_name).inc()
        self.context_build_duration.labels(character_name=character_name).observe(
            duration
        )

    def update_active_sessions(self, character_name: str, count: int) -> None:
        """Update active sessions count."""
        self.active_sessions.labels(character_name=character_name).set(count)

    def update_total_turns(self, character_name: str, count: int) -> None:
        """Update total turns count."""
        self.total_turns.labels(character_name=character_name).set(count)

    def update_database_size(self, size_bytes: int) -> None:
        """Update database size."""
        self.database_size_bytes.set(size_bytes)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        return {
            "active_sessions": getattr(self.active_sessions, "_value", 0),
            "total_turns": getattr(self.total_turns, "_value", 0),
            "database_size_bytes": getattr(self.database_size_bytes, "_value", 0),
        }


# Global metrics instance - use lazy initialization to avoid duplicate registration
_memory_metrics_instance = None


def get_memory_metrics() -> MemoryMetrics:
    """Get the global memory metrics instance."""
    global _memory_metrics_instance
    if _memory_metrics_instance is None:
        _memory_metrics_instance = MemoryMetrics()
    return _memory_metrics_instance
