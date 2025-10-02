"""
Prometheus metrics collection for the character.ai.

Provides comprehensive metrics for latency, system resources, and business logic
to enable production monitoring and alerting.
"""

from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore
from prometheus_client import REGISTRY, Counter, Gauge, Histogram, Info, generate_latest

from .logging import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """Centralized metrics collection for the platform."""

    _instance = None
    _initialized = False

    def __new__(cls) -> "MetricsCollector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not MetricsCollector._initialized:
            self.registry = REGISTRY
            self._initialize_metrics()
            MetricsCollector._initialized = True

    def _initialize_metrics(self) -> None:
        """Initialize all Prometheus metrics."""

        # API Metrics
        self.api_requests_total = Counter(
            "api_requests_total",
            "Total number of API requests",
            ["method", "endpoint", "status_code"],
        )

        self.api_request_duration_seconds = Histogram(
            "api_request_duration_seconds",
            "API request duration in seconds",
            ["method", "endpoint"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        # Character Interaction Metrics
        self.character_interactions_total = Counter(
            "character_interactions_total",
            "Total character interactions",
            ["character_id", "interaction_type"],
        )

        # System Metrics
        self.system_memory_usage_bytes = Gauge(
            "system_memory_usage_bytes", "System memory usage in bytes"
        )

        self.system_cpu_usage_percent = Gauge(
            "system_cpu_usage_percent", "System CPU usage percentage"
        )

        # Error Metrics
        self.errors_total = Counter(
            "errors_total", "Total errors by component", ["component", "error_type"]
        )

        # Platform Info
        self.platform_info = Info(
            "platform_info", "Platform version and configuration information"
        )

        self.platform_info.info({"version": "1.0.0", "component": "character.ai"})

    def record_api_request(
        self, method: str, endpoint: str, status_code: int, duration: float
    ) -> None:
        """Record API request metrics."""
        self.api_requests_total.labels(
            method=method, endpoint=endpoint, status_code=str(status_code)
        ).inc()

        self.api_request_duration_seconds.labels(
            method=method, endpoint=endpoint
        ).observe(duration)

    def record_character_interaction(
        self,
        character_id: str,
        interaction_type: str,
        duration: float,
        component: str = "unknown",
    ) -> None:
        """Record character interaction metrics."""
        self.character_interactions_total.labels(
            character_id=character_id, interaction_type=interaction_type
        ).inc()

    def record_error(self, component: str, error_type: str) -> None:
        """Record error metrics."""
        self.errors_total.labels(component=component, error_type=error_type).inc()

    def update_system_metrics(self) -> None:
        """Update system resource metrics."""
        try:
            if psutil:
                # Memory usage
                memory = psutil.virtual_memory()
                self.system_memory_usage_bytes.set(memory.used)

                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.system_cpu_usage_percent.set(cpu_percent)
            else:
                self.system_memory_usage_bytes.set(0)
                self.system_cpu_usage_percent.set(0.0)

        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")

    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in text format."""
        # Update system metrics before returning
        self.update_system_metrics()
        return generate_latest(self.registry)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
