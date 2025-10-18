"""
Prometheus metrics collection for the character.ai.

Provides comprehensive metrics for latency, system resources, and business logic
to enable production monitoring and alerting.
"""

from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

from .logging import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """Centralized metrics collection for the platform."""

    def __init__(self, registry: Optional[CollectorRegistry] = None) -> None:
        # Use a shared registry to avoid conflicts
        if registry is None:
            # Create a new registry for this instance to avoid conflicts
            registry = CollectorRegistry()
        self.registry = registry
        # Generate unique instance ID to avoid metric name conflicts
        import uuid

        self._instance_id = str(uuid.uuid4())[:8]
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize all Prometheus metrics."""

        # API Metrics
        self.api_requests_total = Counter(
            f"api_requests_total_{self._instance_id}",
            "Total number of API requests",
            ["method", "endpoint", "status_code"],
            registry=self.registry,
        )

        self.api_request_duration_seconds = Histogram(
            f"api_request_duration_seconds_{self._instance_id}",
            "API request duration in seconds",
            ["method", "endpoint"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry,
        )

        # Character Interaction Metrics
        self.character_interactions_total = Counter(
            f"character_interactions_total_{self._instance_id}",
            "Total character interactions",
            ["character_id", "interaction_type"],
            registry=self.registry,
        )

        # System Metrics
        self.system_memory_usage_bytes = Gauge(
            f"system_memory_usage_bytes_{self._instance_id}",
            "System memory usage in bytes",
            registry=self.registry,
        )

        self.system_cpu_usage_percent = Gauge(
            f"system_cpu_usage_percent_{self._instance_id}",
            "System CPU usage percentage",
            registry=self.registry,
        )

        # Error Metrics
        self.errors_total = Counter(
            f"errors_total_{self._instance_id}",
            "Total errors by component",
            ["component", "error_type"],
            registry=self.registry,
        )

        # Platform Info
        self.platform_info = Info(
            f"platform_info_{self._instance_id}",
            "Platform version and configuration information",
            registry=self.registry,
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


def create_metrics_collector() -> MetricsCollector:
    """Create a new metrics collector instance."""
    return MetricsCollector()
