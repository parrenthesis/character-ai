"""
Prometheus metrics export and alerting.

This module provides Prometheus metrics export, alerting rules,
and integration with external monitoring systems.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from pydantic import BaseModel, Field

from .logging import get_logger

logger = get_logger(__name__)


class PrometheusAlert(BaseModel):
    """Prometheus alert rule."""

    alert: str
    expr: str
    for_: str = Field(alias="for")
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)

    class Config:
        validate_by_name = True


class PrometheusRule(BaseModel):
    """Prometheus rule group."""

    name: str
    rules: List[PrometheusAlert]
    interval: str = "30s"


class PrometheusExporter:
    """Prometheus metrics exporter with custom metrics."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()
        logger.info("PrometheusExporter initialized")

    def _setup_metrics(self) -> None:
        """Set up Prometheus metrics."""
        # System metrics
        self.system_uptime = Gauge(
            "icp_system_uptime_seconds",
            "System uptime in seconds",
            registry=self.registry,
        )

        self.system_memory_usage = Gauge(
            "icp_system_memory_usage_bytes",
            "System memory usage in bytes",
            registry=self.registry,
        )

        self.system_cpu_usage = Gauge(
            "icp_system_cpu_usage_percent",
            "System CPU usage percentage",
            registry=self.registry,
        )

        # API metrics
        self.api_requests_total = Counter(
            "icp_api_requests_total",
            "Total API requests",
            ["method", "endpoint", "status_code"],
            registry=self.registry,
        )

        self.api_request_duration = Histogram(
            "icp_api_request_duration_seconds",
            "API request duration in seconds",
            ["method", "endpoint"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry,
        )

        # Character interaction metrics
        self.character_interactions_total = Counter(
            "icp_character_interactions_total",
            "Total character interactions",
            ["character_id", "interaction_type"],
            registry=self.registry,
        )

        self.character_processing_duration = Histogram(
            "icp_character_processing_duration_seconds",
            "Character processing duration in seconds",
            ["character_id", "component"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry,
        )

        # Safety metrics
        self.safety_events_total = Counter(
            "icp_safety_events_total",
            "Total safety events",
            ["event_type", "severity"],
            registry=self.registry,
        )

        self.safety_confidence = Histogram(
            "icp_safety_confidence_score",
            "Safety confidence scores",
            ["event_type"],
            buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            registry=self.registry,
        )

        # Log metrics
        self.log_entries_total = Counter(
            "icp_log_entries_total",
            "Total log entries",
            ["level", "source"],
            registry=self.registry,
        )

        # Error metrics
        self.crashes_total = Counter(
            "icp_crashes_total",
            "Total application crashes",
            ["error_type", "component"],
            registry=self.registry,
        )

        self.errors_total = Counter(
            "icp_errors_total",
            "Total errors",
            ["error_type", "component"],
            registry=self.registry,
        )

        # Performance metrics
        self.performance_latency = Histogram(
            "icp_performance_latency_seconds",
            "Performance latency in seconds",
            ["operation", "component"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry,
        )

        # Streaming metrics
        self.streaming_audio_duration = Histogram(
            "icp_streaming_audio_duration_seconds",
            "Streaming audio processing duration",
            ["component"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry,
        )

        self.streaming_llm_tokens = Counter(
            "icp_streaming_llm_tokens_total",
            "Total streaming LLM tokens generated",
            ["model", "session_id"],
            registry=self.registry,
        )

        # Version info
        self.version_info = Info(
            "icp_version", "CAI version information", registry=self.registry
        )
        self.version_info.info(
            {
                "version": "1.0.0",
                "build_date": datetime.now().isoformat(),
                "python_version": "3.10+",
            }
        )

    def update_system_metrics(
        self, uptime: float, memory_bytes: int, cpu_percent: float
    ) -> None:
        """Update system metrics."""
        self.system_uptime.set(uptime)
        self.system_memory_usage.set(memory_bytes)
        self.system_cpu_usage.set(cpu_percent)

    def record_api_request(
        self, method: str, endpoint: str, status_code: int, duration: float
    ) -> None:
        """Record API request metrics."""
        self.api_requests_total.labels(
            method=method, endpoint=endpoint, status_code=str(status_code)
        ).inc()

        self.api_request_duration.labels(method=method, endpoint=endpoint).observe(
            duration
        )

    def record_character_interaction(
        self, character_id: str, interaction_type: str, duration: float, component: str
    ) -> None:
        """Record character interaction metrics."""
        self.character_interactions_total.labels(
            character_id=character_id, interaction_type=interaction_type
        ).inc()

        self.character_processing_duration.labels(
            character_id=character_id, component=component
        ).observe(duration)

    def record_safety_event(
        self, event_type: str, severity: str, confidence: float
    ) -> None:
        """Record safety event metrics."""
        self.safety_events_total.labels(event_type=event_type, severity=severity).inc()

        self.safety_confidence.labels(event_type=event_type).observe(confidence)

    def record_log_entry(self, level: str, source: str) -> None:
        """Record log entry metrics."""
        self.log_entries_total.labels(level=level, source=source).inc()

    def record_crash(self, error_type: str, component: str) -> None:
        """Record crash metrics."""
        self.crashes_total.labels(error_type=error_type, component=component).inc()

    def record_error(self, error_type: str, component: str) -> None:
        """Record error metrics."""
        self.errors_total.labels(error_type=error_type, component=component).inc()

    def record_performance(
        self, operation: str, component: str, latency: float
    ) -> None:
        """Record performance metrics."""
        self.performance_latency.labels(
            operation=operation, component=component
        ).observe(latency)

    def record_streaming_audio(self, component: str, duration: float) -> None:
        """Record streaming audio metrics."""
        self.streaming_audio_duration.labels(component=component).observe(duration)

    def record_streaming_llm_tokens(
        self, model: str, session_id: str, token_count: int
    ) -> None:
        """Record streaming LLM token metrics."""
        self.streaming_llm_tokens.labels(model=model, session_id=session_id).inc(
            token_count
        )

    def generate_metrics(self) -> str:
        """Generate Prometheus metrics output."""
        return generate_latest(self.registry).decode("utf-8")

    def create_alert_rules(self) -> List[PrometheusRule]:
        """Create Prometheus alert rules for CAI."""
        return [
            PrometheusRule(
                name="icp-api-alerts",
                rules=[
                    PrometheusAlert(
                        alert="HighErrorRate",
                        expr='rate(icp_api_requests_total{status_code=~"5.."}[5m]) > 0.1',
                        **{"for": "2m"},
                        labels={"severity": "warning", "service": "icp"},
                        annotations={
                            "summary": "High error rate detected",
                            "description": "Error rate is above 0.1 requests/sec for 2 minutes",
                        },
                    ),
                    PrometheusAlert(
                        alert="HighLatency",
                        expr="histogram_quantile(0.95, rate(icp_api_request_duration_seconds_bucket[5m])) > 2.0",
                        **{"for": "1m"},
                        labels={"severity": "warning", "service": "icp"},
                        annotations={
                            "summary": "High latency detected",
                            "description": "P95 latency is above 2 seconds",
                        },
                    ),
                ],
            ),
            PrometheusRule(
                name="icp-system-alerts",
                rules=[
                    PrometheusAlert(
                        alert="SystemCrash",
                        expr="increase(icp_crashes_total[1m]) > 0",
                        **{"for": "0m"},
                        labels={"severity": "critical", "service": "icp"},
                        annotations={
                            "summary": "System crash detected",
                            "description": "Application crash detected",
                        },
                    ),
                    PrometheusAlert(
                        alert="HighMemoryUsage",
                        expr="icp_system_memory_usage_bytes > 8e9",
                        **{"for": "5m"},
                        labels={"severity": "warning", "service": "icp"},
                        annotations={
                            "summary": "High memory usage",
                            "description": "Memory usage is above 8GB for 5 minutes",
                        },
                    ),
                    PrometheusAlert(
                        alert="HighCPUUsage",
                        expr="icp_system_cpu_usage_percent > 80",
                        **{"for": "5m"},
                        labels={"severity": "warning", "service": "icp"},
                        annotations={
                            "summary": "High CPU usage",
                            "description": "CPU usage is above 80% for 5 minutes",
                        },
                    ),
                ],
            ),
            PrometheusRule(
                name="icp-safety-alerts",
                rules=[
                    PrometheusAlert(
                        alert="HighSafetyEvents",
                        expr="rate(icp_safety_events_total[5m]) > 0.5",
                        **{"for": "2m"},
                        labels={"severity": "warning", "service": "icp"},
                        annotations={
                            "summary": "High safety event rate",
                            "description": "Safety event rate is above 0.5 events/sec for 2 minutes",
                        },
                    ),
                    PrometheusAlert(
                        alert="LowSafetyConfidence",
                        expr="histogram_quantile(0.5, rate(icp_safety_confidence_score_bucket[5m])) < 0.7",
                        **{"for": "5m"},
                        labels={"severity": "warning", "service": "icp"},
                        annotations={
                            "summary": "Low safety confidence",
                            "description": "Safety confidence is below 0.7 for 5 minutes",
                        },
                    ),
                ],
            ),
            PrometheusRule(
                name="icp-performance-alerts",
                rules=[
                    PrometheusAlert(
                        alert="HighProcessingLatency",
                        expr="histogram_quantile(0.95, rate(icp_character_processing_duration_seconds_bucket[5m])) > 5.0",
                        **{"for": "2m"},
                        labels={"severity": "warning", "service": "icp"},
                        annotations={
                            "summary": "High processing latency",
                            "description": "P95 processing latency is above 5 seconds for 2 minutes",
                        },
                    ),
                    PrometheusAlert(
                        alert="LowThroughput",
                        expr="rate(icp_character_interactions_total[5m]) < 0.1",
                        **{"for": "10m"},
                        labels={"severity": "warning", "service": "icp"},
                        annotations={
                            "summary": "Low interaction throughput",
                            "description": "Character interaction rate is below 0.1 interactions/sec for 10 minutes",
                        },
                    ),
                ],
            ),
        ]

    def export_alert_rules(self, output_path: Path) -> bool:
        """Export Prometheus alert rules to file."""
        try:
            rules = self.create_alert_rules()
            rules_data = {
                "groups": [
                    {
                        "name": rule.name,
                        "interval": rule.interval,
                        "rules": [
                            {
                                "alert": alert.alert,
                                "expr": alert.expr,
                                "for": alert.for_,
                                "labels": alert.labels,
                                "annotations": alert.annotations,
                            }
                            for alert in rule.rules
                        ],
                    }
                    for rule in rules
                ]
            }

            with open(output_path, "w") as f:
                json.dump(rules_data, f, indent=2)

            logger.info("Alert rules exported", output_path=str(output_path))
            return True
        except Exception as e:
            logger.error(f"Failed to export alert rules: {e}")
            return False

    def export_metrics_config(self, output_path: Path) -> bool:
        """Export Prometheus metrics configuration."""
        try:
            config = {
                "global": {"scrape_interval": "15s", "evaluation_interval": "15s"},
                "rule_files": ["icp-alerts.yml"],
                "scrape_configs": [
                    {
                        "job_name": "icp",
                        "static_configs": [{"targets": ["localhost:8000"]}],
                        "metrics_path": "/metrics",
                        "scrape_interval": "15s",
                    }
                ],
                "alerting": {
                    "alertmanagers": [
                        {"static_configs": [{"targets": ["localhost:9093"]}]}
                    ]
                },
            }

            with open(output_path, "w") as f:
                json.dump(config, f, indent=2)

            logger.info("Metrics configuration exported", output_path=str(output_path))
            return True
        except Exception as e:
            logger.error(f"Failed to export metrics configuration: {e}")
            return False


def create_prometheus_exporter() -> PrometheusExporter:
    """Create a Prometheus metrics exporter."""
    return PrometheusExporter()


def export_prometheus_config(
    output_dir: Path = Path.cwd() / "monitoring/prometheus",
) -> bool:
    """Export Prometheus configuration files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        exporter = create_prometheus_exporter()

        # Export alert rules
        alerts_path = output_dir / "icp-alerts.yml"
        exporter.export_alert_rules(alerts_path)

        # Export metrics configuration
        config_path = output_dir / "prometheus.yml"
        exporter.export_metrics_config(config_path)

        logger.info("Prometheus configuration exported", output_dir=str(output_dir))
        return True
    except Exception as e:
        logger.error(f"Failed to export Prometheus configuration: {e}")
        return False
