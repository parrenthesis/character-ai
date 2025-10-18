"""
Observability package for logging, metrics, monitoring, and crash reporting.

Consolidates all observability functionality including:
- Core logging and metrics infrastructure
- Prometheus, Grafana, and ELK integration
- Crash reporting and log aggregation
"""

from .crash_reporting import create_crash_reporter, get_health_status, report_error
from .grafana_config import (
    GrafanaDashboard,
    create_icp_grafana_dashboard,
    export_grafana_config,
)
from .log_aggregation import (
    LogAggregator,
    LogSearchQuery,
    LogSource,
    create_log_aggregator,
)
from .logging import (
    ProcessingTimer,
    clear_request_context,
    configure_logging,
    generate_request_id,
    generate_trace_id,
    get_logger,
    set_request_context,
)
from .metrics import MetricsCollector, create_metrics_collector
from .prometheus_export import (
    PrometheusExporter,
    create_prometheus_exporter,
    export_prometheus_config,
)

__all__ = [
    # Core logging and metrics
    "ProcessingTimer",
    "get_logger",
    "configure_logging",
    "set_request_context",
    "clear_request_context",
    "generate_request_id",
    "generate_trace_id",
    "MetricsCollector",
    "create_metrics_collector",
    # Crash reporting and log aggregation
    "report_error",
    "create_crash_reporter",
    "get_health_status",
    "LogAggregator",
    "LogSearchQuery",
    "LogSource",
    "create_log_aggregator",
    # Monitoring integrations
    "PrometheusExporter",
    "create_prometheus_exporter",
    "export_prometheus_config",
    "GrafanaDashboard",
    "create_icp_grafana_dashboard",
    "export_grafana_config",
]
