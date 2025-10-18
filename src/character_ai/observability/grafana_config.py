"""
Grafana dashboard configuration and integration.

This module provides Grafana dashboard definitions, data source configurations,
and integration utilities for the Character AI.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .logging import get_logger

logger = get_logger(__name__)


class GrafanaDataSource(BaseModel):
    """Grafana data source configuration."""

    name: str
    type: str = "prometheus"
    url: str = "http://localhost:9090"
    access: str = "proxy"
    is_default: bool = True
    json_data: Dict[str, Any] = Field(default_factory=dict)
    secure_json_data: Dict[str, Any] = Field(default_factory=dict)


class GrafanaPanel(BaseModel):
    """Grafana panel configuration."""

    id: int
    title: str
    type: str
    targets: List[Dict[str, Any]]
    grid_pos: Dict[str, int]
    options: Dict[str, Any] = Field(default_factory=dict)
    field_config: Dict[str, Any] = Field(default_factory=dict)
    transformations: List[Dict[str, Any]] = Field(default_factory=list)


class GrafanaDashboard(BaseModel):
    """Grafana dashboard configuration."""

    title: str
    description: str
    tags: List[str] = Field(default_factory=list)
    panels: List[GrafanaPanel] = Field(default_factory=list)
    time_range: Dict[str, str] = Field(
        default_factory=lambda: {"from": "now-1h", "to": "now"}
    )
    refresh: str = "30s"
    schema_version: int = 30
    version: int = 1
    uid: str = Field(default_factory=lambda: f"icp-{int(time.time())}")


class GrafanaIntegration:
    """Grafana integration utilities."""

    def __init__(
        self, grafana_url: str = "http://localhost:3000", api_key: Optional[str] = None
    ):
        self.grafana_url = grafana_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}" if api_key else None,
            "Content-Type": "application/json",
        }
        logger.info("GrafanaIntegration initialized", grafana_url=grafana_url)

    def create_icp_dashboard(self) -> GrafanaDashboard:
        """Create the main CAI dashboard."""
        dashboard = GrafanaDashboard(
            title="Character AI",
            description="Comprehensive monitoring dashboard for CAI",
            tags=["icp", "monitoring", "ai", "character"],
            time_range={"from": "now-1h", "to": "now"},
            refresh="30s",
        )

        # Add panels
        dashboard.panels = [
            self._create_system_overview_panel(),
            self._create_api_metrics_panel(),
            self._create_character_interactions_panel(),
            self._create_safety_events_panel(),
            self._create_performance_metrics_panel(),
            self._create_error_rates_panel(),
            self._create_log_analysis_panel(),
            self._create_resource_usage_panel(),
        ]

        return dashboard

    def _create_system_overview_panel(self) -> GrafanaPanel:
        """Create system overview panel."""
        return GrafanaPanel(
            id=1,
            title="System Overview",
            type="stat",
            grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
            targets=[
                {
                    "expr": "icp_system_uptime_seconds",
                    "refId": "A",
                    "legendFormat": "Uptime",
                }
            ],
            options={
                "colorMode": "value",
                "graphMode": "area",
                "justifyMode": "auto",
                "orientation": "auto",
            },
        )

    def _create_api_metrics_panel(self) -> GrafanaPanel:
        """Create API metrics panel."""
        return GrafanaPanel(
            id=2,
            title="API Request Metrics",
            type="timeseries",
            grid_pos={"x": 12, "y": 0, "w": 12, "h": 8},
            targets=[
                {
                    "expr": "rate(icp_api_requests_total[5m])",
                    "refId": "A",
                    "legendFormat": "Requests/sec",
                },
                {
                    "expr": "histogram_quantile(0.95, rate(icp_api_request_duration_seconds_bucket[5m]))",
                    "refId": "B",
                    "legendFormat": "P95 Latency",
                },
            ],
            options={
                "legend": {"displayMode": "table", "placement": "bottom"},
                "tooltip": {"mode": "single"},
            },
        )

    def _create_character_interactions_panel(self) -> GrafanaPanel:
        """Create character interactions panel."""
        return GrafanaPanel(
            id=3,
            title="Character Interactions",
            type="timeseries",
            grid_pos={"x": 0, "y": 8, "w": 12, "h": 8},
            targets=[
                {
                    "expr": "rate(icp_character_interactions_total[5m])",
                    "refId": "A",
                    "legendFormat": "Interactions/sec",
                },
                {
                    "expr": "icp_character_interactions_total",
                    "refId": "B",
                    "legendFormat": "Total Interactions",
                },
            ],
            options={"legend": {"displayMode": "table", "placement": "bottom"}},
        )

    def _create_safety_events_panel(self) -> GrafanaPanel:
        """Create safety events panel."""
        return GrafanaPanel(
            id=4,
            title="Safety Events",
            type="timeseries",
            grid_pos={"x": 12, "y": 8, "w": 12, "h": 8},
            targets=[
                {
                    "expr": "rate(icp_safety_events_total[5m])",
                    "refId": "A",
                    "legendFormat": "Safety Events/sec",
                },
                {
                    "expr": "sum by (severity) (rate(icp_safety_events_total[5m]))",
                    "refId": "B",
                    "legendFormat": "{{severity}}",
                },
            ],
            options={"legend": {"displayMode": "table", "placement": "bottom"}},
        )

    def _create_performance_metrics_panel(self) -> GrafanaPanel:
        """Create performance metrics panel."""
        return GrafanaPanel(
            id=5,
            title="Performance Metrics",
            type="timeseries",
            grid_pos={"x": 0, "y": 16, "w": 12, "h": 8},
            targets=[
                {
                    "expr": "histogram_quantile(0.50, rate(icp_character_processing_duration_seconds_bucket[5m]))",
                    "refId": "A",
                    "legendFormat": "P50 Processing Time",
                },
                {
                    "expr": "histogram_quantile(0.95, rate(icp_character_processing_duration_seconds_bucket[5m]))",
                    "refId": "B",
                    "legendFormat": "P95 Processing Time",
                },
                {
                    "expr": "histogram_quantile(0.99, rate(icp_character_processing_duration_seconds_bucket[5m]))",
                    "refId": "C",
                    "legendFormat": "P99 Processing Time",
                },
            ],
            options={"legend": {"displayMode": "table", "placement": "bottom"}},
        )

    def _create_error_rates_panel(self) -> GrafanaPanel:
        """Create error rates panel."""
        return GrafanaPanel(
            id=6,
            title="Error Rates",
            type="timeseries",
            grid_pos={"x": 12, "y": 16, "w": 12, "h": 8},
            targets=[
                {
                    "expr": 'rate(icp_api_requests_total{status_code=~"5.."}[5m])',
                    "refId": "A",
                    "legendFormat": "5xx Errors/sec",
                },
                {
                    "expr": "rate(icp_crashes_total[5m])",
                    "refId": "B",
                    "legendFormat": "Crashes/sec",
                },
            ],
            options={"legend": {"displayMode": "table", "placement": "bottom"}},
        )

    def _create_log_analysis_panel(self) -> GrafanaPanel:
        """Create log analysis panel."""
        return GrafanaPanel(
            id=7,
            title="Log Analysis",
            type="timeseries",
            grid_pos={"x": 0, "y": 24, "w": 12, "h": 8},
            targets=[
                {
                    "expr": "rate(icp_log_entries_total[5m])",
                    "refId": "A",
                    "legendFormat": "Log Entries/sec",
                },
                {
                    "expr": "sum by (level) (rate(icp_log_entries_total[5m]))",
                    "refId": "B",
                    "legendFormat": "{{level}}",
                },
            ],
            options={"legend": {"displayMode": "table", "placement": "bottom"}},
        )

    def _create_resource_usage_panel(self) -> GrafanaPanel:
        """Create resource usage panel."""
        return GrafanaPanel(
            id=8,
            title="Resource Usage",
            type="timeseries",
            grid_pos={"x": 12, "y": 24, "w": 12, "h": 8},
            targets=[
                {
                    "expr": "icp_system_memory_usage_bytes",
                    "refId": "A",
                    "legendFormat": "Memory Usage",
                },
                {
                    "expr": "icp_system_cpu_usage_percent",
                    "refId": "B",
                    "legendFormat": "CPU Usage %",
                },
            ],
            options={"legend": {"displayMode": "table", "placement": "bottom"}},
        )

    def export_dashboard_json(
        self, dashboard: GrafanaDashboard, output_path: Path
    ) -> None:
        """Export dashboard to JSON file."""
        dashboard_data = {
            "dashboard": dashboard.model_dump(),
            "overwrite": True,
            "message": f"Updated CAI dashboard at {datetime.now().isoformat()}",
        }

        with open(output_path, "w") as f:
            json.dump(dashboard_data, f, indent=2)

        logger.info("Dashboard exported", output_path=str(output_path))

    def create_alert_rules(self) -> List[Dict[str, Any]]:
        """Create Grafana alert rules for CAI."""
        return [
            {
                "alert": "HighErrorRate",
                "expr": 'rate(icp_api_requests_total{status_code=~"5.."}[5m]) > 0.1',
                "for": "2m",
                "labels": {"severity": "warning", "service": "icp"},
                "annotations": {
                    "summary": "High error rate detected",
                    "description": "Error rate is above 0.1 requests/sec for 2 minutes",
                },
            },
            {
                "alert": "HighLatency",
                "expr": "histogram_quantile(0.95, rate(icp_api_request_duration_seconds_bucket[5m])) > 2.0",
                "for": "1m",
                "labels": {"severity": "warning", "service": "icp"},
                "annotations": {
                    "summary": "High latency detected",
                    "description": "P95 latency is above 2 seconds",
                },
            },
            {
                "alert": "SystemCrash",
                "expr": "increase(icp_crashes_total[1m]) > 0",
                "for": "0m",
                "labels": {"severity": "critical", "service": "icp"},
                "annotations": {
                    "summary": "System crash detected",
                    "description": "Application crash detected",
                },
            },
            {
                "alert": "HighMemoryUsage",
                "expr": "icp_system_memory_usage_bytes > 8e9",
                "for": "5m",
                "labels": {"severity": "warning", "service": "icp"},
                "annotations": {
                    "summary": "High memory usage",
                    "description": "Memory usage is above 8GB for 5 minutes",
                },
            },
        ]


def create_icp_grafana_dashboard() -> GrafanaDashboard:
    """Create the main CAI Grafana dashboard."""
    integration = GrafanaIntegration()
    return integration.create_icp_dashboard()


def export_grafana_config(output_dir: Path = Path.cwd() / "monitoring/grafana") -> None:
    """Export Grafana configuration files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dashboard
    dashboard = create_icp_grafana_dashboard()
    integration = GrafanaIntegration()

    # Export dashboard JSON
    dashboard_path = output_dir / "icp-dashboard.json"
    integration.export_dashboard_json(dashboard, dashboard_path)

    # Export alert rules
    alert_rules = integration.create_alert_rules()
    alerts_path = output_dir / "alert-rules.json"
    with open(alerts_path, "w") as f:
        json.dump(alert_rules, f, indent=2)

    # Export data source configuration
    data_source = GrafanaDataSource(name="CAI Prometheus", url="http://localhost:9090")
    ds_path = output_dir / "datasource.json"
    with open(ds_path, "w") as f:
        json.dump(data_source.model_dump(), f, indent=2)

    logger.info("Grafana configuration exported", output_dir=str(output_dir))
