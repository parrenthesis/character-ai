"""
Monitoring API endpoints for external monitoring integration.

This module provides REST API endpoints for Grafana, ELK stack,
and Prometheus integration with the Character AI.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from ...observability import (
    create_icp_grafana_dashboard,
    create_metrics_collector,
    create_prometheus_exporter,
    export_grafana_config,
    export_prometheus_config,
    get_logger,
)
from ...observability.log_aggregation import (
    LogLevel,
    LogSearchQuery,
    LogSource,
    create_log_aggregator,
)

logger = get_logger(__name__)

# Create monitoring router
monitoring_router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


@monitoring_router.get("/metrics")
async def get_prometheus_metrics() -> PlainTextResponse:
    """Get Prometheus metrics in text format."""
    try:
        exporter = create_prometheus_exporter()
        metrics_output = exporter.generate_metrics()
        return PlainTextResponse(content=metrics_output, media_type="text/plain")
    except Exception as e:
        logger.error(f"Failed to get Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.get("/grafana/dashboard")
async def get_grafana_dashboard() -> Dict[str, Any]:
    """Get Grafana dashboard configuration."""
    try:
        dashboard = create_icp_grafana_dashboard()
        return dashboard.model_dump()
    except Exception as e:
        logger.error(f"Failed to get Grafana dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.post("/grafana/export")
async def export_grafana_configuration(
    output_dir: str = "monitoring/grafana",
) -> Dict[str, str]:
    """Export Grafana configuration files."""
    try:
        output_path = Path.cwd() / output_dir
        export_grafana_config(output_path)
        return {"message": f"Grafana configuration exported to {output_dir}"}
    except Exception as e:
        logger.error(f"Failed to export Grafana configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.get("/prometheus/config")
async def get_prometheus_config() -> Dict[str, Any]:
    """Get Prometheus configuration."""
    try:
        exporter = create_prometheus_exporter()
        rules = exporter.create_alert_rules()

        return {
            "alert_rules": [rule.model_dump() for rule in rules],
            "scrape_config": {
                "job_name": "icp",
                "static_configs": [{"targets": ["localhost:8000"]}],
                "metrics_path": "/api/v1/monitoring/metrics",
                "scrape_interval": "15s",
            },
        }
    except Exception as e:
        logger.error(f"Failed to get Prometheus configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.post("/prometheus/export")
async def export_prometheus_configuration(
    output_dir: str = "monitoring/prometheus",
) -> Dict[str, str]:
    """Export Prometheus configuration files."""
    try:
        output_path = Path.cwd() / output_dir
        success = export_prometheus_config(output_path)

        if success:
            return {"message": f"Prometheus configuration exported to {output_dir}"}
        else:
            raise HTTPException(
                status_code=500, detail="Failed to export Prometheus configuration"
            )
    except Exception as e:
        logger.error(f"Failed to export Prometheus configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.get("/logs/search")
async def search_logs(
    query: Optional[str] = Query(None, description="Search query"),
    level: Optional[LogLevel] = Query(None, description="Log level filter"),
    source: Optional[LogSource] = Query(None, description="Log source filter"),
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
) -> Dict[str, Any]:
    """Search logs using the log aggregation system."""
    try:
        aggregator = create_log_aggregator()

        search_query = LogSearchQuery(
            message_pattern=query,
            levels=[level] if level else None,
            sources=[source] if source else None,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset,
        )

        result = await aggregator.search_logs(search_query)

        return {
            "entries": [entry.dict() for entry in result.entries],  # type: ignore
            "total_hits": result.total_count,
            "limit": result.query.limit,
            "offset": result.query.offset,
            "execution_time_ms": result.execution_time_ms,
        }
    except Exception as e:
        logger.error(f"Failed to search logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.get("/logs/correlation/{trace_id}")
async def get_correlation_trace(trace_id: str) -> Dict[str, Any]:
    """Get log correlation trace by trace ID."""
    try:
        aggregator = create_log_aggregator()
        trace = await aggregator.get_correlation_trace(trace_id)

        return {
            "trace_id": trace_id,
            "entries": [entry.dict() for entry in trace],  # type: ignore
            "total_entries": len(trace),
        }
    except Exception as e:
        logger.error(f"Failed to get correlation trace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.get("/logs/errors/summary")
async def get_error_summary(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
) -> Dict[str, Any]:
    """Get error summary for the specified time period."""
    try:
        aggregator = create_log_aggregator()
        summary = await aggregator.get_error_summary(hours=hours)

        return summary
    except Exception as e:
        logger.error(f"Failed to get error summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.get("/logs/stats")
async def get_log_statistics() -> Dict[str, Any]:
    """Get log statistics."""
    try:
        aggregator = create_log_aggregator()
        stats = aggregator.get_stats()

        return stats
    except Exception as e:
        logger.error(f"Failed to get log statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.get("/metrics/summary")
async def get_metrics_summary() -> Dict[str, Any]:
    """Get metrics summary."""
    try:
        create_metrics_collector()

        # Get basic metrics
        summary = {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": {
                "uptime_seconds": time.time(),
                "memory_usage_bytes": 0,  # Would be populated by actual system metrics
                "cpu_usage_percent": 0,  # Would be populated by actual system metrics
            },
            "api_metrics": {
                "total_requests": 0,  # Would be populated by actual metrics
                "error_rate": 0.0,  # Would be populated by actual metrics
                "avg_latency_ms": 0.0,  # Would be populated by actual metrics
            },
            "character_metrics": {
                "total_interactions": 0,  # Would be populated by actual metrics
                "avg_processing_time_ms": 0.0,  # Would be populated by actual metrics
            },
            "safety_metrics": {
                "total_events": 0,  # Would be populated by actual metrics
                "avg_confidence": 0.0,  # Would be populated by actual metrics
            },
        }

        return summary
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.get("/health/detailed")
async def get_detailed_health() -> Dict[str, Any]:
    """Get detailed health status for monitoring."""
    try:
        # Get system metrics
        create_metrics_collector()
        aggregator = create_log_aggregator()

        # Get recent error count
        error_summary = await aggregator.get_error_summary(hours=1)
        recent_errors = error_summary.get("error_count", 0)

        # Get log statistics
        log_stats = aggregator.get_stats()

        health_status: Dict[str, Any] = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time(),
            "components": {
                "api": {"status": "healthy", "last_check": datetime.now().isoformat()},
                "logging": {
                    "status": "healthy",
                    "last_check": datetime.now().isoformat(),
                },
                "metrics": {
                    "status": "healthy",
                    "last_check": datetime.now().isoformat(),
                },
                "log_aggregation": {
                    "status": "healthy",
                    "last_check": datetime.now().isoformat(),
                },
            },
            "metrics": {
                "recent_errors": recent_errors,
                "total_logs": log_stats.get("total_logs", 0),
                "errors_24h": log_stats.get("errors_24h", 0),
                "warnings_24h": log_stats.get("warnings_24h", 0),
            },
            "alerts": [],
        }

        # Add alerts based on metrics
        if recent_errors > 10:
            health_status["alerts"].append(
                {
                    "level": "warning",
                    "message": f"High error rate: {recent_errors} errors in the last hour",
                }
            )

        if log_stats.get("errors_24h", 0) > 100:
            health_status["alerts"].append(
                {
                    "level": "critical",
                    "message": f"Very high error rate: {log_stats.get('errors_24h', 0)} errors in the last 24 hours",
                }
            )

        # Update overall status based on alerts
        if any(alert["level"] == "critical" for alert in health_status["alerts"]):
            health_status["status"] = "critical"
        elif any(alert["level"] == "warning" for alert in health_status["alerts"]):
            health_status["status"] = "warning"

        return health_status
    except Exception as e:
        logger.error(f"Failed to get detailed health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.post("/export/all")
async def export_all_monitoring_configs(
    output_dir: str = "monitoring/complete",
) -> Dict[str, Any]:
    """Export all monitoring configurations (Grafana, Prometheus)."""
    try:
        output_path = Path.cwd() / output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        # Export Grafana configuration
        grafana_path = output_path / "grafana"
        export_grafana_config(grafana_path)

        # Export Prometheus configuration
        prometheus_path = output_path / "prometheus"
        export_prometheus_config(prometheus_path)

        # Create README
        readme_content = f"""# Character AI - Monitoring Configuration


This directory contains complete monitoring configuration for the Interactive Character
Platform.

## Directory Structure

- `grafana/` - Grafana dashboard and configuration files
- `prometheus/` - Prometheus metrics and alerting configuration files

## Setup Instructions

### 1. Grafana Setup
1. Install Grafana
2. Import the dashboard from `grafana/icp-dashboard.json`
3. Configure the Prometheus data source to point to your Prometheus instance

### 2. Prometheus Setup
1. Install Prometheus
2. Use the configuration from `prometheus/prometheus.yml`
3. Load the alert rules from `prometheus/icp-alerts.yml`

## API Endpoints

The platform exposes the following monitoring endpoints:

- `GET /api/v1/monitoring/metrics` - Prometheus metrics
- `GET /api/v1/monitoring/grafana/dashboard` - Grafana dashboard configuration
- `GET /api/v1/monitoring/prometheus/config` - Prometheus configuration
- `GET /api/v1/monitoring/logs/search` - Log search
- `GET /api/v1/monitoring/health/detailed` - Detailed health status

## Generated on: {datetime.now().isoformat()}
"""

        readme_path = output_path / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)

        return {
            "message": f"All monitoring configurations exported to {output_dir}",
            "directories": {
                "grafana": str(grafana_path),
                "prometheus": str(prometheus_path),
            },
            "readme": str(readme_path),
        }
    except Exception as e:
        logger.error(f"Failed to export all monitoring configurations: {e}")
        raise HTTPException(status_code=500, detail=str(e))
