"""
Health check API endpoints with detailed diagnostics.

This module provides health check endpoints including:
- Basic health status
- Detailed component health
- System metrics and diagnostics
- Crash report analysis
"""

import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from ..core.crash_reporting import get_crash_reporter, get_health_status
from ..core.logging import get_logger

logger = get_logger(__name__)

# Create health router
health_router = APIRouter(prefix="/health", tags=["health"])


@health_router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    try:
        health_status = get_health_status()

        return {
            "status": health_status.status,
            "timestamp": health_status.timestamp,
            "uptime_seconds": health_status.uptime_seconds,
            "recent_crashes": health_status.recent_crashes,
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=500, detail="Health check failed")


@health_router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with component status and system metrics."""
    try:
        health_status = get_health_status()
        crash_reporter = get_crash_reporter()

        # Get recent crashes for analysis
        recent_crashes = crash_reporter.get_recent_crashes(5)

        return {
            "status": health_status.status,
            "timestamp": health_status.timestamp,
            "uptime_seconds": health_status.uptime_seconds,
            "components": health_status.components,
            "system_metrics": health_status.system_metrics,
            "recent_crashes": health_status.recent_crashes,
            "recent_crash_details": [
                {
                    "crash_id": crash.crash_id,
                    "timestamp": crash.timestamp,
                    "component": crash.component,
                    "severity": crash.severity,
                    "error_type": crash.error_type,
                    "error_message": (
                        crash.error_message[:100] + "..."
                        if len(crash.error_message) > 100
                        else crash.error_message
                    ),
                }
                for crash in recent_crashes
            ],
        }
    except Exception as e:
        logger.error("Detailed health check failed", error=str(e))
        raise HTTPException(status_code=500, detail="Detailed health check failed")


@health_router.get("/components")
async def component_health() -> Dict[str, Any]:
    """Get health status for all components."""
    try:
        health_status = get_health_status()

        return {
            "timestamp": health_status.timestamp,
            "components": health_status.components,
        }
    except Exception as e:
        logger.error("Component health check failed", error=str(e))
        raise HTTPException(status_code=500, detail="Component health check failed")


@health_router.get("/system")
async def system_health() -> Dict[str, Any]:
    """Get system health metrics."""
    try:
        health_status = get_health_status()

        return {
            "timestamp": health_status.timestamp,
            "system_metrics": health_status.system_metrics,
            "uptime_seconds": health_status.uptime_seconds,
        }
    except Exception as e:
        logger.error("System health check failed", error=str(e))
        raise HTTPException(status_code=500, detail="System health check failed")


@health_router.get("/crashes")
async def get_crashes(
    limit: int = 10, component: Optional[str] = None, severity: Optional[str] = None
) -> Dict[str, Any]:
    """Get crash reports with optional filtering."""
    try:
        crash_reporter = get_crash_reporter()

        if component:
            crashes = crash_reporter.get_crashes_by_component(component)
        elif severity:
            crashes = crash_reporter.get_crashes_by_severity(severity)
        else:
            crashes = crash_reporter.get_recent_crashes(limit)

        # Limit results
        crashes = crashes[-limit:] if len(crashes) > limit else crashes

        return {
            "timestamp": time.time(),
            "count": len(crashes),
            "crashes": [
                {
                    "crash_id": crash.crash_id,
                    "timestamp": crash.timestamp,
                    "component": crash.component,
                    "severity": crash.severity,
                    "error_type": crash.error_type,
                    "error_message": crash.error_message,
                    "context": crash.context,
                    "system_info": crash.system_info,
                }
                for crash in crashes
            ],
        }
    except Exception as e:
        logger.error("Failed to get crashes", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get crashes")


@health_router.get("/crashes/{crash_id}")
async def get_crash_details(crash_id: str) -> Dict[str, Any]:
    """Get detailed information for a specific crash."""
    try:
        crash_reporter = get_crash_reporter()
        all_crashes = crash_reporter.buffer.get_all()

        # Find the specific crash
        crash = next((c for c in all_crashes if c.crash_id == crash_id), None)
        if not crash:
            raise HTTPException(status_code=404, detail="Crash not found")

        return {
            "crash_id": crash.crash_id,
            "timestamp": crash.timestamp,
            "component": crash.component,
            "severity": crash.severity,
            "error_type": crash.error_type,
            "error_message": crash.error_message,
            "traceback": crash.traceback,
            "context": crash.context,
            "system_info": crash.system_info,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get crash details", error=str(e), crash_id=crash_id)
        raise HTTPException(status_code=500, detail="Failed to get crash details")


@health_router.post("/crashes/clear")
async def clear_crashes(max_age_seconds: int = 86400) -> Dict[str, Any]:
    """Clear old crash reports."""
    try:
        crash_reporter = get_crash_reporter()
        old_count = crash_reporter.buffer.size()

        crash_reporter.clear_old_reports(max_age_seconds)

        new_count = crash_reporter.buffer.size()
        cleared_count = old_count - new_count

        logger.info(
            "Cleared old crash reports",
            cleared_count=cleared_count,
            remaining_count=new_count,
            max_age_seconds=max_age_seconds,
        )

        return {
            "success": True,
            "cleared_count": cleared_count,
            "remaining_count": new_count,
            "max_age_seconds": max_age_seconds,
        }
    except Exception as e:
        logger.error("Failed to clear crashes", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to clear crashes")


@health_router.post("/test-crash")
async def test_crash_reporting() -> Dict[str, Any]:
    """Test crash reporting system (for testing purposes only)."""
    try:
        crash_reporter = get_crash_reporter()

        # Report a test error
        crash_id = crash_reporter.report_error(
            error_type="TestError",
            error_message="This is a test error for crash reporting validation",
            component="health_api",
            severity="INFO",
            context={"test": True, "purpose": "validation"},
        )

        logger.info("Test crash reported", crash_id=crash_id)

        return {
            "success": True,
            "crash_id": crash_id,
            "message": "Test crash reported successfully",
        }
    except Exception as e:
        logger.error("Failed to test crash reporting", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to test crash reporting")


@health_router.get("/metrics")
async def health_metrics() -> Dict[str, Any]:
    """Get health-related metrics for monitoring."""
    try:
        health_status = get_health_status()
        crash_reporter = get_crash_reporter()

        # Get crash statistics
        all_crashes = crash_reporter.buffer.get_all()
        recent_crashes = [
            c for c in all_crashes if time.time() - c.timestamp < 3600
        ]  # Last hour

        # Count by severity
        severity_counts: Dict[str, int] = {}
        for crash in recent_crashes:
            severity_counts[crash.severity] = severity_counts.get(crash.severity, 0) + 1

        # Count by component
        component_counts: Dict[str, int] = {}
        for crash in recent_crashes:
            component_counts[crash.component] = (
                component_counts.get(crash.component, 0) + 1
            )

        return {
            "timestamp": health_status.timestamp,
            "uptime_seconds": health_status.uptime_seconds,
            "overall_status": health_status.status,
            "total_crashes": len(all_crashes),
            "recent_crashes": len(recent_crashes),
            "severity_breakdown": severity_counts,
            "component_breakdown": component_counts,
            "system_metrics": health_status.system_metrics,
        }
    except Exception as e:
        logger.error("Failed to get health metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get health metrics")
