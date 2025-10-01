"""
Crash reporting system with ring buffer persistence.

This module provides crash reporting capabilities including:
- Ring buffer for crash logs with persistence
- Error aggregation and analysis
- Graceful error handling and recovery
- Health check endpoints with detailed diagnostics
"""

import asyncio
import json
import os
import signal
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class CrashReport:
    """Represents a crash report with metadata."""

    timestamp: float
    error_type: str
    error_message: str
    traceback: str
    component: str
    severity: str  # CRITICAL, ERROR, WARNING, INFO
    context: Dict[str, Any]
    system_info: Dict[str, Any]
    crash_id: str


@dataclass
class HealthStatus:
    """Represents system health status."""

    status: str  # HEALTHY, DEGRADED, CRITICAL
    timestamp: float
    components: Dict[str, Dict[str, Any]]
    system_metrics: Dict[str, Any]
    recent_crashes: int
    uptime_seconds: float


class RingBuffer:
    """Thread-safe ring buffer for crash reports."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buffer: deque[CrashReport] = deque(maxlen=max_size)
        self._lock = threading.RLock()

    def append(self, item: CrashReport) -> None:
        """Add item to ring buffer."""
        with self._lock:
            self._buffer.append(item)

    def get_all(self) -> List[CrashReport]:
        """Get all items from ring buffer."""
        with self._lock:
            return list(self._buffer)

    def get_recent(self, count: int = 10) -> List[CrashReport]:
        """Get recent items from ring buffer."""
        with self._lock:
            return list(self._buffer)[-count:]

    def clear(self) -> None:
        """Clear the ring buffer."""
        with self._lock:
            self._buffer.clear()

    def size(self) -> int:
        """Get current buffer size."""
        with self._lock:
            return len(self._buffer)


class CrashReporter:
    """Main crash reporting system."""

    def __init__(
        self,
        buffer_size: int = 1000,
        persistence_file: Optional[Path] = None,
        auto_save_interval: int = 300,
    ):  # 5 minutes
        self.buffer = RingBuffer(buffer_size)
        self.persistence_file = persistence_file or Path("logs/crash_reports.json")
        self.auto_save_interval = auto_save_interval
        self.start_time = time.time()
        self._save_task: Optional[asyncio.Task] = None
        self._shutdown = False

        # Don't create directory during instantiation - create when actually needed

        # Load existing crash reports if file exists
        self._load_persisted_reports()

        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()

        logger.info(
            "CrashReporter initialized",
            buffer_size=buffer_size,
            persistence_file=str(self.persistence_file),
        )

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(signum: int, frame: Any) -> None:
            logger.info("Received shutdown signal", signal=signum)
            self._shutdown = True
            self._save_reports_sync()
            sys.exit(0)

        # Only set up signal handlers in main thread
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except ValueError:
            # Signal handlers can only be set in main thread
            logger.debug("Signal handlers not set (not in main thread)")

    def _load_persisted_reports(self) -> None:
        """Load crash reports from persistence file."""
        try:
            if self.persistence_file.exists():
                with open(self.persistence_file, "r") as f:
                    data = json.load(f)
                    for report_data in data.get("reports", []):
                        # Convert back to CrashReport
                        report = CrashReport(**report_data)
                        self.buffer.append(report)

                logger.info(
                    "Loaded persisted crash reports", count=len(data.get("reports", []))

                )
        except Exception as e:
            logger.error("Failed to load persisted crash reports", error=str(e))

    def _save_reports_sync(self) -> None:
        """Synchronously save crash reports to persistence file."""
        try:
            # Create directory if it doesn't exist
            if not self.persistence_file.parent.exists():
                self.persistence_file.parent.mkdir(parents=True, exist_ok=True)

            reports = self.buffer.get_all()
            data = {
                "timestamp": time.time(),
                "reports": [asdict(report) for report in reports],
            }

            # Write to temporary file first, then rename (atomic operation)
            temp_file = self.persistence_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)

            temp_file.rename(self.persistence_file)

            logger.info(
                "Saved crash reports to persistence",
                count=len(reports),
                file=str(self.persistence_file),
            )
        except Exception as e:
            logger.error("Failed to save crash reports", error=str(e))

    async def _auto_save_loop(self) -> None:
        """Background task for auto-saving crash reports."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.auto_save_interval)
                if not self._shutdown:
                    self._save_reports_sync()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in auto-save loop", error=str(e))

    async def start(self) -> None:
        """Start the crash reporter background tasks."""
        if self._save_task is None:
            self._save_task = asyncio.create_task(self._auto_save_loop())
            logger.info("Crash reporter auto-save started")

    async def stop(self) -> None:
        """Stop the crash reporter and save final reports."""
        self._shutdown = True
        if self._save_task:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass

        self._save_reports_sync()
        logger.info("Crash reporter stopped")

    def report_crash(
        self,
        error: Exception,
        component: str,
        severity: str = "ERROR",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Report a crash/error."""
        crash_id = f"crash_{int(time.time() * 1000)}"

        # Get system information
        system_info = self._get_system_info()

        # Create crash report
        report = CrashReport(
            timestamp=time.time(),
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            component=component,
            severity=severity,
            context=context or {},
            system_info=system_info,
            crash_id=crash_id,
        )

        # Add to ring buffer
        self.buffer.append(report)

        # Log the crash
        logger.error(
            "Crash reported",
            crash_id=crash_id,
            component=component,
            severity=severity,
            error_type=type(error).__name__,
            error_message=str(error),
        )

        return crash_id

    def report_error(
        self,
        error_type: str,
        error_message: str,
        component: str,
        severity: str = "ERROR",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Report an error without exception object."""
        crash_id = f"error_{int(time.time() * 1000)}"

        # Get system information
        system_info = self._get_system_info()

        # Create crash report
        report = CrashReport(
            timestamp=time.time(),
            error_type=error_type,
            error_message=error_message,
            traceback="",  # No traceback for manual error reports
            component=component,
            severity=severity,
            context=context or {},
            system_info=system_info,
            crash_id=crash_id,
        )

        # Add to ring buffer
        self.buffer.append(report)

        # Log the error
        logger.error(
            "Error reported",
            crash_id=crash_id,
            component=component,
            severity=severity,
            error_type=error_type,
            error_message=error_message,
        )

        return crash_id

    def get_recent_crashes(self, count: int = 10) -> List[CrashReport]:
        """Get recent crash reports."""
        return self.buffer.get_recent(count)

    def get_crashes_by_component(self, component: str) -> List[CrashReport]:
        """Get crash reports for a specific component."""
        all_reports = self.buffer.get_all()
        return [report for report in all_reports if report.component == component]

    def get_crashes_by_severity(self, severity: str) -> List[CrashReport]:
        """Get crash reports by severity level."""
        all_reports = self.buffer.get_all()
        return [report for report in all_reports if report.severity == severity]

    def get_health_status(self) -> HealthStatus:
        """Get current system health status."""
        recent_crashes = self.get_recent_crashes(10)
        critical_crashes = [c for c in recent_crashes if c.severity == "CRITICAL"]
        error_crashes = [c for c in recent_crashes if c.severity == "ERROR"]

        # Determine overall health status
        if critical_crashes:
            status = "CRITICAL"
        elif error_crashes:
            status = "DEGRADED"
        else:
            status = "HEALTHY"

        # Get component health
        components = self._get_component_health()

        # Get system metrics
        system_metrics = self._get_system_metrics()

        return HealthStatus(
            status=status,
            timestamp=time.time(),
            components=components,
            system_metrics=system_metrics,
            recent_crashes=len(recent_crashes),
            uptime_seconds=time.time() - self.start_time,
        )

    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        try:
            import psutil

            return {
                "python_version": sys.version,
                "platform": sys.platform,
                "cpu_count": os.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "cpu_percent": psutil.cpu_percent(),
                "disk_usage": psutil.disk_usage("/").percent,
                "process_id": os.getpid(),
                "thread_count": threading.active_count(),
            }
        except ImportError:
            logger.warning("psutil not available, using basic system info")
            return {
                "python_version": sys.version,
                "platform": sys.platform,
                "cpu_count": os.cpu_count(),
                "process_id": os.getpid(),
                "thread_count": threading.active_count(),
                "psutil_available": False,
            }
        except Exception as e:
            logger.warning("Failed to get system info", error=str(e))
            return {"error": str(e)}

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            return {
                "memory_usage_percent": memory.percent,
                "memory_available_bytes": memory.available,
                "cpu_percent": psutil.cpu_percent(),
                "disk_usage_percent": psutil.disk_usage("/").percent,
                "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
            }
        except ImportError:
            logger.warning("psutil not available, using basic metrics")
            return {
                "psutil_available": False,
                "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
            }
        except Exception as e:
            logger.warning("Failed to get system metrics", error=str(e))
            return {"error": str(e)}

    def _get_component_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for each component."""
        components = {}

        # Get crashes by component
        all_reports = self.buffer.get_all()
        component_crashes: Dict[str, List[CrashReport]] = {}

        for report in all_reports:
            if report.component not in component_crashes:
                component_crashes[report.component] = []
            component_crashes[report.component].append(report)

        # Determine health for each component
        for component, crashes in component_crashes.items():
            recent_crashes = [
                c for c in crashes if time.time() - c.timestamp < 3600
            ]  # Last hour
            critical_crashes = [c for c in recent_crashes if c.severity == "CRITICAL"]
            error_crashes = [c for c in recent_crashes if c.severity == "ERROR"]

            if critical_crashes:
                health_status = "CRITICAL"
            elif error_crashes:
                health_status = "DEGRADED"
            else:
                health_status = "HEALTHY"

            components[component] = {
                "status": health_status,
                "total_crashes": len(crashes),
                "recent_crashes": len(recent_crashes),
                "critical_crashes": len(critical_crashes),
                "error_crashes": len(error_crashes),
                "last_crash": (
                    max(crashes, key=lambda x: x.timestamp).timestamp
                    if crashes
                    else None
                ),
            }

        return components

    def clear_old_reports(self, max_age_seconds: int = 86400) -> int:  # 24 hours
        """Clear old crash reports."""
        cutoff_time = time.time() - max_age_seconds
        all_reports = self.buffer.get_all()
        recent_reports = [r for r in all_reports if r.timestamp > cutoff_time]

        # Clear buffer and re-add recent reports
        self.buffer.clear()
        for report in recent_reports:
            self.buffer.append(report)

        removed_count = len(all_reports) - len(recent_reports)
        logger.info(
            "Cleared old crash reports",
            removed=removed_count,
            remaining=len(recent_reports),
        )

        return removed_count


# Global crash reporter instance
_crash_reporter: Optional[CrashReporter] = None


def get_crash_reporter() -> CrashReporter:
    """Get the global crash reporter instance."""
    global _crash_reporter
    if _crash_reporter is None:
        _crash_reporter = CrashReporter()
    return _crash_reporter


def report_crash(
    error: Exception,
    component: str,
    severity: str = "ERROR",
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Convenience function to report a crash."""
    return get_crash_reporter().report_crash(error, component, severity, context)


def report_error(
    error_type: str,
    error_message: str,
    component: str,
    severity: str = "ERROR",
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Convenience function to report an error."""
    return get_crash_reporter().report_error(
        error_type, error_message, component, severity, context
    )


def get_health_status() -> HealthStatus:
    """Convenience function to get health status."""
    return get_crash_reporter().get_health_status()
