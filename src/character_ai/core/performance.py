"""
Performance testing and optimization system.

This module provides:
- Latency budgets for each component
- Performance regression testing
- Stress testing for concurrent users
- P50/P95/P99 latency tracking
- Performance monitoring and alerting
"""

import asyncio
import statistics
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from ..observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyBudget:
    """Defines latency budgets for different components."""

    component: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    description: str = ""


@dataclass
class PerformanceMetrics:
    """Performance metrics for a component."""

    component: str
    operation: str
    duration_ms: float
    timestamp: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Performance report with statistics."""

    component: str
    operation: str
    total_operations: int
    successful_operations: int
    failed_operations: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    min_ms: float
    avg_ms: float
    budget_violations: int
    time_window_seconds: float


class PerformanceTracker:
    """Tracks performance metrics and enforces latency budgets."""

    def __init__(self, max_metrics: int = 10000) -> None:
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.lock = threading.RLock()

        # Define latency budgets for each component
        self.budgets = {
            "stt": LatencyBudget(
                "stt", 1000, 2000, 3000, 5000, "Speech-to-Text processing"
            ),
            "llm": LatencyBudget(
                "llm", 2000, 5000, 8000, 15000, "Language model inference"
            ),
            "tts": LatencyBudget(
                "tts", 1500, 3000, 5000, 10000, "Text-to-Speech generation"
            ),
            "safety": LatencyBudget("safety", 100, 300, 500, 1000, "Safety filtering"),
            "character": LatencyBudget(
                "character", 200, 500, 1000, 2000, "Character management"
            ),
            "memory": LatencyBudget(
                "memory", 50, 150, 300, 500, "Session memory operations"
            ),
            "api": LatencyBudget("api", 100, 500, 1000, 2000, "API request processing"),
        }

        logger.info(
            "PerformanceTracker initialized",
            max_metrics=max_metrics,
            components=list(self.budgets.keys()),
        )

    def record_metric(
        self,
        component: str,
        operation: str,
        duration_ms: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a performance metric."""
        with self.lock:
            metric = PerformanceMetrics(
                component=component,
                operation=operation,
                duration_ms=duration_ms,
                timestamp=time.time(),
                success=success,
                metadata=metadata or {},
            )
            self.metrics.append(metric)

            # Check for budget violations
            if component in self.budgets:
                budget = self.budgets[component]
                if duration_ms > budget.p95_ms:
                    logger.warning(
                        "Performance budget violation",
                        component=component,
                        operation=operation,
                        duration_ms=duration_ms,
                        budget_p95_ms=budget.p95_ms,
                        severity=(
                            "warning" if duration_ms <= budget.p99_ms else "critical"
                        ),
                    )

    def get_component_metrics(
        self, component: str, time_window_seconds: float = 3600
    ) -> List[PerformanceMetrics]:
        """Get metrics for a specific component within a time window."""
        cutoff_time = time.time() - time_window_seconds

        with self.lock:
            return [
                metric
                for metric in self.metrics
                if metric.component == component and metric.timestamp >= cutoff_time
            ]

    def get_performance_report(
        self,
        component: str,
        operation: Optional[str] = None,
        time_window_seconds: float = 3600,
    ) -> PerformanceReport:
        """Generate a performance report for a component."""
        metrics = self.get_component_metrics(component, time_window_seconds)

        if operation:
            metrics = [m for m in metrics if m.operation == operation]

        if not metrics:
            return PerformanceReport(
                component=component,
                operation=operation or "all",
                total_operations=0,
                successful_operations=0,
                failed_operations=0,
                p50_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                max_ms=0.0,
                min_ms=0.0,
                avg_ms=0.0,
                budget_violations=0,
                time_window_seconds=time_window_seconds,
            )

        durations = [m.duration_ms for m in metrics]
        successful = [m for m in metrics if m.success]
        failed = [m for m in metrics if not m.success]

        # Calculate percentiles
        p50 = statistics.median(durations)
        p95 = self._percentile(durations, 95)
        p99 = self._percentile(durations, 99)

        # Count budget violations
        budget_violations = 0
        if component in self.budgets:
            budget = self.budgets[component]
            budget_violations = sum(1 for d in durations if d > budget.p95_ms)

        return PerformanceReport(
            component=component,
            operation=operation or "all",
            total_operations=len(metrics),
            successful_operations=len(successful),
            failed_operations=len(failed),
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            max_ms=max(durations),
            min_ms=min(durations),
            avg_ms=statistics.mean(durations),
            budget_violations=budget_violations,
            time_window_seconds=time_window_seconds,
        )

    def get_all_reports(
        self, time_window_seconds: float = 3600
    ) -> Dict[str, PerformanceReport]:
        """Get performance reports for all components."""
        reports = {}
        for component in self.budgets.keys():
            reports[component] = self.get_performance_report(
                component, time_window_seconds=time_window_seconds
            )
        return reports

    def check_budget_compliance(
        self, component: str, time_window_seconds: float = 3600
    ) -> Dict[str, Any]:
        """Check if a component is within its latency budget."""
        if component not in self.budgets:
            return {"error": f"Unknown component: {component}"}

        budget = self.budgets[component]
        report = self.get_performance_report(
            component, time_window_seconds=time_window_seconds
        )

        compliance = {
            "component": component,
            "budget": {
                "p50_ms": budget.p50_ms,
                "p95_ms": budget.p95_ms,
                "p99_ms": budget.p99_ms,
                "max_ms": budget.max_ms,
            },
            "actual": {
                "p50_ms": report.p50_ms,
                "p95_ms": report.p95_ms,
                "p99_ms": report.p99_ms,
                "max_ms": report.max_ms,
            },
            "violations": {
                "p50": report.p50_ms > budget.p50_ms,
                "p95": report.p95_ms > budget.p95_ms,
                "p99": report.p99_ms > budget.p99_ms,
                "max": report.max_ms > budget.max_ms,
            },
            "status": "compliant" if report.budget_violations == 0 else "violations",
            "total_violations": report.budget_violations,
        }

        return compliance

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of a dataset."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)

        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    def clear_old_metrics(self, max_age_seconds: float = 86400) -> int:
        """Clear metrics older than specified age."""
        cutoff_time = time.time() - max_age_seconds

        with self.lock:
            old_count = len(self.metrics)
            # Keep only recent metrics
            recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
            self.metrics.clear()
            for metric in recent_metrics:
                self.metrics.append(metric)

            removed = old_count - len(self.metrics)
            logger.info(
                "Cleared old performance metrics",
                removed=removed,
                remaining=len(self.metrics),
            )
            return removed


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(
        self,
        tracker: PerformanceTracker,
        component: str,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.tracker = tracker
        self.component = component
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time: Optional[float] = None
        self.duration_ms: Optional[float] = None
        self.success = True

    def __enter__(self) -> "PerformanceTimer":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time:
            self.duration_ms = (time.time() - self.start_time) * 1000
            self.success = exc_type is None

            self.tracker.record_metric(
                component=self.component,
                operation=self.operation,
                duration_ms=self.duration_ms,
                success=self.success,
                metadata=self.metadata,
            )

    async def __aenter__(self) -> "PerformanceTimer":
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time:
            self.duration_ms = (time.time() - self.start_time) * 1000
            self.success = exc_type is None

            self.tracker.record_metric(
                component=self.component,
                operation=self.operation,
                duration_ms=self.duration_ms,
                success=self.success,
                metadata=self.metadata,
            )


class StressTester:
    """Stress testing for concurrent operations."""

    def __init__(self, tracker: PerformanceTracker) -> None:
        self.tracker = tracker
        self.results: List[Dict[str, Any]] = []

    async def run_concurrent_test(
        self,
        operation: Callable,
        concurrency: int,
        duration_seconds: float,
        component: str = "stress_test",
    ) -> Dict[str, Any]:
        """Run a stress test with concurrent operations."""
        logger.info(
            "Starting stress test",
            concurrency=concurrency,
            duration_seconds=duration_seconds,
            component=component,
        )

        start_time = time.time()
        end_time = start_time + duration_seconds

        # Track results
        results: Dict[str, Any] = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "durations": [],
            "errors": [],
        }

        async def worker() -> None:
            while time.time() < end_time:
                try:
                    with PerformanceTimer(
                        self.tracker, component, "stress_test"
                    ) as timer:
                        await operation()

                    results["total_operations"] += 1
                    if timer.success:
                        results["successful_operations"] += 1
                    else:
                        results["failed_operations"] += 1

                    if timer.duration_ms:
                        results["durations"].append(timer.duration_ms)

                except Exception as e:
                    results["failed_operations"] += 1
                    results["errors"].append(str(e))

        # Run concurrent workers
        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate statistics
        if results["durations"]:
            results["avg_duration_ms"] = statistics.mean(results["durations"])
            results["p95_duration_ms"] = self.tracker._percentile(
                results["durations"], 95
            )
            results["max_duration_ms"] = max(results["durations"])

        results["throughput_ops_per_second"] = (
            results["total_operations"] / duration_seconds
        )
        results["success_rate"] = results["successful_operations"] / max(
            results["total_operations"], 1
        )

        logger.info(
            "Stress test completed",
            total_operations=results["total_operations"],
            success_rate=results["success_rate"],
            throughput=results["throughput_ops_per_second"],
        )

        self.results.append(results)
        return results

    async def run_load_test(
        self,
        operation: Callable,
        initial_load: int,
        max_load: int,
        step: int,
        duration_per_step: float,
    ) -> List[Dict[str, Any]]:
        """Run a load test with increasing load."""
        results = []

        for load in range(initial_load, max_load + 1, step):
            logger.info("Running load test", load=load)

            result = await self.run_concurrent_test(
                operation=operation,
                concurrency=load,
                duration_seconds=duration_per_step,
                component="load_test",
            )

            result["load"] = load
            results.append(result)

        return results


def create_performance_tracker() -> PerformanceTracker:
    """Create a new performance tracker instance."""
    return PerformanceTracker()


def track_performance(
    component: str, operation: str, metadata: Optional[Dict[str, Any]] = None
) -> Callable:
    """Decorator for tracking performance of functions."""

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with PerformanceTimer(
                    create_performance_tracker(), component, operation, metadata
                ):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with PerformanceTimer(
                    create_performance_tracker(), component, operation, metadata
                ):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


@asynccontextmanager
async def performance_timer(
    component: str, operation: str, metadata: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[PerformanceTimer, None]:
    """Async context manager for performance timing."""
    tracker = create_performance_tracker()
    timer = PerformanceTimer(tracker, component, operation, metadata)
    try:
        yield timer
    finally:
        timer.__exit__(None, None, None)
