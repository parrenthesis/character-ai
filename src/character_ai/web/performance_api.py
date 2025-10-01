"""
Performance API endpoints for monitoring and testing.

This module provides:
- Performance metrics endpoints
- Budget compliance checking
- Stress testing endpoints
- Performance reports and analytics
"""

import asyncio
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..core.config import Config
from ..core.logging import get_logger
from ..core.performance import StressTester, get_performance_tracker

logger = get_logger(__name__)

# Create performance router
performance_router = APIRouter(prefix="/performance", tags=["performance"])


class PerformanceTestRequest(BaseModel):
    """Request model for performance testing."""

    component: str
    operation: str
    concurrency: int = 1
    duration_seconds: float = 10.0
    metadata: Optional[Dict[str, Any]] = None


class LoadTestRequest(BaseModel):
    """Request model for load testing."""

    component: str
    operation: str
    initial_load: int = 1
    max_load: int = 10
    step: int = 1
    duration_per_step: float = 5.0
    metadata: Optional[Dict[str, Any]] = None


@performance_router.get("/metrics")
async def get_performance_metrics(
    component: Optional[str] = None, time_window_seconds: float = 3600
) -> Dict[str, Any]:
    """Get performance metrics for components."""
    try:
        tracker = get_performance_tracker()

        if component:
            # Get metrics for specific component
            metrics = tracker.get_component_metrics(component, time_window_seconds)
            report = tracker.get_performance_report(
                component, time_window_seconds=time_window_seconds
            )

            return {
                "component": component,
                "time_window_seconds": time_window_seconds,
                "total_metrics": len(metrics),
                "report": {
                    "total_operations": report.total_operations,
                    "successful_operations": report.successful_operations,
                    "failed_operations": report.failed_operations,
                    "p50_ms": report.p50_ms,
                    "p95_ms": report.p95_ms,
                    "p99_ms": report.p99_ms,
                    "max_ms": report.max_ms,
                    "min_ms": report.min_ms,
                    "avg_ms": report.avg_ms,
                    "budget_violations": report.budget_violations,
                },
            }
        else:
            # Get metrics for all components
            reports = tracker.get_all_reports(time_window_seconds)
            return {
                "time_window_seconds": time_window_seconds,
                "components": {
                    comp: {
                        "total_operations": report.total_operations,
                        "successful_operations": report.successful_operations,
                        "failed_operations": report.failed_operations,
                        "p50_ms": report.p50_ms,
                        "p95_ms": report.p95_ms,
                        "p99_ms": report.p99_ms,
                        "max_ms": report.max_ms,
                        "avg_ms": report.avg_ms,
                        "budget_violations": report.budget_violations,
                    }
                    for comp, report in reports.items()
                },
            }
    except Exception as e:
        logger.error("Failed to get performance metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get performance metrics")



@performance_router.get("/budgets")
async def get_latency_budgets() -> Dict[str, Any]:
    """Get latency budgets for all components."""
    try:
        tracker = get_performance_tracker()
        budgets = {}

        for component, budget in tracker.budgets.items():
            budgets[component] = {
                "p50_ms": budget.p50_ms,
                "p95_ms": budget.p95_ms,
                "p99_ms": budget.p99_ms,
                "max_ms": budget.max_ms,
                "description": budget.description,
            }

        return {"budgets": budgets, "timestamp": time.time()}
    except Exception as e:
        logger.error("Failed to get latency budgets", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get latency budgets")


@performance_router.get("/compliance")
async def check_budget_compliance(
    component: Optional[str] = None, time_window_seconds: float = 3600
) -> Dict[str, Any]:
    """Check budget compliance for components."""
    try:
        tracker = get_performance_tracker()

        if component:
            # Check specific component
            compliance = tracker.check_budget_compliance(component, time_window_seconds)

            return {
                "component": component,
                "compliance": compliance,
                "timestamp": time.time(),
            }
        else:
            # Check all components
            compliance_results = {}
            for comp in tracker.budgets.keys():
                compliance_results[comp] = tracker.check_budget_compliance(
                    comp, time_window_seconds
                )

            return {"components": compliance_results, "timestamp": time.time()}
    except Exception as e:
        logger.error("Failed to check budget compliance", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to check budget compliance")



@performance_router.post("/test")
async def run_performance_test(request: PerformanceTestRequest) -> Dict[str, Any]:
    """Run a performance test for a component."""
    try:
        tracker = get_performance_tracker()
        stress_tester = StressTester(tracker)

        # Create a test operation
        config = Config()

        async def test_operation() -> None:
            # Simulate some work based on component using configurable delays
            if not config.streaming.simulation_delays:
                # Skip delays if simulation is disabled
                return

            if request.component == "stt":
                await asyncio.sleep(
                    config.streaming.token_generation_delay * 2
                )  # Simulate STT processing
            elif request.component == "llm":
                await asyncio.sleep(
                    config.streaming.token_generation_delay * 10
                )  # Simulate LLM inference
            elif request.component == "tts":
                await asyncio.sleep(
                    config.streaming.token_generation_delay * 6
                )  # Simulate TTS generation
            elif request.component == "safety":
                await asyncio.sleep(
                    config.streaming.token_generation_delay
                )  # Simulate safety filtering
            else:
                await asyncio.sleep(
                    config.streaming.token_generation_delay * 2
                )  # Default simulation

        # Run the test
        result = await stress_tester.run_concurrent_test(
            operation=test_operation,
            concurrency=request.concurrency,
            duration_seconds=request.duration_seconds,
            component=request.component,
        )

        logger.info(
            "Performance test completed",
            component=request.component,
            concurrency=request.concurrency,
            duration=request.duration_seconds,
            total_operations=result["total_operations"],
        )

        return {
            "success": True,
            "component": request.component,
            "test_config": {
                "concurrency": request.concurrency,
                "duration_seconds": request.duration_seconds,
            },
            "results": result,
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error("Failed to run performance test", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to run performance test")


@performance_router.post("/load-test")
async def run_load_test(request: LoadTestRequest) -> Dict[str, Any]:
    """Run a load test with increasing load."""
    try:
        tracker = get_performance_tracker()
        stress_tester = StressTester(tracker)

        # Create a test operation
        config = Config()

        async def test_operation() -> None:
            # Simulate some work based on component using configurable delays
            if not config.streaming.simulation_delays:
                # Skip delays if simulation is disabled
                return

            if request.component == "stt":
                await asyncio.sleep(config.streaming.token_generation_delay * 2)
            elif request.component == "llm":
                await asyncio.sleep(config.streaming.token_generation_delay * 10)
            elif request.component == "tts":
                await asyncio.sleep(config.streaming.token_generation_delay * 6)
            elif request.component == "safety":
                await asyncio.sleep(config.streaming.token_generation_delay)
            else:
                await asyncio.sleep(config.streaming.token_generation_delay * 2)

        # Run the load test
        results = await stress_tester.run_load_test(
            operation=test_operation,
            initial_load=request.initial_load,
            max_load=request.max_load,
            step=request.step,
            duration_per_step=request.duration_per_step,
        )

        logger.info(
            "Load test completed",
            component=request.component,
            max_load=request.max_load,
            steps=len(results),
        )

        return {
            "success": True,
            "component": request.component,
            "test_config": {
                "initial_load": request.initial_load,
                "max_load": request.max_load,
                "step": request.step,
                "duration_per_step": request.duration_per_step,
            },
            "results": results,
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error("Failed to run load test", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to run load test")


@performance_router.get("/reports")
async def get_performance_reports(
    time_window_seconds: float = 3600, format: str = "json"
) -> Dict[str, Any]:
    """Get comprehensive performance reports."""
    try:
        tracker = get_performance_tracker()
        reports = tracker.get_all_reports(time_window_seconds)

        # Calculate overall statistics
        total_operations = sum(r.total_operations for r in reports.values())
        total_violations = sum(r.budget_violations for r in reports.values())

        # Find worst performing component
        worst_component = None
        worst_p95: float = 0.0
        for comp, report in reports.items():
            if report.p95_ms > worst_p95:
                worst_p95 = float(report.p95_ms)
                worst_component = comp

        return {
            "time_window_seconds": time_window_seconds,
            "summary": {
                "total_operations": total_operations,
                "total_budget_violations": total_violations,
                "worst_performing_component": worst_component,
                "worst_p95_ms": worst_p95,
            },
            "components": {
                comp: {
                    "total_operations": report.total_operations,
                    "success_rate": report.successful_operations
                    / max(report.total_operations, 1),
                    "p50_ms": report.p50_ms,
                    "p95_ms": report.p95_ms,
                    "p99_ms": report.p99_ms,
                    "budget_violations": report.budget_violations,
                    "status": (
                        "healthy" if report.budget_violations == 0 else "degraded"
                    ),
                }
                for comp, report in reports.items()
            },
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error("Failed to get performance reports", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get performance reports")



@performance_router.post("/clear-metrics")
async def clear_old_metrics(max_age_seconds: float = 86400) -> Dict[str, Any]:
    """Clear old performance metrics."""
    try:
        tracker = get_performance_tracker()
        removed_count = tracker.clear_old_metrics(max_age_seconds)

        logger.info(
            "Cleared old performance metrics",
            removed=removed_count,
            max_age_seconds=max_age_seconds,
        )

        return {
            "success": True,
            "removed_metrics": removed_count,
            "max_age_seconds": max_age_seconds,
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error("Failed to clear metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to clear metrics")


@performance_router.get("/health")
async def performance_health() -> Dict[str, Any]:
    """Get performance system health status."""
    try:
        tracker = get_performance_tracker()

        # Check recent performance
        recent_reports = tracker.get_all_reports(
            time_window_seconds=300
        )  # Last 5 minutes

        # Determine health status
        total_violations = sum(r.budget_violations for r in recent_reports.values())
        if total_violations == 0:
            status = "healthy"
        elif total_violations < 10:
            status = "degraded"
        else:
            status = "critical"

        return {
            "status": status,
            "total_violations": total_violations,
            "components": {
                comp: {
                    "status": (
                        "healthy" if report.budget_violations == 0 else "degraded"
                    ),
                    "violations": report.budget_violations,
                    "p95_ms": report.p95_ms,
                }
                for comp, report in recent_reports.items()
            },
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error("Failed to get performance health", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get performance health")
