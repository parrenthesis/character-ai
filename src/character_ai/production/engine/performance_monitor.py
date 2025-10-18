"""Performance monitoring for real-time interaction engine."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Tracks performance metrics for the real-time interaction engine."""

    def __init__(self) -> None:
        # Performance tracking
        self.performance_metrics = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "average_latency": 0.0,
            "last_latency": 0.0,
        }

    def update_metrics(self, latency: float, success: bool) -> None:
        """Update performance tracking metrics."""
        self.performance_metrics["total_interactions"] += 1
        if success:
            self.performance_metrics["successful_interactions"] += 1

        self.performance_metrics["last_latency"] = latency

        # Update average latency (simple moving average)
        total = self.performance_metrics["total_interactions"]
        current_avg = self.performance_metrics["average_latency"]
        self.performance_metrics["average_latency"] = (
            current_avg * (total - 1) + latency
        ) / total

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()

    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        total = self.performance_metrics["total_interactions"]
        if total == 0:
            return 0.0
        return (self.performance_metrics["successful_interactions"] / total) * 100

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.performance_metrics = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "average_latency": 0.0,
            "last_latency": 0.0,
        }
        logger.info("Performance metrics reset")

    def log_performance_summary(self) -> None:
        """Log a summary of current performance metrics."""
        metrics = self.get_metrics()
        success_rate = self.get_success_rate()

        logger.info("Performance Summary:")
        logger.info(f"  Total Interactions: {metrics['total_interactions']}")
        logger.info(f"  Success Rate: {success_rate:.1f}%")
        logger.info(f"  Average Latency: {metrics['average_latency']:.3f}s")
        logger.info(f"  Last Latency: {metrics['last_latency']:.3f}s")
