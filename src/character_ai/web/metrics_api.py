"""
Prometheus metrics API endpoint for the character.ai.

Provides /metrics endpoint for Prometheus scraping and monitoring.
"""

from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse

from ..core.logging import get_logger
from ..core.metrics import get_metrics_collector

logger = get_logger(__name__)

# Create metrics router
metrics_router = APIRouter(prefix="/metrics", tags=["metrics"])


@metrics_router.get("/")
async def get_metrics() -> Response:
    """Get Prometheus metrics in text format."""
    try:
        collector = get_metrics_collector()
        metrics_data = collector.get_metrics()

        logger.info("Metrics endpoint accessed")

        return PlainTextResponse(
            content=metrics_data.decode('utf-8'), media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return PlainTextResponse(
            content="# Error retrieving metrics\n",
            status_code=500,
            media_type="text/plain",
        )


@metrics_router.get("/health")
async def metrics_health() -> dict:
    """Health check for metrics endpoint."""
    try:
        collector = get_metrics_collector()
        # Try to get metrics to verify collector is working
        collector.get_metrics()

        return {
            "status": "healthy",
            "component": "metrics",
            "message": "Metrics collector is working",
        }
    except Exception as e:
        logger.error(f"Metrics health check failed: {e}")
        return {"status": "unhealthy", "component": "metrics", "error": str(e)}
