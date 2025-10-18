"""
Log Search API

Provides REST endpoints for log search, correlation, and analysis.
Builds on the log aggregation system to enable advanced debugging
and monitoring capabilities.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...features.security import DeviceIdentity
from ...observability import get_logger
from ...observability.log_aggregation import (
    LogLevel,
    LogSearchQuery,
    LogSource,
    create_log_aggregator,
)
from ..security_deps import require_admin

logger = get_logger(__name__)

# Create router
log_search_router = APIRouter(prefix="/api/v1/logs", tags=["log-search"])


class LogSearchRequest(BaseModel):
    """Request model for log search."""

    # Time range
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Filters
    levels: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    request_ids: Optional[List[str]] = None
    trace_ids: Optional[List[str]] = None
    device_ids: Optional[List[str]] = None
    character_ids: Optional[List[str]] = None

    # Text search
    message_pattern: Optional[str] = None
    component_pattern: Optional[str] = None

    # Pagination
    limit: int = Field(default=1000, ge=1, le=10000)
    offset: int = Field(default=0, ge=0)

    # Sorting
    sort_by: str = Field(default="timestamp", pattern="^(timestamp|level|source)$")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")


class LogSearchResponse(BaseModel):
    """Response model for log search."""

    entries: List[Dict[str, Any]]
    total_count: int
    execution_time_ms: float
    query: Dict[str, Any]


class LogCorrelationResponse(BaseModel):
    """Response model for log correlation."""

    trace_id: str
    entries: List[Dict[str, Any]]
    total_count: int
    execution_time_ms: float


class ErrorSummaryResponse(BaseModel):
    """Response model for error summary."""

    total_errors: int
    by_component: Dict[str, int]
    by_error_type: Dict[str, int]
    by_device: Dict[str, int]
    recent_errors: List[Dict[str, Any]]
    time_range_hours: int


class LogStatsResponse(BaseModel):
    """Response model for log statistics."""

    stats: Dict[str, Any]
    memory_logs_count: int
    index_sizes: Dict[str, Any]


@log_search_router.post("/search", response_model=LogSearchResponse)
async def search_logs(
    request: LogSearchRequest, device: DeviceIdentity = Depends(require_admin)
) -> LogSearchResponse:
    """Search logs with advanced filtering and pagination."""
    try:
        # Convert string enums to actual enums
        levels = None
        if request.levels:
            try:
                levels = [LogLevel(level) for level in request.levels]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid log level: {e}")

        sources = None
        if request.sources:
            try:
                sources = [LogSource(source) for source in request.sources]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid log source: {e}")

        # Create search query
        query = LogSearchQuery(
            start_time=request.start_time,
            end_time=request.end_time,
            levels=levels,
            sources=sources,
            request_ids=request.request_ids,
            trace_ids=request.trace_ids,
            device_ids=request.device_ids,
            character_ids=request.character_ids,
            message_pattern=request.message_pattern,
            component_pattern=request.component_pattern,
            limit=request.limit,
            offset=request.offset,
            sort_by=request.sort_by,
            sort_order=request.sort_order,
        )

        # Perform search
        aggregator = create_log_aggregator()
        result = await aggregator.search_logs(query)

        # Convert entries to dictionaries
        entries = [entry.to_dict() for entry in result.entries]

        logger.info(
            "Log search completed",
            device_id=device.device_id,
            total_results=result.total_count,
            execution_time_ms=result.execution_time_ms,
        )

        return LogSearchResponse(
            entries=entries,
            total_count=result.total_count,
            execution_time_ms=result.execution_time_ms,
            query=request.dict(),
        )

    except Exception as e:
        logger.error(f"Log search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@log_search_router.get("/trace/{trace_id}", response_model=LogCorrelationResponse)
async def get_trace_logs(
    trace_id: str, device: DeviceIdentity = Depends(require_admin)
) -> LogCorrelationResponse:
    """Get all logs for a specific trace ID."""
    try:
        aggregator = create_log_aggregator()
        start_time = datetime.now()

        entries = await aggregator.get_correlation_trace(trace_id)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        logger.info(
            "Trace correlation completed",
            device_id=device.device_id,
            trace_id=trace_id,
            total_entries=len(entries),
            execution_time_ms=execution_time,
        )

        return LogCorrelationResponse(
            trace_id=trace_id,
            entries=[entry.to_dict() for entry in entries],
            total_count=len(entries),
            execution_time_ms=execution_time,
        )

    except Exception as e:
        logger.error(f"Trace correlation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@log_search_router.get("/request/{request_id}", response_model=LogCorrelationResponse)
async def get_request_logs(
    request_id: str, device: DeviceIdentity = Depends(require_admin)
) -> LogCorrelationResponse:
    """Get all logs for a specific request ID."""
    try:
        aggregator = create_log_aggregator()
        start_time = datetime.now()

        entries = await aggregator.get_request_trace(request_id)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        logger.info(
            "Request correlation completed",
            device_id=device.device_id,
            request_id=request_id,
            total_entries=len(entries),
            execution_time_ms=execution_time,
        )

        return LogCorrelationResponse(
            trace_id=request_id,
            entries=[entry.to_dict() for entry in entries],
            total_count=len(entries),
            execution_time_ms=execution_time,
        )

    except Exception as e:
        logger.error(f"Request correlation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@log_search_router.get("/errors/summary", response_model=ErrorSummaryResponse)
async def get_error_summary(
    hours: int = Query(default=24, ge=1, le=168),  # 1 hour to 1 week
    device: DeviceIdentity = Depends(require_admin),
) -> ErrorSummaryResponse:
    """Get error summary for the last N hours."""
    try:
        aggregator = create_log_aggregator()
        error_analysis = await aggregator.get_error_summary(hours)

        logger.info(
            "Error summary generated",
            device_id=device.device_id,
            hours=hours,
            total_errors=error_analysis["total_errors"],
        )

        return ErrorSummaryResponse(
            total_errors=error_analysis["total_errors"],
            by_component=dict(error_analysis["by_component"]),
            by_error_type=dict(error_analysis["by_error_type"]),
            by_device=dict(error_analysis["by_device"]),
            recent_errors=[
                entry.to_dict() for entry in error_analysis["recent_errors"]
            ],
            time_range_hours=hours,
        )

    except Exception as e:
        logger.error(f"Error summary generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@log_search_router.get("/stats", response_model=LogStatsResponse)
async def get_log_stats(
    device: DeviceIdentity = Depends(require_admin),
) -> LogStatsResponse:
    """Get log aggregation statistics."""
    try:
        aggregator = create_log_aggregator()
        stats = aggregator.get_stats()

        logger.info(
            "Log stats retrieved",
            device_id=device.device_id,
            total_logs=stats["stats"]["total_logs"],
        )

        return LogStatsResponse(**stats)

    except Exception as e:
        logger.error(f"Log stats retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@log_search_router.get("/levels")
async def get_available_levels() -> List[str]:
    """Get available log levels."""
    return [level.value for level in LogLevel]


@log_search_router.get("/sources")
async def get_available_sources() -> List[str]:
    """Get available log sources."""
    return [source.value for source in LogSource]


@log_search_router.get("/health")
async def log_search_health() -> Dict[str, Any]:
    """Health check for log search system."""
    try:
        aggregator = create_log_aggregator()
        stats = aggregator.get_stats()

        return {
            "status": "healthy",
            "total_logs": stats["stats"]["total_logs"],
            "memory_logs": stats["memory_logs_count"],
            "error_rate": stats["stats"]["error_rate"],
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Log search health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
