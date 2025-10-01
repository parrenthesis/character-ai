"""
Log Aggregation System

Provides centralized log collection, search, and analysis capabilities.
Builds on the existing structured logging foundation to enable advanced
debugging and monitoring capabilities.
"""

import json
import re
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles

from .logging import get_logger

logger = get_logger(__name__)


class LogLevel(Enum):
    """Log levels for filtering and analysis."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogSource(Enum):
    """Sources of log entries."""

    APPLICATION = "application"
    API = "api"
    SECURITY = "security"
    PERFORMANCE = "performance"
    SAFETY = "safety"
    CHARACTER = "character"
    AUDIO = "audio"
    LLM = "llm"
    SYSTEM = "system"


@dataclass
class LogEntry:
    """Structured log entry for aggregation and analysis."""

    # Core fields
    id: str
    timestamp: datetime
    level: LogLevel
    source: LogSource
    message: str
    logger_name: str

    # Context fields
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    device_id: Optional[str] = None
    character_id: Optional[str] = None
    session_id: Optional[str] = None

    # Additional metadata
    component: Optional[str] = None
    duration_ms: Optional[float] = None
    status_code: Optional[int] = None
    error_type: Optional[str] = None
    confidence: Optional[float] = None

    # Custom fields
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        result["level"] = self.level.value
        result["source"] = self.source.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        """Create from dictionary."""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["level"] = LogLevel(data["level"])
        data["source"] = LogSource(data["source"])
        return cls(**data)


@dataclass
class LogSearchQuery:
    """Query parameters for log search."""

    # Time range
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Filters
    levels: Optional[List[LogLevel]] = None
    sources: Optional[List[LogSource]] = None
    request_ids: Optional[List[str]] = None
    trace_ids: Optional[List[str]] = None
    device_ids: Optional[List[str]] = None
    character_ids: Optional[List[str]] = None

    # Text search
    message_pattern: Optional[str] = None
    component_pattern: Optional[str] = None

    # Pagination
    limit: int = 1000
    offset: int = 0

    # Sorting
    sort_by: str = "timestamp"
    sort_order: str = "desc"


@dataclass
class LogSearchResult:
    """Result of a log search query."""

    entries: List[LogEntry]
    total_count: int
    query: LogSearchQuery
    execution_time_ms: float


class LogAggregator:
    """Centralized log aggregation and search system."""

    def __init__(self, storage_path: Path = Path.cwd() / "logs/aggregated"):
        self.storage_path = storage_path
        # Don't create directory during instantiation - create when actually needed

        # In-memory storage for recent logs
        self.recent_logs: List[LogEntry] = []
        self.max_memory_logs = 10000

        # Indexes for fast searching
        self._indexes: Dict[str, Any] = {
            "by_timestamp": {},
            "by_level": defaultdict(list),
            "by_source": defaultdict(list),
            "by_request_id": defaultdict(list),
            "by_trace_id": defaultdict(list),
            "by_device_id": defaultdict(list),
            "by_character_id": defaultdict(list),
        }

        # Statistics
        self.stats = {
            "total_logs": 0,
            "by_level": Counter(),
            "by_source": Counter(),
            "error_rate": 0.0,
            "avg_response_time": 0.0,
        }

        logger.info("LogAggregator initialized", storage_path=str(storage_path))

    async def add_log_entry(self, entry: LogEntry) -> None:
        """Add a log entry to the aggregation system."""
        try:
            # Add to recent logs
            self.recent_logs.append(entry)

            # Maintain memory limit
            if len(self.recent_logs) > self.max_memory_logs:
                removed = self.recent_logs.pop(0)
                self._remove_from_indexes(removed)

            # Update indexes
            self._add_to_indexes(entry)

            # Update statistics
            self._update_stats(entry)

            # Persist to disk
            await self._persist_entry(entry)

        except Exception as e:
            logger.error(f"Failed to add log entry: {e}", exc_info=True)

    def _add_to_indexes(self, entry: LogEntry) -> None:
        """Add entry to search indexes."""
        timestamp_key = entry.timestamp.isoformat()
        self._indexes["by_timestamp"][timestamp_key] = entry
        self._indexes["by_level"][entry.level].append(entry)
        self._indexes["by_source"][entry.source].append(entry)

        if entry.request_id:
            self._indexes["by_request_id"][entry.request_id].append(entry)
        if entry.trace_id:
            self._indexes["by_trace_id"][entry.trace_id].append(entry)
        if entry.device_id:
            self._indexes["by_device_id"][entry.device_id].append(entry)
        if entry.character_id:
            self._indexes["by_character_id"][entry.character_id].append(entry)

    def _remove_from_indexes(self, entry: LogEntry) -> None:
        """Remove entry from search indexes."""
        # Remove from level and source indexes
        if entry in self._indexes["by_level"][entry.level]:
            self._indexes["by_level"][entry.level].remove(entry)
        if entry in self._indexes["by_source"][entry.source]:
            self._indexes["by_source"][entry.source].remove(entry)

        # Remove from correlation indexes
        if (
            entry.request_id
            and entry in self._indexes["by_request_id"][entry.request_id]
        ):
            self._indexes["by_request_id"][entry.request_id].remove(entry)
        if entry.trace_id and entry in self._indexes["by_trace_id"][entry.trace_id]:
            self._indexes["by_trace_id"][entry.trace_id].remove(entry)
        if entry.device_id and entry in self._indexes["by_device_id"][entry.device_id]:
            self._indexes["by_device_id"][entry.device_id].remove(entry)
        if (
            entry.character_id
            and entry in self._indexes["by_character_id"][entry.character_id]
        ):
            self._indexes["by_character_id"][entry.character_id].remove(entry)

    def _update_stats(self, entry: LogEntry) -> None:
        """Update aggregation statistics."""
        self.stats["total_logs"] += 1  # type: ignore
        self.stats["by_level"][entry.level.value] += 1  # type: ignore
        self.stats["by_source"][entry.source.value] += 1  # type: ignore

        # Calculate error rate
        error_count = (
            self.stats["by_level"][LogLevel.ERROR.value]  # type: ignore
            + self.stats["by_level"][LogLevel.CRITICAL.value]  # type: ignore
        )
        total_logs = self.stats["total_logs"]
        if isinstance(total_logs, (int, float)):
            self.stats["error_rate"] = (
                error_count / int(total_logs)
                if int(total_logs) > 0
                else 0.0
            )
        else:
            self.stats["error_rate"] = 0.0

        # Update average response time
        if entry.duration_ms is not None:
            current_avg = self.stats["avg_response_time"]
            total_logs = self.stats["total_logs"]
            self.stats["avg_response_time"] = (
                (current_avg * (total_logs - 1)) + entry.duration_ms  # type: ignore
            ) / total_logs  # type: ignore

    async def _persist_entry(self, entry: LogEntry) -> None:
        """Persist log entry to disk."""
        try:
            # Create directory if it doesn't exist
            if not self.storage_path.exists():
                self.storage_path.mkdir(parents=True, exist_ok=True)

            # Create daily log files
            date_str = entry.timestamp.strftime("%Y-%m-%d")
            log_file = self.storage_path / f"logs_{date_str}.jsonl"

            async with aiofiles.open(log_file, "a") as f:
                await f.write(json.dumps(entry.to_dict()) + "\n")

        except Exception as e:
            logger.error(f"Failed to persist log entry: {e}", exc_info=True)

    async def search_logs(self, query: LogSearchQuery) -> LogSearchResult:
        """Search logs based on query parameters."""
        start_time = time.time()

        try:
            # Start with all recent logs
            candidates = self.recent_logs.copy()

            # Apply time filters
            if query.start_time:
                candidates = [
                    log for log in candidates if log.timestamp >= query.start_time
                ]
            if query.end_time:
                candidates = [
                    log for log in candidates if log.timestamp <= query.end_time
                ]

            # Apply level filters
            if query.levels:
                candidates = [log for log in candidates if log.level in query.levels]

            # Apply source filters
            if query.sources:
                candidates = [log for log in candidates if log.source in query.sources]

            # Apply correlation filters
            if query.request_ids:
                candidates = [
                    log for log in candidates if log.request_id in query.request_ids
                ]
            if query.trace_ids:
                candidates = [
                    log for log in candidates if log.trace_id in query.trace_ids
                ]
            if query.device_ids:
                candidates = [
                    log for log in candidates if log.device_id in query.device_ids
                ]
            if query.character_ids:
                candidates = [
                    log for log in candidates if log.character_id in query.character_ids

                ]

            # Apply text search
            if query.message_pattern:
                pattern = re.compile(query.message_pattern, re.IGNORECASE)
                candidates = [log for log in candidates if pattern.search(log.message)]

            if query.component_pattern:
                pattern = re.compile(query.component_pattern, re.IGNORECASE)
                candidates = [
                    log
                    for log in candidates
                    if log.component and pattern.search(log.component)
                ]

            # Sort results
            reverse = query.sort_order == "desc"
            if query.sort_by == "timestamp":
                candidates.sort(key=lambda x: x.timestamp, reverse=reverse)
            elif query.sort_by == "level":
                candidates.sort(key=lambda x: x.level.value, reverse=reverse)

            # Apply pagination
            total_count = len(candidates)
            entries = candidates[query.offset : query.offset + query.limit]

            execution_time = (time.time() - start_time) * 1000

            return LogSearchResult(
                entries=entries,
                total_count=total_count,
                query=query,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            logger.error(f"Failed to search logs: {e}", exc_info=True)
            return LogSearchResult(
                entries=[],
                total_count=0,
                query=query,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def get_correlation_trace(self, trace_id: str) -> List[LogEntry]:
        """Get all logs for a specific trace ID."""
        query = LogSearchQuery(trace_ids=[trace_id], limit=10000)
        result = await self.search_logs(query)
        return result.entries

    async def get_request_trace(self, request_id: str) -> List[LogEntry]:
        """Get all logs for a specific request ID."""
        query = LogSearchQuery(request_ids=[request_id], limit=10000)
        result = await self.search_logs(query)
        return result.entries

    async def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the last N hours."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        query = LogSearchQuery(
            start_time=start_time,
            end_time=end_time,
            levels=[LogLevel.ERROR, LogLevel.CRITICAL],
            limit=10000,
        )

        result = await self.search_logs(query)

        # Analyze errors
        error_analysis = {
            "total_errors": len(result.entries),
            "by_component": Counter(),
            "by_error_type": Counter(),
            "by_device": Counter(),
            "recent_errors": result.entries[-10:] if result.entries else [],
        }

        for entry in result.entries:
            if entry.component:
                error_analysis["by_component"][entry.component] += 1  # type: ignore
            if entry.error_type:
                error_analysis["by_error_type"][entry.error_type] += 1  # type: ignore
            if entry.device_id:
                error_analysis["by_device"][entry.device_id] += 1  # type: ignore

        return error_analysis

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        return {
            "stats": self.stats,
            "memory_logs_count": len(self.recent_logs),
            "index_sizes": {
                "by_level": {
                    level.value: len(entries)
                    for level, entries in self._indexes["by_level"].items()
                },
                "by_source": {
                    source.value: len(entries)
                    for source, entries in self._indexes["by_source"].items()
                },
                "by_request_id": len(self._indexes["by_request_id"]),
                "by_trace_id": len(self._indexes["by_trace_id"]),
                "by_device_id": len(self._indexes["by_device_id"]),
                "by_character_id": len(self._indexes["by_character_id"]),
            },
        }


# Global aggregator instance
_aggregator: Optional[LogAggregator] = None


def get_log_aggregator() -> LogAggregator:
    """Get the global log aggregator instance."""
    global _aggregator
    if _aggregator is None:
        _aggregator = LogAggregator()
    return _aggregator


async def add_log_entry(
    level: LogLevel, source: LogSource, message: str, logger_name: str, **kwargs: Any
) -> None:
    """Add a log entry to the aggregation system."""
    entry = LogEntry(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        level=level,
        source=source,
        message=message,
        logger_name=logger_name,
        **kwargs,
    )

    aggregator = get_log_aggregator()
    await aggregator.add_log_entry(entry)
