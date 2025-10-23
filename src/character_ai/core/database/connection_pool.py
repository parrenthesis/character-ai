"""
Database connection pool for efficient SQLite operations.

Provides connection pooling, batch operations, and transaction
management for the memory system database operations.
"""

import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Configuration for database connection pool."""

    max_connections: int = 5
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0  # 5 minutes
    enable_wal_mode: bool = True
    enable_foreign_keys: bool = True
    journal_mode: str = "WAL"  # WAL mode for better concurrency


class SQLiteConnectionPool:
    """Thread-safe SQLite connection pool."""

    def __init__(self, db_path: str, config: Optional[PoolConfig] = None):
        """Initialize connection pool."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.config = config or PoolConfig()
        self._connections: List[sqlite3.Connection] = []
        self._in_use: set = set()
        self._lock = threading.RLock()
        self._last_used: Dict[sqlite3.Connection, float] = {}

        # Initialize database with optimized settings
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database with optimized settings."""
        with self._get_connection() as conn:
            # Enable WAL mode for better concurrency
            if self.config.enable_wal_mode:
                conn.execute("PRAGMA journal_mode=WAL")

            # Enable foreign keys
            if self.config.enable_foreign_keys:
                conn.execute("PRAGMA foreign_keys=ON")

            # Optimize for performance
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB

            conn.commit()

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=self.config.connection_timeout,
            check_same_thread=False,
        )

        # Set row factory for easier data access
        conn.row_factory = sqlite3.Row

        # Enable WAL mode if configured
        if self.config.enable_wal_mode:
            conn.execute("PRAGMA journal_mode=WAL")

        # Enable foreign keys if configured
        if self.config.enable_foreign_keys:
            conn.execute("PRAGMA foreign_keys=ON")

        return conn

    def _get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool."""
        with self._lock:
            # Clean up idle connections
            self._cleanup_idle_connections()

            # Try to get an existing connection
            for conn in self._connections:
                if conn not in self._in_use:
                    self._in_use.add(conn)
                    self._last_used[conn] = time.time()
                    return conn

            # Create new connection if under limit
            if len(self._connections) < self.config.max_connections:
                conn = self._create_connection()
                self._connections.append(conn)
                self._in_use.add(conn)
                self._last_used[conn] = time.time()
                return conn

            # Wait for a connection to become available
            start_time = time.time()
            while time.time() - start_time < self.config.connection_timeout:
                for conn in self._connections:
                    if conn not in self._in_use:
                        self._in_use.add(conn)
                        self._last_used[conn] = time.time()
                        return conn

                time.sleep(0.01)  # Small delay to avoid busy waiting

            raise RuntimeError(
                f"Could not get connection within {self.config.connection_timeout}s"
            )

    def _return_connection(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool."""
        with self._lock:
            if conn in self._in_use:
                self._in_use.remove(conn)
                self._last_used[conn] = time.time()

    def _cleanup_idle_connections(self) -> None:
        """Clean up idle connections."""
        current_time = time.time()
        to_remove = []

        for conn in self._connections:
            if (
                conn not in self._in_use
                and current_time - self._last_used.get(conn, 0)
                > self.config.idle_timeout
            ):
                to_remove.append(conn)

        for conn in to_remove:
            try:
                conn.close()
                self._connections.remove(conn)
                if conn in self._last_used:
                    del self._last_used[conn]
            except Exception as e:
                logger.warning(f"Error closing idle connection: {e}")

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a connection from the pool with automatic cleanup."""
        conn = None
        try:
            conn = self._get_connection()
            yield conn
        finally:
            if conn:
                self._return_connection(conn)

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Execute operations within a transaction."""
        with self.get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def execute_batch(self, query: str, params_list: List[Tuple[Any, ...]]) -> None:
        """Execute a batch of operations efficiently."""
        with self.transaction() as conn:
            conn.executemany(query, params_list)

    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self._lock:
            for conn in self._connections:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")

            self._connections.clear()
            self._in_use.clear()
            self._last_used.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            return {
                "total_connections": len(self._connections),
                "in_use_connections": len(self._in_use),
                "available_connections": len(self._connections) - len(self._in_use),
                "max_connections": self.config.max_connections,
            }

    def __del__(self) -> None:
        """Cleanup on destruction."""
        self.close_all()
