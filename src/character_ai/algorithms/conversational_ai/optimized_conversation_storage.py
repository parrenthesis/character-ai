"""
Optimized SQLite-based persistent storage for conversation history.

Uses connection pooling, batch operations, and optimized queries for
high-performance conversation storage with automatic cleanup and GDPR compliance.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...core.database.connection_pool import PoolConfig, SQLiteConnectionPool
from .session_memory import ConversationTurn

logger = logging.getLogger(__name__)


@dataclass
class ConversationSession:
    """Represents a conversation session."""

    session_id: str
    user_id: str
    character_name: str
    start_time: float
    end_time: Optional[float] = None
    turn_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "character_name": self.character_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "turn_count": self.turn_count,
        }


class OptimizedConversationStorage:
    """High-performance SQLite storage for conversation history."""

    def __init__(
        self,
        db_path: str,
        max_age_days: int = 30,
        pool_config: Optional[PoolConfig] = None,
    ):
        """Initialize with database path and connection pool."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_age_days = max_age_days

        # Initialize connection pool
        self.pool = SQLiteConnectionPool(str(self.db_path), pool_config)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema with optimized indexes."""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()

            # Create conversations table with optimized schema
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    turn_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    character_name TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    user_input TEXT NOT NULL,
                    character_response TEXT NOT NULL,
                    metadata TEXT,
                    created_at REAL DEFAULT (julianday('now')),
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """
            )

            # Create sessions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    character_name TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    turn_count INTEGER DEFAULT 0,
                    created_at REAL DEFAULT (julianday('now'))
                )
            """
            )

            # Create optimized indexes for common queries
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_user_character_time
                ON conversations (user_id, character_name, timestamp DESC)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_session
                ON conversations (session_id, timestamp)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_timestamp
                ON conversations (timestamp)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_user_character
                ON sessions (user_id, character_name, start_time DESC)
            """
            )

            # Create partial indexes for active sessions
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_active
                ON sessions (user_id, character_name)
                WHERE end_time IS NULL
            """
            )

            conn.commit()

    def start_session(self, session_id: str, user_id: str, character_name: str) -> None:
        """Start a new conversation session."""
        with self.pool.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO sessions
                (session_id, user_id, character_name, start_time, end_time, turn_count)
                VALUES (?, ?, ?, ?, NULL, 0)
            """,
                (session_id, user_id, character_name, time.time()),
            )

            logger.debug(f"Started session {session_id} for user {user_id}")

    def end_session(self, session_id: str) -> None:
        """End a conversation session."""
        with self.pool.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE sessions
                SET end_time = ?
                WHERE session_id = ?
            """,
                (time.time(), session_id),
            )

            logger.debug(f"Ended session {session_id}")

    def store_turn(
        self,
        turn: ConversationTurn,
        session_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a conversation turn."""
        with self.pool.transaction() as conn:
            cursor = conn.cursor()

            # Store the turn
            metadata_json = json.dumps(metadata) if metadata else None
            cursor.execute(
                """
                INSERT INTO conversations
                (turn_id, session_id, user_id, character_name, timestamp,
                 user_input, character_response, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    turn.turn_id,
                    session_id,
                    user_id,
                    turn.character_name,
                    turn.timestamp,
                    turn.user_input,
                    turn.character_response,
                    metadata_json,
                ),
            )

            # Update session turn count
            cursor.execute(
                """
                UPDATE sessions
                SET turn_count = turn_count + 1
                WHERE session_id = ?
            """,
                (session_id,),
            )

            logger.debug(f"Stored turn {turn.turn_id} in session {session_id}")

    def store_turns_batch(
        self, turns: List[Tuple[ConversationTurn, str, str, Optional[Dict[str, Any]]]]
    ) -> None:
        """Store multiple turns in a single transaction."""
        if not turns:
            return

        with self.pool.transaction() as conn:
            cursor = conn.cursor()

            # Prepare batch data
            turn_data = []
            session_updates = {}

            for turn, session_id, user_id, metadata in turns:
                metadata_json = json.dumps(metadata) if metadata else None
                turn_data.append(
                    (
                        turn.turn_id,
                        session_id,
                        user_id,
                        turn.character_name,
                        turn.timestamp,
                        turn.user_input,
                        turn.character_response,
                        metadata_json,
                    )
                )

                # Track session updates
                if session_id not in session_updates:
                    session_updates[session_id] = 0
                session_updates[session_id] += 1

            # Batch insert turns
            cursor.executemany(
                """
                INSERT INTO conversations
                (turn_id, session_id, user_id, character_name, timestamp,
                 user_input, character_response, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                turn_data,
            )

            # Batch update session turn counts
            for session_id, count in session_updates.items():
                cursor.execute(
                    """
                    UPDATE sessions
                    SET turn_count = turn_count + ?
                    WHERE session_id = ?
                """,
                    (count, session_id),
                )

            logger.debug(f"Stored {len(turns)} turns in batch")

    def get_recent_turns(
        self, user_id: str, character_name: str, limit: int = 10
    ) -> List[ConversationTurn]:
        """Get recent conversation turns for a user-character pair."""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT turn_id, character_name, timestamp, user_input, character_response
                FROM conversations
                WHERE user_id = ? AND character_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (user_id, character_name, limit),
            )

            turns = []
            for row in cursor.fetchall():
                turn = ConversationTurn(
                    turn_id=row[0],
                    character_name=row[1],
                    timestamp=row[2],
                    user_input=row[3],
                    character_response=row[4],
                )
                turns.append(turn)

            return list(reversed(turns))  # Return in chronological order

    def get_session_turns(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[ConversationTurn]:
        """Get all turns for a specific session."""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT turn_id, character_name, timestamp, user_input, character_response
                FROM conversations
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """
            if limit:
                query += " LIMIT ?"
                params = (session_id, limit)
            else:
                params = (session_id,)  # type: ignore

            cursor.execute(query, params)

            turns = []
            for row in cursor.fetchall():
                turn = ConversationTurn(
                    turn_id=row[0],
                    character_name=row[1],
                    timestamp=row[2],
                    user_input=row[3],
                    character_response=row[4],
                )
                turns.append(turn)

            return turns

    def search_conversations(
        self, user_id: str, character_name: str, query: str, limit: int = 20
    ) -> List[ConversationTurn]:
        """Search conversations by content."""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT turn_id, character_name, timestamp, user_input, character_response
                FROM conversations
                WHERE user_id = ? AND character_name = ?
                AND (user_input LIKE ? OR character_response LIKE ?)
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (user_id, character_name, f"%{query}%", f"%{query}%", limit),
            )

            turns = []
            for row in cursor.fetchall():
                turn = ConversationTurn(
                    turn_id=row[0],
                    character_name=row[1],
                    timestamp=row[2],
                    user_input=row[3],
                    character_response=row[4],
                )
                turns.append(turn)

            return turns

    def get_session_stats(self, user_id: str, character_name: str) -> Dict[str, Any]:
        """Get conversation statistics for a user-character pair."""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()

            # Get total turns
            cursor.execute(
                """
                SELECT COUNT(*) FROM conversations
                WHERE user_id = ? AND character_name = ?
            """,
                (user_id, character_name),
            )
            total_turns = cursor.fetchone()[0]

            # Get total sessions
            cursor.execute(
                """
                SELECT COUNT(*) FROM sessions
                WHERE user_id = ? AND character_name = ?
            """,
                (user_id, character_name),
            )
            total_sessions = cursor.fetchone()[0]

            # Get last interaction
            cursor.execute(
                """
                SELECT MAX(timestamp) FROM conversations
                WHERE user_id = ? AND character_name = ?
            """,
                (user_id, character_name),
            )
            last_interaction = cursor.fetchone()[0]

            return {
                "total_turns": total_turns,
                "total_sessions": total_sessions,
                "last_interaction": last_interaction,
            }

    def cleanup_old_data(self, days: Optional[int] = None) -> int:
        """Clean up old conversation data."""
        cutoff_days = days or self.max_age_days
        cutoff_time = time.time() - (cutoff_days * 24 * 60 * 60)

        with self.pool.transaction() as conn:
            cursor = conn.cursor()

            # Delete old conversations
            cursor.execute(
                """
                DELETE FROM conversations
                WHERE timestamp < ?
            """,
                (cutoff_time,),
            )
            deleted_turns = cursor.rowcount

            # Delete old sessions
            cursor.execute(
                """
                DELETE FROM sessions
                WHERE start_time < ?
            """,
                (cutoff_time,),
            )
            deleted_sessions = cursor.rowcount

            logger.info(
                f"Cleaned up {deleted_turns} turns and {deleted_sessions} sessions"
            )

            return int(deleted_turns + deleted_sessions)

    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all data for a user (GDPR compliance)."""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()

            # Get all conversations
            cursor.execute(
                """
                SELECT turn_id, session_id, character_name, timestamp,
                       user_input, character_response, metadata
                FROM conversations
                WHERE user_id = ?
                ORDER BY timestamp
            """,
                (user_id,),
            )

            conversations = []
            for row in cursor.fetchall():
                conversations.append(
                    {
                        "turn_id": row[0],
                        "session_id": row[1],
                        "character_name": row[2],
                        "timestamp": row[3],
                        "user_input": row[4],
                        "character_response": row[5],
                        "metadata": json.loads(row[6]) if row[6] else None,
                    }
                )

            # Get all sessions
            cursor.execute(
                """
                SELECT session_id, character_name, start_time, end_time, turn_count
                FROM sessions
                WHERE user_id = ?
                ORDER BY start_time
            """,
                (user_id,),
            )

            sessions = []
            for row in cursor.fetchall():
                sessions.append(
                    {
                        "session_id": row[0],
                        "character_name": row[1],
                        "start_time": row[2],
                        "end_time": row[3],
                        "turn_count": row[4],
                    }
                )

            return {
                "user_id": user_id,
                "conversations": conversations,
                "sessions": sessions,
                "export_timestamp": time.time(),
            }

    def delete_user_data(self, user_id: str) -> bool:
        """Delete all data for a user (GDPR compliance)."""
        try:
            with self.pool.transaction() as conn:
                cursor = conn.cursor()

                # Delete conversations
                cursor.execute(
                    """
                    DELETE FROM conversations
                    WHERE user_id = ?
                """,
                    (user_id,),
                )

                # Delete sessions
                cursor.execute(
                    """
                    DELETE FROM sessions
                    WHERE user_id = ?
                """,
                    (user_id,),
                )

                logger.info(f"Deleted all data for user {user_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete data for user {user_id}: {e}")
            return False

    def vacuum_database(self) -> None:
        """Optimize database by running VACUUM."""
        with self.pool.get_connection() as conn:
            conn.execute("VACUUM")
            logger.info("Database vacuum completed")

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()

            # Get table sizes
            cursor.execute("SELECT COUNT(*) FROM conversations")
            total_turns = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
            unique_users = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT character_name) FROM conversations")
            unique_characters = cursor.fetchone()[0]

            # Get database file size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

            return {
                "total_turns": total_turns,
                "total_sessions": total_sessions,
                "unique_users": unique_users,
                "unique_characters": unique_characters,
                "database_size_bytes": db_size,
                "database_size_mb": round(db_size / (1024 * 1024), 2),
            }

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return self.pool.get_stats()

    def close(self) -> None:
        """Close the connection pool."""
        self.pool.close_all()
