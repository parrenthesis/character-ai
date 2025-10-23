"""
SQLite-based persistent storage for conversation history.

Provides indexed storage, session tracking, and search capabilities for
conversation turns with automatic cleanup and GDPR compliance.
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class ConversationStorage:
    """SQLite-based storage for conversation history."""

    def __init__(self, db_path: str, max_age_days: int = 30):
        """Initialize with database path and cleanup settings."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_age_days = max_age_days
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create conversations table
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
                    turn_count INTEGER DEFAULT 0
                )
            """
            )

            # Create indexes for efficient queries
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_user_character
                ON conversations (user_id, character_name, timestamp)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_session
                ON conversations (session_id)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_timestamp
                ON conversations (timestamp)
            """
            )

            conn.commit()

    def start_session(self, session_id: str, user_id: str, character_name: str) -> None:
        """Start a new conversation session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO sessions
                (session_id, user_id, character_name, start_time, end_time, turn_count)
                VALUES (?, ?, ?, ?, NULL, 0)
            """,
                (session_id, user_id, character_name, time.time()),
            )

            conn.commit()
            logger.debug(f"Started session {session_id} for user {user_id}")

    def end_session(self, session_id: str) -> None:
        """End a conversation session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE sessions
                SET end_time = ?
                WHERE session_id = ?
            """,
                (time.time(), session_id),
            )

            conn.commit()
            logger.debug(f"Ended session {session_id}")

    def store_turn(
        self,
        turn: ConversationTurn,
        session_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a conversation turn."""
        with sqlite3.connect(self.db_path) as conn:
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

            conn.commit()
            logger.debug(f"Stored turn {turn.turn_id} in session {session_id}")

    def get_recent_turns(
        self, user_id: str, character_name: str, limit: int = 10
    ) -> List[ConversationTurn]:
        """Get recent conversation turns for a user-character pair."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT turn_id, timestamp, user_input, character_response, character_name
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
                    timestamp=row[1],
                    user_input=row[2],
                    character_response=row[3],
                    character_name=row[4],
                )
                turns.append(turn)

            return turns

    def get_session_turns(self, session_id: str) -> List[ConversationTurn]:
        """Get all turns for a specific session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT turn_id, timestamp, user_input, character_response, character_name
                FROM conversations
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """,
                (session_id,),
            )

            turns = []
            for row in cursor.fetchall():
                turn = ConversationTurn(
                    turn_id=row[0],
                    timestamp=row[1],
                    user_input=row[2],
                    character_response=row[3],
                    character_name=row[4],
                )
                turns.append(turn)

            return turns

    def search_conversations(
        self, user_id: str, character_name: str, query: str, limit: int = 20
    ) -> List[ConversationTurn]:
        """Search conversation history for a user-character pair."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT turn_id, timestamp, user_input, character_response, character_name
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
                    timestamp=row[1],
                    user_input=row[2],
                    character_response=row[3],
                    character_name=row[4],
                )
                turns.append(turn)

            return turns

    def get_conversation_stats(
        self, user_id: str, character_name: str
    ) -> Dict[str, Any]:
        """Get conversation statistics for a user-character pair."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total turns
            cursor.execute(
                """
                SELECT COUNT(*) FROM conversations
                WHERE user_id = ? AND character_name = ?
            """,
                (user_id, character_name),
            )
            total_turns = cursor.fetchone()[0]

            # Total sessions
            cursor.execute(
                """
                SELECT COUNT(*) FROM sessions
                WHERE user_id = ? AND character_name = ?
            """,
                (user_id, character_name),
            )
            total_sessions = cursor.fetchone()[0]

            # First and last interaction
            cursor.execute(
                """
                SELECT MIN(timestamp), MAX(timestamp) FROM conversations
                WHERE user_id = ? AND character_name = ?
            """,
                (user_id, character_name),
            )
            first_last = cursor.fetchone()
            first_interaction = first_last[0] if first_last[0] else None
            last_interaction = first_last[1] if first_last[1] else None

            return {
                "user_id": user_id,
                "character_name": character_name,
                "total_turns": total_turns,
                "total_sessions": total_sessions,
                "first_interaction": first_interaction,
                "last_interaction": last_interaction,
            }

    def cleanup_old_data(self) -> int:
        """Remove old conversation data based on max_age_days."""
        cutoff_time = time.time() - (self.max_age_days * 24 * 60 * 60)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Count turns to be deleted
            cursor.execute(
                """
                SELECT COUNT(*) FROM conversations
                WHERE timestamp < ?
            """,
                (cutoff_time,),
            )
            turns_to_delete = cursor.fetchone()[0]

            # Delete old turns
            cursor.execute(
                """
                DELETE FROM conversations
                WHERE timestamp < ?
            """,
                (cutoff_time,),
            )

            # Delete orphaned sessions
            cursor.execute(
                """
                DELETE FROM sessions
                WHERE session_id NOT IN (
                    SELECT DISTINCT session_id FROM conversations
                )
            """
            )

            conn.commit()

            if turns_to_delete > 0:
                logger.info(f"Cleaned up {turns_to_delete} old conversation turns")

            return int(turns_to_delete)

    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all conversation data for a user (GDPR compliance)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get all conversations
            cursor.execute(
                """
                SELECT turn_id, session_id, character_name, timestamp,
                       user_input, character_response, metadata
                FROM conversations
                WHERE user_id = ?
                ORDER BY timestamp ASC
            """,
                (user_id,),
            )

            conversations = []
            for row in cursor.fetchall():
                conversation = {
                    "turn_id": row[0],
                    "session_id": row[1],
                    "character_name": row[2],
                    "timestamp": row[3],
                    "user_input": row[4],
                    "character_response": row[5],
                    "metadata": json.loads(row[6]) if row[6] else None,
                }
                conversations.append(conversation)

            # Get all sessions
            cursor.execute(
                """
                SELECT session_id, character_name, start_time, end_time, turn_count
                FROM sessions
                WHERE user_id = ?
                ORDER BY start_time ASC
            """,
                (user_id,),
            )

            sessions = []
            for row in cursor.fetchall():
                session = {
                    "session_id": row[0],
                    "character_name": row[1],
                    "start_time": row[2],
                    "end_time": row[3],
                    "turn_count": row[4],
                }
                sessions.append(session)

            return {
                "user_id": user_id,
                "conversations": conversations,
                "sessions": sessions,
                "export_timestamp": time.time(),
            }

    def delete_user_data(self, user_id: str) -> bool:
        """Delete all conversation data for a user (GDPR compliance)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Delete conversations
                cursor.execute(
                    "DELETE FROM conversations WHERE user_id = ?", (user_id,)
                )
                conversations_deleted = cursor.rowcount

                # Delete sessions
                cursor.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
                sessions_deleted = cursor.rowcount

                conn.commit()

                logger.info(
                    f"Deleted {conversations_deleted} conversations and "
                    f"{sessions_deleted} sessions for user {user_id}"
                )

                return True

        except sqlite3.Error as e:
            logger.error(f"Failed to delete data for user {user_id}: {e}")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Count conversations
            cursor.execute("SELECT COUNT(*) FROM conversations")
            total_conversations = cursor.fetchone()[0]

            # Count sessions
            cursor.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = cursor.fetchone()[0]

            # Count unique users
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
            unique_users = cursor.fetchone()[0]

            # Database size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

            return {
                "total_conversations": total_conversations,
                "total_sessions": total_sessions,
                "unique_users": unique_users,
                "database_size_bytes": db_size,
                "max_age_days": self.max_age_days,
            }
