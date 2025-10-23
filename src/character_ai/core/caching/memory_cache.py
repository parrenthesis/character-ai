"""
Extended caching for memory system components.

Extends ResponseCache to support caching of summaries, preferences, and
other memory-related data with smart cache management.
"""

import hashlib
import logging
import time
from typing import Any, Dict, Optional, Tuple

from .response_cache import ResponseCache

logger = logging.getLogger(__name__)


class MemoryCache(ResponseCache):
    """Extended cache for memory system components."""

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        summary_ttl_seconds: int = 7200,  # Summaries last longer
        preference_ttl_seconds: int = 86400,  # Preferences last 24 hours
    ):
        """Initialize with different TTL settings for different data types."""
        super().__init__(max_size, ttl_seconds)
        self.summary_ttl = summary_ttl_seconds
        self.preference_ttl = preference_ttl_seconds

        # Separate caches for different data types
        self.summary_cache: Dict[str, Tuple[str, float]] = {}
        self.preference_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}

    def cache_summary(
        self, user_id: str, character_name: str, summary_text: str
    ) -> None:
        """Cache a conversation summary."""
        cache_key = self._hash_memory_key(user_id, character_name, "summary")
        self.summary_cache[cache_key] = (summary_text, time.time())
        logger.debug(f"Cached summary for {user_id} with {character_name}")

    def get_cached_summary(self, user_id: str, character_name: str) -> Optional[str]:
        """Get cached summary if available and not expired."""
        cache_key = self._hash_memory_key(user_id, character_name, "summary")

        if cache_key in self.summary_cache:
            summary_text, timestamp = self.summary_cache[cache_key]
            if time.time() - timestamp < self.summary_ttl:
                logger.debug(f"Cache hit for summary: {user_id} with {character_name}")
                return summary_text
            else:
                del self.summary_cache[cache_key]

        return None

    def cache_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        """Cache user preferences."""
        cache_key = self._hash_memory_key(user_id, "preferences", "prefs")
        self.preference_cache[cache_key] = (preferences, time.time())
        logger.debug(f"Cached preferences for {user_id}")

    def get_cached_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached preferences if available and not expired."""
        cache_key = self._hash_memory_key(user_id, "preferences", "prefs")

        if cache_key in self.preference_cache:
            preferences, timestamp = self.preference_cache[cache_key]
            if time.time() - timestamp < self.preference_ttl:
                logger.debug(f"Cache hit for preferences: {user_id}")
                return preferences
            else:
                del self.preference_cache[cache_key]

        return None

    def cache_conversation_context(
        self, user_id: str, character_name: str, context: str
    ) -> None:
        """Cache conversation context for faster retrieval."""
        cache_key = self._hash_memory_key(user_id, character_name, "context")
        self.cache[cache_key] = (context, time.time())
        logger.debug(f"Cached context for {user_id} with {character_name}")

    def get_cached_context(self, user_id: str, character_name: str) -> Optional[str]:
        """Get cached conversation context."""
        cache_key = self._hash_memory_key(user_id, character_name, "context")
        return self.get(cache_key, character_name)  # Reuse parent method

    def _hash_memory_key(
        self, user_id: str, character_name: str, data_type: str
    ) -> str:
        """Create hash for memory cache keys."""
        combined = f"{user_id}:{character_name}:{data_type}"
        return hashlib.md5(  # nosec B324
            combined.encode(), usedforsecurity=False
        ).hexdigest()

    def cleanup_expired_memory(self) -> Dict[str, int]:
        """Clean up expired memory cache entries."""
        current_time = time.time()
        cleanup_stats = {
            "summary_expired": 0,
            "preference_expired": 0,
            "context_expired": 0,
        }

        # Cleanup summary cache
        expired_summaries = []
        for key, (_, timestamp) in self.summary_cache.items():
            if current_time - timestamp >= self.summary_ttl:
                expired_summaries.append(key)

        for key in expired_summaries:
            del self.summary_cache[key]
        cleanup_stats["summary_expired"] = len(expired_summaries)

        # Cleanup preference cache
        expired_preferences = []
        for key, (_, timestamp) in self.preference_cache.items():
            if current_time - timestamp >= self.preference_ttl:
                expired_preferences.append(key)

        for key in expired_preferences:
            del self.preference_cache[key]
        cleanup_stats["preference_expired"] = len(expired_preferences)

        # Cleanup context cache (use parent method)
        context_expired = self.cleanup_expired()
        cleanup_stats["context_expired"] = context_expired

        if any(cleanup_stats.values()):
            logger.debug(f"Cleaned up expired memory cache: {cleanup_stats}")

        return cleanup_stats

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        base_stats = self.get_stats()

        memory_stats = {
            "base_cache": base_stats,
            "summary_cache_size": len(self.summary_cache),
            "preference_cache_size": len(self.preference_cache),
            "summary_ttl_seconds": self.summary_ttl,
            "preference_ttl_seconds": self.preference_ttl,
        }

        return memory_stats

    def clear_memory_cache(self) -> None:
        """Clear all memory caches."""
        self.clear()  # Clear base cache
        self.summary_cache.clear()
        self.preference_cache.clear()
        logger.info("Cleared all memory caches")
