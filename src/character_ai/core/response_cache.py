"""Response caching system for frequent interactions."""

import hashlib
import logging
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ResponseCache:
    """Cache frequent responses for instant retrieval."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[str, float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hit_count = 0
        self.miss_count = 0

    def get(self, transcribed_text: str, character_name: str) -> Optional[str]:
        """Get cached response if available and not expired."""
        cache_key = self._hash_input(transcribed_text, character_name)

        if cache_key in self.cache:
            response, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.ttl_seconds:
                self.hit_count += 1
                logger.debug(f"Cache hit for: {transcribed_text[:50]}...")
                return response
            else:
                del self.cache[cache_key]

        self.miss_count += 1
        return None

    def set(self, transcribed_text: str, character_name: str, response: str) -> None:
        """Cache a response."""
        # LRU eviction if cache full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.items(), key=lambda x: x[1][1])[0]
            del self.cache[oldest_key]

        cache_key = self._hash_input(transcribed_text, character_name)
        self.cache[cache_key] = (response, time.time())
        logger.debug(f"Cached response for: {transcribed_text[:50]}...")

    def _hash_input(self, text: str, character: str) -> str:
        """Create hash for cache lookup using transcribed text."""
        # Normalize text for better cache hits
        normalized = text.lower().strip()
        combined = f"{character}:{normalized}"
        return hashlib.md5(combined.encode()).hexdigest()

    def get_stats(self) -> Dict[str, int | float]:
        """Get cache statistics."""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        return {
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": hit_rate,
            "cached_items": len(self.cache),
        }

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        logger.info("Response cache cleared")

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed items."""
        current_time = time.time()
        expired_keys = []

        for key, (_, timestamp) in self.cache.items():
            if current_time - timestamp >= self.ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)
