"""
Analytics operations for catalog management.

Handles comprehensive catalog statistics and analytics.

⚠️  BETA FEATURE: This analytics system is experimental and may change.
    Use with caution in production environments.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Handles catalog analytics and statistics.

    ⚠️  BETA: This service is experimental and may change in future versions.
    """

    def __init__(self, search_manager: Any) -> None:
        self.search_manager = search_manager

    async def get_catalog_statistics(self) -> Dict[str, Any]:
        """Get comprehensive catalog statistics."""
        try:
            search_index = await self.search_manager._load_search_index()

            stats: Dict[str, Any] = {
                "total_characters": len(search_index.get("characters", {})),
                "franchises": {},
                "species_distribution": {},
                "archetype_distribution": {},
                "personality_traits_distribution": {},
                "abilities_distribution": {},
                "topics_distribution": {},
                "voice_stats": {"total_with_voice": 0, "voice_availability": 0.0},
                "usage_stats": {"most_used": [], "total_usage": 0},
            }

            # Analyze characters
            for char_data in search_index.get("characters", {}).values():
                franchise = char_data.get("franchise", "unknown")
                if franchise not in stats["franchises"]:
                    stats["franchises"][franchise] = 0
                stats["franchises"][franchise] += 1

                # Species distribution
                species = char_data.get("species", "unknown")
                stats["species_distribution"][species] = (
                    stats["species_distribution"].get(species, 0) + 1
                )

                # Archetype distribution
                archetype = char_data.get("archetype", "unknown")
                stats["archetype_distribution"][archetype] = (
                    stats["archetype_distribution"].get(archetype, 0) + 1
                )

                # Personality traits distribution
                for trait in char_data.get("personality_traits", []):
                    stats["personality_traits_distribution"][trait] = (
                        stats["personality_traits_distribution"].get(trait, 0) + 1
                    )

                # Abilities distribution
                for ability in char_data.get("abilities", []):
                    stats["abilities_distribution"][ability] = (
                        stats["abilities_distribution"].get(ability, 0) + 1
                    )

                # Topics distribution
                for topic in char_data.get("topics", []):
                    stats["topics_distribution"][topic] = (
                        stats["topics_distribution"].get(topic, 0) + 1
                    )

                # Voice stats
                if char_data.get("voice_available", False):
                    stats["voice_stats"]["total_with_voice"] += 1

                # Usage stats
                usage_count = char_data.get("usage_count", 0)
                stats["usage_stats"]["total_usage"] += usage_count
                if usage_count > 0:
                    stats["usage_stats"]["most_used"].append(
                        {"name": char_data["name"], "usage_count": usage_count}
                    )

            # Calculate voice availability percentage
            total_chars = stats["total_characters"]
            if total_chars > 0:
                stats["voice_stats"]["voice_availability"] = (
                    stats["voice_stats"]["total_with_voice"] / total_chars
                )

            # Sort most used characters
            stats["usage_stats"]["most_used"].sort(
                key=lambda x: x["usage_count"], reverse=True
            )
            stats["usage_stats"]["most_used"] = stats["usage_stats"]["most_used"][
                :10
            ]  # Top 10

            return stats

        except Exception as e:
            logger.error(f"Error getting catalog statistics: {e}")
            return {}
