"""
Search operations for catalog management.

Handles advanced search functionality with text search and fuzzy matching.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from ...management.types import Character

logger = logging.getLogger(__name__)


class SearchService:
    """Handles advanced character search operations."""

    def __init__(self, catalog_dir: Path, storage_manager: Any) -> None:
        self.catalog_dir = catalog_dir
        self.metadata_dir = catalog_dir / "metadata"
        self.storage_manager = storage_manager

    async def search_characters(self, query: Dict[str, Any]) -> List[Character]:
        """Advanced search characters by criteria with text search and fuzzy
        matching."""
        results = []

        try:
            # Load search index
            search_index = await self._load_search_index()

            # Handle text search
            if "text" in query:
                text_results = await self._text_search(query["text"], search_index)
                for character_data in text_results:
                    if self._matches_search_criteria(character_data, query):
                        character = await self.storage_manager.load_character(
                            character_data["name"], character_data["franchise"]
                        )
                        if character:
                            results.append(character)
            else:
                # Standard criteria search
                for character_data in search_index.get("characters", {}).values():
                    if self._matches_search_criteria(character_data, query):
                        character = await self.storage_manager.load_character(
                            character_data["name"], character_data["franchise"]
                        )
                        if character:
                            results.append(character)

            # Sort results by relevance if text search
            if "text" in query:
                results = await self._sort_by_relevance(results, query["text"])

            return results

        except Exception as e:
            logger.error(f"Error searching characters: {e}")
            return []

    def _matches_search_criteria(
        self, character_data: Dict[str, Any], criteria: Dict[str, Any]
    ) -> bool:
        """Check if character matches search criteria."""
        # Franchise filter
        if "franchise" in criteria:
            if character_data.get("franchise") != criteria["franchise"]:
                return False

        # Species filter
        if "species" in criteria:
            if character_data.get("species") != criteria["species"]:
                return False

        # Archetype filter
        if "archetype" in criteria:
            if character_data.get("archetype") != criteria["archetype"]:
                return False

        # Personality traits filter
        if "personality_traits" in criteria:
            required_traits = set(criteria["personality_traits"])
            character_traits = set(character_data.get("personality_traits", []))
            if not required_traits.issubset(character_traits):
                return False

        # Abilities filter
        if "abilities" in criteria:
            required_abilities = set(criteria["abilities"])
            character_abilities = set(character_data.get("abilities", []))
            if not required_abilities.issubset(character_abilities):
                return False

        # Topics filter
        if "topics" in criteria:
            required_topics = set(criteria["topics"])
            character_topics = set(character_data.get("topics", []))
            if not required_topics.issubset(character_topics):
                return False

        # Voice availability filter
        if "voice_available" in criteria:
            if character_data.get("voice_available") != criteria["voice_available"]:
                return False

        # Name filter
        if "name" in criteria:
            if criteria["name"].lower() not in character_data.get("name", "").lower():
                return False

        return True

    async def _load_search_index(self) -> Dict[str, Any]:
        """Load or create search index."""
        # Create metadata directory if it doesn't exist
        if not self.metadata_dir.exists():
            self.metadata_dir.mkdir(parents=True, exist_ok=True)

        search_index_file = self.metadata_dir / "search_index.json"

        if search_index_file.exists():
            try:
                with open(search_index_file, "r") as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except Exception as e:
                logger.warning(f"Error loading search index: {e}")

        # Create new search index
        search_index: dict[str, Any] = {
            "version": "1.0",
            "characters": {},
            "text_index": {},
            "last_updated": None,
        }

        # Build search index from character files
        characters_dir = self.catalog_dir / "characters"
        if characters_dir.exists():
            for franchise_dir in characters_dir.glob("franchise=*"):
                franchise_name = franchise_dir.name.replace("franchise=", "")
                for char_file in franchise_dir.glob("*.yaml"):
                    try:
                        from ....core.config.yaml_loader import YAMLConfigLoader

                        char_data = YAMLConfigLoader.load_yaml(char_file)

                        if char_data and "name" in char_data:
                            character_name = char_data["name"]
                            search_data = {
                                "name": character_name,
                                "franchise": franchise_name,
                                "species": char_data.get("dimensions", {}).get(
                                    "species", ""
                                ),
                                "archetype": char_data.get("dimensions", {}).get(
                                    "archetype", ""
                                ),
                                "personality_traits": char_data.get(
                                    "dimensions", {}
                                ).get("personality_traits", []),
                                "abilities": char_data.get("dimensions", {}).get(
                                    "abilities", []
                                ),
                                "topics": char_data.get("dimensions", {}).get(
                                    "topics", []
                                ),
                                "voice_available": char_data.get("voice", {}).get(
                                    "available", False
                                ),
                                "tags": char_data.get("tags", []),
                                "description": char_data.get("description", ""),
                                "usage_count": char_data.get("usage_count", 0),
                            }

                            search_index["characters"][
                                f"{franchise_name}:{character_name}"
                            ] = search_data

                            # Build text index
                            text_content = f"{character_name} {char_data.get('description', '')} {' '.join(char_data.get('tags', []))}"
                            words = re.findall(r"\b\w+\b", text_content.lower())
                            for word in words:
                                if word not in search_index["text_index"]:
                                    search_index["text_index"][word] = []
                                search_index["text_index"][word].append(
                                    f"{franchise_name}:{character_name}"
                                )

                    except Exception as e:
                        logger.warning(
                            f"Error processing character file {char_file}: {e}"
                        )

        # Save search index
        try:
            with open(search_index_file, "w") as f:
                json.dump(search_index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving search index: {e}")

        return search_index

    async def _text_search(
        self, query: str, search_index: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Perform text search on character data."""
        results = []
        query_words = re.findall(r"\b\w+\b", query.lower())

        # Find characters that match any query word
        matching_characters = set()
        for word in query_words:
            if word in search_index.get("text_index", {}):
                matching_characters.update(search_index["text_index"][word])

        # Get character data for matches
        for char_key in matching_characters:
            if char_key in search_index.get("characters", {}):
                results.append(search_index["characters"][char_key])

        return results

    async def _sort_by_relevance(
        self, characters: List[Character], query: str
    ) -> List[Character]:
        """Sort characters by relevance to search query."""
        import re

        query_words = set(re.findall(r"\b\w+\b", query.lower()))

        def relevance_score(character: Character) -> int:
            score = 0

            # Name match gets highest score
            if any(word in character.name.lower() for word in query_words):
                score += 100

            # Description match gets medium score
            description = getattr(character, "description", "")
            if any(word in description.lower() for word in query_words):
                score += 50

            # Tag match gets lower score
            # Note: tags are not directly accessible from Character object
            # This would need to be enhanced if tag matching is needed

            # Usage count as tiebreaker
            usage_count = getattr(character, "usage_count", 0)
            score += usage_count

            return score

        return sorted(characters, key=relevance_score, reverse=True)
