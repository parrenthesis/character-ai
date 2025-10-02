"""
Character catalog management system.

Provides YAML import/export functionality for character collections with
enterprise features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .catalog_storage import CatalogStorage
from .types import Character

logger = logging.getLogger(__name__)


@dataclass
class CatalogMetadata:
    """Metadata for character catalog."""

    name: str
    version: str = "1.0"
    description: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    total_characters: int = 0
    franchises: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    author: str = ""
    license: str = ""


@dataclass
class CharacterCollection:
    """Collection of characters with metadata."""

    metadata: CatalogMetadata
    characters: List[Character] = field(default_factory=list)

    def add_character(self, character: Character) -> None:
        """Add character to collection."""
        self.characters.append(character)
        self.metadata.total_characters = len(self.characters)
        self.metadata.updated_at = datetime.now(timezone.utc).isoformat()

    def remove_character(self, character_name: str) -> bool:
        """Remove character from collection."""
        for i, char in enumerate(self.characters):
            if char.name.lower() == character_name.lower():
                del self.characters[i]
                self.metadata.total_characters = len(self.characters)
                self.metadata.updated_at = datetime.now(timezone.utc).isoformat()
                return True
        return False

    def get_character(self, character_name: str) -> Optional[Character]:
        """Get character by name."""
        for char in self.characters:
            if char.name.lower() == character_name.lower():
                return char
        return None

    def search_characters(self, criteria: Dict[str, Any]) -> List[Character]:
        """Search characters by criteria."""
        results = []

        for character in self.characters:
            if self._matches_criteria(character, criteria):
                results.append(character)

        return results

    def _matches_criteria(self, character: Character, criteria: Dict[str, Any]) -> bool:
        """Check if character matches search criteria."""
        # Species filter
        if "species" in criteria:
            if character.dimensions.species.value != criteria["species"]:
                return False

        # Archetype filter
        if "archetype" in criteria:
            if character.dimensions.archetype.value != criteria["archetype"]:
                return False

        # Personality traits filter
        if "personality_traits" in criteria:
            required_traits = set(criteria["personality_traits"])
            character_traits = set(
                trait.value for trait in character.dimensions.personality_traits
            )
            if not required_traits.issubset(character_traits):
                return False

        # Abilities filter
        if "abilities" in criteria:
            required_abilities = set(criteria["abilities"])
            character_abilities = set(
                ability.value for ability in character.dimensions.abilities
            )
            if not required_abilities.issubset(character_abilities):
                return False

        # Topics filter
        if "topics" in criteria:
            required_topics = set(criteria["topics"])
            character_topics = set(topic.value for topic in character.dimensions.topics)

            if not required_topics.issubset(character_topics):
                return False

        # Name filter
        if "name" in criteria:
            if criteria["name"].lower() not in character.name.lower():
                return False

        return True


class CharacterCatalog:
    """Character catalog management system."""

    def __init__(self, catalog_storage: Optional[CatalogStorage] = None):
        """Initialize character catalog."""
        self.catalog_storage = catalog_storage or CatalogStorage()
        self.collections: Dict[str, CharacterCollection] = {}

    async def create_collection(
        self, name: str, description: str = "", author: str = "", license: str = ""
    ) -> CharacterCollection:
        """Create new character collection."""
        metadata = CatalogMetadata(
            name=name, description=description, author=author, license=license
        )

        collection = CharacterCollection(metadata=metadata)
        self.collections[name] = collection

        logger.info(f"Created character collection: {name}")
        return collection

    async def load_collection(self, name: str) -> Optional[CharacterCollection]:
        """Load existing character collection."""
        if name in self.collections:
            return self.collections[name]

        # Try to load from storage
        try:
            # This would load from catalog storage
            # For now, return None
            return None
        except Exception as e:
            logger.error(f"Error loading collection {name}: {e}")
            return None

    async def save_collection(self, collection: CharacterCollection) -> bool:
        """Save character collection to storage."""
        try:
            # Store each character in the collection
            for character in collection.characters:
                franchise = "catalog"  # Default franchise for catalog collections
                await self.catalog_storage.store_character(character, franchise)

            # Update collection metadata
            self.collections[collection.metadata.name] = collection

            logger.info(f"Saved character collection: {collection.metadata.name}")
            return True

        except Exception as e:
            logger.error(f"Error saving collection: {e}")
            return False

    async def export_collection(
        self, collection_name: str, output_file: Optional[Path] = None
    ) -> Path:
        """Export character collection to YAML file."""
        try:
            if collection_name not in self.collections:
                raise ValueError(f"Collection '{collection_name}' not found")

            collection = self.collections[collection_name]

            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = Path(f"catalog_{collection_name}_{timestamp}.yaml")

            # Convert collection to export format
            export_data: Dict[str, Any] = {
                "catalog_metadata": {
                    "name": collection.metadata.name,
                    "version": collection.metadata.version,
                    "description": collection.metadata.description,
                    "created_at": collection.metadata.created_at,
                    "updated_at": collection.metadata.updated_at,
                    "total_characters": collection.metadata.total_characters,
                    "franchises": collection.metadata.franchises,
                    "tags": collection.metadata.tags,
                    "author": collection.metadata.author,
                    "license": collection.metadata.license,
                },
                "characters": [],
            }

            # Convert characters to dictionary format
            for character in collection.characters:
                char_data = self._character_to_export_dict(character)
                export_data["characters"].append(char_data)

            # Write to file
            with open(output_file, "w") as f:
                yaml.dump(export_data, f, default_flow_style=False)

            logger.info(f"Exported collection '{collection_name}' to {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error exporting collection: {e}")
            raise

    def _character_to_export_dict(self, character: Character) -> Dict[str, Any]:
        """Convert character to export dictionary format."""
        return {
            "name": character.name,
            "voice_style": character.voice_style,
            "language": character.language,
            "metadata": character.metadata,
            "dimensions": {
                "species": character.dimensions.species.value,
                "archetype": character.dimensions.archetype.value,
                "personality_traits": [
                    trait.value for trait in character.dimensions.personality_traits
                ],
                "abilities": [
                    ability.value for ability in character.dimensions.abilities
                ],
                "topics": [topic.value for topic in character.dimensions.topics],
                "backstory": character.dimensions.backstory,
                "goals": character.dimensions.goals,
                "fears": character.dimensions.fears,
                "likes": character.dimensions.likes,
                "dislikes": character.dimensions.dislikes,
            },
        }

    async def import_collection(self, catalog_file: Path) -> CharacterCollection:
        """Import character collection from YAML file."""
        try:
            with open(catalog_file, "r") as f:
                data = yaml.safe_load(f)

            if "catalog_metadata" not in data:
                raise ValueError(
                    "Invalid catalog format: missing 'catalog_metadata' section"
                )

            # Parse metadata
            metadata_data = data["catalog_metadata"]
            metadata = CatalogMetadata(
                name=metadata_data.get("name", "imported_catalog"),
                version=metadata_data.get("version", "1.0"),
                description=metadata_data.get("description", ""),
                created_at=metadata_data.get(
                    "created_at", datetime.now(timezone.utc).isoformat()
                ),
                updated_at=metadata_data.get(
                    "updated_at", datetime.now(timezone.utc).isoformat()
                ),
                total_characters=metadata_data.get("total_characters", 0),
                franchises=metadata_data.get("franchises", []),
                tags=metadata_data.get("tags", []),
                author=metadata_data.get("author", ""),
                license=metadata_data.get("license", ""),
            )

            # Parse characters
            characters = []
            for char_data in data.get("characters", []):
                try:
                    character = self._import_dict_to_character(char_data)
                    characters.append(character)
                except Exception as e:
                    logger.warning(f"Error importing character: {e}")
                    continue

            # Create collection
            collection = CharacterCollection(metadata=metadata, characters=characters)

            # Store in collections
            self.collections[metadata.name] = collection

            logger.info(
                f"Imported collection '{metadata.name}' with "
                f"{len(characters)} characters"
            )
            return collection

        except Exception as e:
            logger.error(f"Error importing collection: {e}")
            raise

    def _import_dict_to_character(self, char_data: Dict[str, Any]) -> Character:
        """Convert import dictionary to Character."""
        from .types import (
            Ability,
            Archetype,
            CharacterDimensions,
            PersonalityTrait,
            Species,
            Topic,
        )

        # Parse dimensions
        dimensions_data = char_data["dimensions"]
        species = Species(dimensions_data["species"])
        archetype = Archetype(dimensions_data["archetype"])

        personality_traits = [
            PersonalityTrait(trait)
            for trait in dimensions_data.get("personality_traits", [])
        ]

        abilities = [
            Ability(ability) for ability in dimensions_data.get("abilities", [])
        ]

        topics = [Topic(topic) for topic in dimensions_data.get("topics", [])]

        dimensions = CharacterDimensions(
            species=species,
            archetype=archetype,
            personality_traits=personality_traits,
            abilities=abilities,
            topics=topics,
            backstory=dimensions_data.get("backstory"),
            goals=dimensions_data.get("goals", []),
            fears=dimensions_data.get("fears", []),
            likes=dimensions_data.get("likes", []),
            dislikes=dimensions_data.get("dislikes", []),
        )

        return Character(
            name=char_data["name"],
            dimensions=dimensions,
            voice_style=char_data.get("voice_style", "neutral"),
            language=char_data.get("language", "en"),
            metadata=char_data.get("metadata", {}),
        )

    async def search_all_collections(
        self, criteria: Dict[str, Any]
    ) -> List[Tuple[str, Character]]:
        """Search across all collections."""
        results = []

        for collection_name, collection in self.collections.items():
            matching_characters = collection.search_characters(criteria)
            for character in matching_characters:
                results.append((collection_name, character))

        return results

    async def get_collection_statistics(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a specific collection."""
        if collection_name not in self.collections:
            return {}

        collection = self.collections[collection_name]

        stats: Dict[str, Any] = {
            "collection_name": collection_name,
            "total_characters": len(collection.characters),
            "metadata": {
                "name": collection.metadata.name,
                "version": collection.metadata.version,
                "description": collection.metadata.description,
                "created_at": collection.metadata.created_at,
                "updated_at": collection.metadata.updated_at,
                "author": collection.metadata.author,
                "license": collection.metadata.license,
            },
            "species_distribution": {},
            "archetype_distribution": {},
            "personality_traits_distribution": {},
            "abilities_distribution": {},
            "topics_distribution": {},
        }

        # Analyze characters
        for character in collection.characters:
            # Species distribution
            species = character.dimensions.species.value
            stats["species_distribution"][species] = (
                stats["species_distribution"].get(species, 0) + 1
            )

            # Archetype distribution
            archetype = character.dimensions.archetype.value
            stats["archetype_distribution"][archetype] = (
                stats["archetype_distribution"].get(archetype, 0) + 1
            )

            # Personality traits distribution
            for trait in character.dimensions.personality_traits:
                trait_name = trait.value
                stats["personality_traits_distribution"][trait_name] = (
                    stats["personality_traits_distribution"].get(trait_name, 0) + 1
                )

            # Abilities distribution
            for ability in character.dimensions.abilities:
                ability_name = ability.value
                stats["abilities_distribution"][ability_name] = (
                    stats["abilities_distribution"].get(ability_name, 0) + 1
                )

            # Topics distribution
            for topic in character.dimensions.topics:
                topic_name = topic.value
                stats["topics_distribution"][topic_name] = (
                    stats["topics_distribution"].get(topic_name, 0) + 1
                )

        return stats

    async def get_all_collections(self) -> List[str]:
        """Get list of all collection names."""
        return list(self.collections.keys())

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete character collection."""
        if collection_name in self.collections:
            del self.collections[collection_name]
            logger.info(f"Deleted collection: {collection_name}")
            return True
        return False
