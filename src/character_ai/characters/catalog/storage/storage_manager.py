"""
Core storage operations for catalog management.

Handles character storage, loading, and data conversion.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ....core.config.yaml_loader import YAMLConfigLoader
from ...management.manager import CharacterService
from ...management.types import Character

logger = logging.getLogger(__name__)


class CharacterRepository:
    """Handles core character storage and retrieval operations."""

    def __init__(self, catalog_dir: Path, character_manager: CharacterService):
        self.catalog_dir = catalog_dir
        self.characters_dir = catalog_dir / "characters"
        self.index_file = catalog_dir / "index.json"
        self.character_manager = character_manager

    def _load_or_create_index(self) -> Dict[str, Any]:
        """Load existing index or create new one."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    return dict(json.load(f))
            except Exception as e:
                logger.warning(f"Error loading index: {e}, creating new index")

        # Create new index
        return {
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_characters": 0,
            "franchises": {},
            "voice_stats": {"total_with_voice": 0, "voice_availability": 0.0},
            "usage_stats": {"most_used": [], "recent_adaptations": 0},
        }

    async def store_character(
        self, character: Character, index: Dict[str, Any], franchise: str = "original"
    ) -> str:
        """Store character with franchise organization."""
        try:
            # Create directories if they don't exist
            if not self.characters_dir.exists():
                self.characters_dir.mkdir(parents=True, exist_ok=True)

            # Create franchise directory
            franchise_dir = self.characters_dir / f"franchise={franchise}"
            franchise_dir.mkdir(exist_ok=True)

            # Save character file
            char_file = (
                franchise_dir / f"{character.name.lower().replace(' ', '_')}.yaml"
            )
            char_data = self._character_to_dict(character, franchise)

            logger.debug(f"Storing character data: {char_data}")

            with open(char_file, "w") as f:
                yaml.dump(char_data, f, default_flow_style=False)

            # Update index
            await self._update_index(character.name, franchise, str(char_file), index)

            logger.info(
                f"Stored character '{character.name}' in franchise '{franchise}'"
            )
            return str(char_file)

        except Exception as e:
            logger.error(f"Error storing character: {e}")
            raise

    def _character_to_dict(
        self, character: Character, franchise: str
    ) -> Dict[str, Any]:
        """Convert Character to dictionary for storage."""

        # Convert dimensions
        dimensions_data = {
            "species": character.dimensions.species.value,
            "archetype": character.dimensions.archetype.value,
            "personality_traits": [
                trait.value for trait in character.dimensions.personality_traits
            ],
            "abilities": [ability.value for ability in character.dimensions.abilities],
            "topics": [topic.value for topic in character.dimensions.topics],
            "backstory": character.dimensions.backstory,
            "goals": character.dimensions.goals,
            "fears": character.dimensions.fears,
            "likes": character.dimensions.likes,
            "dislikes": character.dimensions.dislikes,
        }

        # Convert relationships
        relationships_data = []
        for rel in character.relationships:
            relationships_data.append(
                {
                    "character_name": rel.character,
                    "relationship_type": rel.relationship,
                    "description": rel.description,
                    "strength": rel.strength,
                }
            )

        # Convert localizations
        localizations_data = []
        for loc in character.localizations:
            localizations_data.append(
                {
                    "language": loc.language,
                    "name": loc.name,
                    "description": loc.description,
                    "backstory": loc.backstory,
                }
            )

        # Convert licensing
        licensing_data = {}
        if character.licensing:
            licensing_data = {
                "license_type": character.licensing.license_type,
                "owner": character.licensing.owner,
                "rights": character.licensing.rights,
                "restrictions": character.licensing.restrictions,
                "expiration": character.licensing.expiration,
                "territories": character.licensing.territories,
            }

        # Build complete character data
        char_data = {
            "name": character.name,
            "voice_style": character.voice_style,
            "language": character.language,
            "franchise": franchise,
            "dimensions": dimensions_data,
            "relationships": relationships_data,
            "localizations": localizations_data,
            "licensing": licensing_data,
            "metadata": character.metadata,
        }

        return char_data

    def _extract_tags(self, character: Character) -> list[str]:
        """Extract searchable tags from character."""
        tags = []

        # Add name and description words
        tags.extend(character.name.lower().split())
        description = getattr(character, "description", "")
        tags.extend(description.lower().split())

        # Add dimension tags
        tags.append(character.dimensions.species.value.lower())
        tags.append(character.dimensions.archetype.value.lower())

        for trait in character.dimensions.personality_traits:
            tags.append(trait.value.lower())

        for ability in character.dimensions.abilities:
            tags.append(ability.value.lower())

        for topic in character.dimensions.topics:
            tags.append(topic.value.lower())

        # Add relationship tags
        for rel in character.relationships:
            character_name = getattr(
                rel, "character_name", getattr(rel, "character", "")
            )
            relationship_type = getattr(
                rel, "relationship_type", getattr(rel, "relationship", "")
            )
            tags.append(character_name.lower())
            tags.append(relationship_type.lower())

        # Add localization tags
        for loc in character.localizations:
            tags.append(loc.language.lower())
            tags.append(loc.name.lower())

        # Remove duplicates and empty strings
        return list(set(tag for tag in tags if tag.strip()))

    async def _update_index(
        self, character_name: str, franchise: str, char_file: str, index: Dict[str, Any]
    ) -> None:
        """Update catalog index with new character."""
        try:
            # Update franchise info
            if franchise not in index["franchises"]:
                index["franchises"][franchise] = {"count": 0, "characters": []}

            if character_name not in index["franchises"][franchise]["characters"]:
                index["franchises"][franchise]["characters"].append(character_name)

                index["franchises"][franchise]["count"] += 1
                index["total_characters"] += 1

            # Update timestamp
            index["last_updated"] = datetime.now(timezone.utc).isoformat()

            # Save index
            with open(self.index_file, "w") as f:
                json.dump(index, f, indent=2)

        except Exception as e:
            logger.error(f"Error updating index: {e}")

    async def load_character(
        self, character_name: str, franchise: str = "original"
    ) -> Optional[Character]:
        """Load character from catalog storage."""
        try:
            franchise_dir = self.characters_dir / f"franchise={franchise}"
            char_file = (
                franchise_dir / f"{character_name.lower().replace(' ', '_')}.yaml"
            )

            if not char_file.exists():
                logger.warning(f"Character file not found: {char_file}")
                return None

            data = YAMLConfigLoader.load_yaml(char_file)

            if not data:
                logger.error(f"YAML file is empty or invalid: {char_file}")
                return None

            if "dimensions" not in data:
                logger.error(f"Character data missing 'dimensions' field: {data}")
                return None

            # Convert back to Character
            return self._dict_to_character(data)

        except Exception as e:
            logger.error(f"Error loading character: {e}")
            return None

    def _dict_to_character(self, data: Dict[str, Any]) -> Character:
        """Convert dictionary back to Character."""
        from ...management.types import (
            Ability,
            Archetype,
            CharacterDimensions,
            CharacterLicensing,
            CharacterLocalization,
            CharacterRelationship,
            PersonalityTrait,
            Species,
            Topic,
        )

        if data is None:
            raise ValueError("Data is None in _dict_to_character")

        # Parse dimensions
        dimensions_data = data.get("dimensions", {})
        if not dimensions_data:
            raise ValueError(f"Missing or empty dimensions data: {data}")

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

        # Parse enterprise features
        relationships = []
        for rel_data in data.get("relationships", []):
            if rel_data is None:
                continue
            relationships.append(
                CharacterRelationship(
                    character=rel_data.get("character_name", ""),
                    relationship=rel_data.get("relationship_type", ""),
                    description=rel_data.get("description", ""),
                    strength=rel_data.get("strength", 0.5),
                )
            )

        localizations = []
        for loc_data in data.get("localizations", []):
            if loc_data is None:
                continue
            localizations.append(
                CharacterLocalization(
                    language=loc_data.get("language", ""),
                    name=loc_data.get("name", ""),
                    description=loc_data.get("description", ""),
                    backstory=loc_data.get("backstory", ""),
                )
            )

        # Parse licensing
        licensing_data = data.get("licensing", {})
        licensing = CharacterLicensing(
            owner=licensing_data.get("owner", ""),
            license_type=licensing_data.get("license_type", "proprietary"),
            rights=licensing_data.get("rights", []),
            restrictions=licensing_data.get("restrictions", []),
            expiration=licensing_data.get("expiration", ""),
            territories=licensing_data.get("territories", []),
        )

        # Voice data is not part of the Character class

        # Parse timestamps
        if data.get("created_at"):
            try:
                datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            except ValueError:
                pass

        if data.get("updated_at"):
            try:
                datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))
            except ValueError:
                pass

        if data.get("last_used"):
            try:
                datetime.fromisoformat(data["last_used"].replace("Z", "+00:00"))
            except ValueError:
                pass

        # Create Character object
        return Character(
            name=data["name"],
            dimensions=dimensions,
            voice_style=data.get("voice_style", "neutral"),
            language=data.get("language", "en"),
            metadata=data.get("metadata", {}),
            relationships=relationships,
            localizations=localizations,
            licensing=licensing,
        )
