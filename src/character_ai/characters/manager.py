"""
Character manager with multi-dimensional character support.

Provides advanced character management with AI generation, templates, and search.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .ai_generator import AICharacterGenerator
from .types import (
    CHARACTER_TEMPLATES,
    Ability,
    Archetype,
    Character,
    CharacterDimensions,
    CharacterTemplate,
    PersonalityTrait,
    Species,
    Topic,
)

logger = logging.getLogger(__name__)


class CharacterManager:
    """Character manager with multi-dimensional support."""

    def __init__(self) -> None:
        """Initialize the character manager."""
        self.characters: Dict[str, Character] = {}
        self.templates: Dict[str, CharacterTemplate] = CHARACTER_TEMPLATES.copy()
        self.ai_generator = AICharacterGenerator()
        self.characters_dir = Path.cwd() / "characters"
        # Don't create directory during instantiation - create when actually needed
        self._active_character: Optional[str] = None

    async def initialize(self) -> None:
        """Initialize the character manager."""
        try:
            # Load characters
            await self._load_characters()

            logger.info(
                f"Character manager initialized with {len(self.characters)} characters"
            )

        except Exception as e:
            logger.error(f"Error initializing character manager: {e}")

    async def _load_characters(self) -> None:
        """Load characters from storage."""
        try:
            # Create directory if it doesn't exist
            if not self.characters_dir.exists():
                self.characters_dir.mkdir(parents=True, exist_ok=True)

            for char_file in self.characters_dir.glob("*.yaml"):
                character = await self._load_character_from_file(char_file)
                if character:
                    self.characters[character.name.lower()] = character

            logger.info(f"Loaded {len(self.characters)} characters")

        except Exception as e:
            logger.error(f"Error loading characters: {e}")

    async def _load_character_from_file(self, file_path: Path) -> Optional[Character]:
        """Load a character from a YAML file."""
        try:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)

            if not data:
                return None

            # Check if it's a character with dimensions
            if "dimensions" not in data:
                return None

            dimensions_data = data["dimensions"]

            # Parse dimensions
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

            character = Character(
                name=data["name"],
                dimensions=dimensions,
                voice_style=data.get("voice_style", "neutral"),
                language=data.get("language", "en"),
                metadata=data.get("metadata", {}),
            )

            return character

        except Exception as e:
            logger.error(f"Error loading character from {file_path}: {e}")
            return None

    async def create_character_from_template(
        self, template_name: str, custom_name: Optional[str] = None
    ) -> Optional[Character]:
        """Create a character from a template."""
        try:
            if template_name not in self.templates:
                logger.error(f"Template not found: {template_name}")
                return None

            template = self.templates[template_name]
            character = template.create_character(custom_name)

            # Save character
            await self._save_character(character)
            self.characters[character.name.lower()] = character

            logger.info(
                f"Created character '{character.name}' from template '{template_name}'"
            )
            return character

        except Exception as e:
            logger.error(f"Error creating character from template: {e}")
            return None

    async def generate_character_from_description(
        self, description: str, custom_name: Optional[str] = None
    ) -> Optional[Character]:
        """Generate a character from a natural language description."""
        try:
            character = await self.ai_generator.generate_from_description(
                description, custom_name
            )

            if character:
                # Save character
                await self._save_character(character)
                self.characters[character.name.lower()] = character

                logger.info(f"Generated character '{character.name}' from description")

            return character

        except Exception as e:
            logger.error(f"Error generating character: {e}")
            return None

    async def _save_character(self, character: Character) -> None:
        """Save a character to storage."""
        try:
            # Convert to dictionary
            char_data = {
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

            # Create directory if it doesn't exist
            if not self.characters_dir.exists():
                self.characters_dir.mkdir(parents=True, exist_ok=True)

            # Save to file
            char_file = (
                self.characters_dir / f"{character.name.lower().replace(' ', '_')}.yaml"
            )
            with open(char_file, "w") as f:
                yaml.dump(char_data, f, default_flow_style=False)

            logger.info(f"Saved character '{character.name}' to {char_file}")

        except Exception as e:
            logger.error(f"Error saving character: {e}")
            raise

    def get_character(self, name: str) -> Optional[Character]:
        """Get a character by name."""
        return self.characters.get(name.lower())

    def list_characters(self) -> List[Character]:
        """List all characters."""
        return list(self.characters.values())

    def search_characters(self, query: Dict[str, Any]) -> List[Character]:
        """Search characters by criteria."""
        results = []

        for character in self.characters.values():
            if self._matches_search_criteria(character, query):
                results.append(character)

        return results

    def _matches_search_criteria(
        self, character: Character, criteria: Dict[str, Any]
    ) -> bool:
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

    def get_character_suggestions(
        self, user_preferences: Dict[str, Any]
    ) -> List[CharacterTemplate]:
        """Get character suggestions based on user preferences."""
        return self.ai_generator.get_character_suggestions(user_preferences)

    def get_available_templates(self) -> List[CharacterTemplate]:
        """Get all available character templates."""
        return list(self.templates.values())

    def get_template(self, template_name: str) -> Optional[CharacterTemplate]:
        """Get a specific template."""
        return self.templates.get(template_name)

    def add_custom_template(self, template: CharacterTemplate) -> None:
        """Add a custom template."""
        self.templates[template.name.lower().replace(" ", "_")] = template
        logger.info(f"Added custom template: {template.name}")

    def remove_character(self, name: str) -> bool:
        """Remove a character."""
        try:
            if name.lower() not in self.characters:
                return False

            # Remove from memory
            del self.characters[name.lower()]

            # Remove file
            char_file = self.characters_dir / f"{name.lower().replace(' ', '_')}.yaml"
            if char_file.exists():
                char_file.unlink()

            logger.info(f"Removed character: {name}")
            return True

        except Exception as e:
            logger.error(f"Error removing character: {e}")
            return False

    def get_character_statistics(self) -> Dict[str, Any]:
        """Get statistics about characters."""
        stats: Dict[str, Any] = {
            "total_characters": len(self.characters),
            "species_distribution": {},
            "archetype_distribution": {},
            "personality_traits_distribution": {},
            "abilities_distribution": {},
            "topics_distribution": {},
        }

        for character in self.characters.values():
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

    def get_available_characters(self) -> List[str]:
        """Get list of available character names."""
        return list(self.characters.keys())

    def get_active_character(self) -> Optional[Character]:
        """Get the currently active character."""
        # For now, return the first character or None
        # This would need proper active character tracking
        return next(iter(self.characters.values())) if self.characters else None

    def get_character_info(
        self, character_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get character information."""
        if character_name:
            character = self.get_character(character_name)
            if character:
                return {
                    "name": character.name,
                    "voice_style": character.voice_style,
                    "language": character.language,
                    "dimensions": {
                        "species": character.dimensions.species.value,
                        "archetype": character.dimensions.archetype.value,
                        "personality_traits": [
                            trait.value
                            for trait in character.dimensions.personality_traits
                        ],
                    },
                }
            return {}
        else:
            # Return info for all characters
            return {
                name: {
                    "name": char.name,
                    "voice_style": char.voice_style,
                    "language": char.language,
                }
                for name, char in self.characters.items()
            }

    async def reload_profiles(self) -> bool:
        """Reload character profiles from storage."""
        try:
            await self._load_characters()
            return True
        except Exception as e:
            logger.error(f"Error reloading profiles: {e}")
            return False

    async def set_active_character(self, character_name: str) -> bool:
        """Set the active character."""
        try:
            character = self.get_character(character_name)
            if character:
                self._active_character = character_name
                return True
            return False
        except Exception as e:
            logger.error(f"Error setting active character: {e}")
            return False
