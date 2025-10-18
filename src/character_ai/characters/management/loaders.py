"""
Character loading interfaces and implementations.

Provides unified character loading with different strategies for different formats.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ...core.config.yaml_loader import YAMLConfigLoader
from .types import (
    Ability,
    Archetype,
    Character,
    CharacterDimensions,
    PersonalityTrait,
    Species,
    Topic,
)

logger = logging.getLogger(__name__)


class CharacterLoaderProtocol(ABC):
    """Protocol for character loading implementations."""

    @abstractmethod
    def load_character(self, path: Path) -> Optional[Character]:
        """
        Load a character from the given path.

        Args:
            path: Path to character data (file or directory)

        Returns:
            Character object if successful, None otherwise
        """
        pass

    @abstractmethod
    def can_load(self, path: Path) -> bool:
        """
        Check if this loader can handle the given path.

        Args:
            path: Path to check

        Returns:
            True if this loader can handle the path
        """
        pass


class YAMLCharacterLoader(CharacterLoaderProtocol):
    """Loads characters from YAML files with dimensions format."""

    def can_load(self, path: Path) -> bool:
        """Check if path is a YAML file with dimensions."""
        if not path.is_file() or not path.suffix.lower() == ".yaml":
            return False

        try:
            data = YAMLConfigLoader.load_yaml(path)
            return data is not None and "dimensions" in data
        except Exception:
            return False

    def load_character(self, path: Path) -> Optional[Character]:
        """Load character from YAML file with dimensions format."""
        try:
            data = YAMLConfigLoader.load_yaml(path)

            if not data or "dimensions" not in data:
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

            # Load metadata including prompt_template from llm section and TTS config
            metadata = data.get("metadata", {})
            if "llm" in data and "prompt_template" in data["llm"]:
                metadata["prompt_template"] = data["llm"]["prompt_template"]
            if "tts" in data:
                metadata["tts_config"] = data["tts"]
                logger.info(f"Loaded TTS config for {data['name']}: {data['tts']}")

            # Extract franchise from file path (parent directory of character directory)
            franchise = "original"  # default
            if len(path.parts) >= 2:
                # Look for franchise in path structure
                for part in path.parts:
                    if part.startswith("franchise="):
                        franchise = part.split("=", 1)[1]
                        break

            # Add franchise to metadata instead of as a direct parameter
            metadata["franchise"] = franchise

            return Character(
                name=data["name"],
                dimensions=dimensions,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to load character from {path}: {e}")
            return None


class ProfileDirectoryLoader(CharacterLoaderProtocol):
    """Loads characters from profile directory structure (schema format)."""

    def can_load(self, path: Path) -> bool:
        """Check if path is a directory with profile.yaml."""
        if not path.is_dir():
            return False
        return (path / "profile.yaml").exists()

    def load_character(self, path: Path) -> Optional[Character]:
        """Load character from profile directory."""
        try:
            from .profile_models import load_profile_dir

            profile_data = load_profile_dir(path)

            # Extract franchise from directory path
            # Path structure: configs/characters/{franchise}/{character_name}/
            franchise = "original"  # default
            path_parts = path.parts
            if len(path_parts) >= 3 and path_parts[-3] == "characters":
                franchise = path_parts[
                    -2
                ]  # The franchise is the parent directory of the character

            # Load the actual profile.yaml to get the correct dimensions
            profile_path = path / "profile.yaml"
            if profile_path.exists():
                from ...core.config.yaml_loader import YAMLConfigLoader

                yaml_data = YAMLConfigLoader.load_yaml(profile_path)

                # Extract dimensions from YAML data
                dimensions_data = yaml_data.get("dimensions", {})

                # Convert profile data to Character format using actual values from YAML
                dimensions = CharacterDimensions(
                    species=Species(dimensions_data.get("species", "human")),
                    archetype=Archetype(dimensions_data.get("archetype", "adventurer")),
                    personality_traits=[
                        PersonalityTrait(trait)
                        for trait in dimensions_data.get("personality_traits", [])
                    ],
                    abilities=[
                        Ability(ability)
                        for ability in dimensions_data.get("abilities", [])
                    ],
                    topics=[
                        Topic(topic) for topic in dimensions_data.get("topics", [])
                    ],
                    backstory=profile_data.get("metadata", {}).get("backstory"),
                    goals=[],
                    fears=[],
                    likes=[],
                    dislikes=[],
                )
            else:
                # Fallback to simplified conversion if profile.yaml not found
                dimensions = CharacterDimensions(
                    species=Species.HUMAN,  # Default for profile format
                    archetype=Archetype.ADVENTURER,  # Default
                    personality_traits=[],
                    abilities=[],
                    topics=[Topic(topic) for topic in profile_data.get("topics", [])],
                    backstory=profile_data.get("metadata", {}).get("backstory"),
                    goals=[],
                    fears=[],
                    likes=[],
                    dislikes=[],
                )

            # Add processed voice path to metadata if available
            metadata = profile_data.get("metadata", {})
            # Add character ID to metadata for voice manager lookup
            if "id" in profile_data:
                metadata["id"] = profile_data["id"]

            # Add franchise to metadata
            metadata["franchise"] = franchise

            # Look for processed voice file in processed_samples directory
            processed_voice_path = (
                path / "processed_samples" / f"{profile_data['id']}_voice.wav"
            )
            if processed_voice_path.exists():
                metadata["voice_path"] = str(processed_voice_path)
                logger.info(f"Found processed voice file: {processed_voice_path}")
            elif profile_data.get("voice_path"):
                # Fallback to original voice path if no processed version
                metadata["voice_path"] = profile_data["voice_path"]

            character = Character(
                name=profile_data["name"],
                dimensions=dimensions,
                metadata=metadata,
            )

            # Set franchise as an attribute for easy access
            setattr(character, "franchise", franchise)

            return character

        except Exception as e:
            logger.error(f"Failed to load character from profile directory {path}: {e}")
            return None


class CharacterLoaderRegistry:
    """Registry for character loaders with automatic selection."""

    def __init__(self) -> None:
        """Initialize with default loaders."""
        self.loaders = [
            YAMLCharacterLoader(),
            ProfileDirectoryLoader(),
        ]

    def load_character(self, path: Path) -> Optional[Character]:
        """
        Load character using the first compatible loader.

        Args:
            path: Path to character data

        Returns:
            Character object if successful, None otherwise
        """
        for loader in self.loaders:
            if loader.can_load(path):
                logger.debug(f"Using {loader.__class__.__name__} for {path}")
                return loader.load_character(path)

        logger.warning(f"No compatible loader found for {path}")
        return None

    def add_loader(self, loader: CharacterLoaderProtocol) -> None:
        """Add a new loader to the registry."""
        self.loaders.append(loader)

    def get_compatible_loaders(self, path: Path) -> list[CharacterLoaderProtocol]:
        """Get all loaders that can handle the given path."""
        return [loader for loader in self.loaders if loader.can_load(path)]


# Global registry instance
_character_loader_registry = CharacterLoaderRegistry()


def load_character(path: Path) -> Optional[Character]:
    """
    Convenience function to load a character using the global registry.

    Args:
        path: Path to character data

    Returns:
        Character object if successful, None otherwise
    """
    return _character_loader_registry.load_character(path)


def get_character_loader_registry() -> CharacterLoaderRegistry:
    """Get the global character loader registry."""
    return _character_loader_registry
