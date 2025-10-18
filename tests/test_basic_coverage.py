"""
Basic tests to improve test coverage.
"""

from unittest.mock import Mock, patch

from src.character_ai.characters.ai_generator import AICharacterGenerator
from src.character_ai.characters.management.types import (
    Ability,
    Archetype,
    Character,
    CharacterDimensions,
    PersonalityTrait,
    Species,
    Topic,
)
from src.character_ai.characters.management.validation import CharacterValidator


class TestBasicCoverage:
    """Basic tests for coverage improvement."""

    def test_ai_generator_initialization(self) -> None:
        """Test AI generator can be initialized."""
        with (
            patch(
                "src.character_ai.characters.ai_generator.LLMFactory"
            ) as mock_factory,
            patch(
                "src.character_ai.characters.ai_generator.CharacterValidator"
            ) as mock_validator,
        ):
            mock_llm = Mock()
            mock_factory.return_value.get_character_creation_llm.return_value = mock_llm

            mock_validator_instance = Mock()
            mock_validator.return_value = mock_validator_instance

            generator = AICharacterGenerator()

            assert generator.llm == mock_llm
            assert generator.validator == mock_validator_instance

    def test_character_validator_initialization(self) -> None:
        """Test character validator can be initialized."""
        validator = CharacterValidator()
        assert validator is not None

    def test_enhanced_character_creation(self) -> None:
        """Test enhanced character can be created."""
        dimensions = CharacterDimensions(
            species=Species.ROBOT,
            archetype=Archetype.COMPANION,
            personality_traits=[PersonalityTrait.FRIENDLY],
            abilities=[Ability.TEACHING],
            topics=[Topic.FRIENDSHIP],
            backstory="A friendly robot",
            goals=["Help people"],
            fears=["Being unhelpful"],
            likes=["Teaching"],
            dislikes=["Rudeness"],
        )

        character = Character(name="TestBot", dimensions=dimensions)

        assert character.name == "TestBot"
        assert character.dimensions.species == Species.ROBOT
        assert character.dimensions.archetype == Archetype.COMPANION

    def test_character_relationships(self) -> None:
        """Test character relationship functionality."""
        dimensions = CharacterDimensions(
            species=Species.ROBOT,
            archetype=Archetype.COMPANION,
            personality_traits=[PersonalityTrait.FRIENDLY],
            abilities=[Ability.TEACHING],
            topics=[Topic.FRIENDSHIP],
            backstory="A friendly robot",
            goals=["Help people"],
            fears=["Being unhelpful"],
            likes=["Teaching"],
            dislikes=["Rudeness"],
        )

        character = Character(name="TestBot", dimensions=dimensions)

        # Test adding relationships
        character.add_relationship("Sidekick", "friend", 0.8, "A loyal companion")
        assert len(character.relationships) == 1
        assert character.relationships[0].character == "Sidekick"
        assert character.relationships[0].relationship == "friend"

    def test_character_localization(self) -> None:
        """Test character localization functionality."""
        dimensions = CharacterDimensions(
            species=Species.ROBOT,
            archetype=Archetype.COMPANION,
            personality_traits=[PersonalityTrait.FRIENDLY],
            abilities=[Ability.TEACHING],
            topics=[Topic.FRIENDSHIP],
            backstory="A friendly robot",
            goals=["Help people"],
            fears=["Being unhelpful"],
            likes=["Teaching"],
            dislikes=["Rudeness"],
        )

        character = Character(name="TestBot", dimensions=dimensions)

        # Test adding localizations
        character.add_localization(
            "es", "RobotTest", "Un robot amigable", "Un robot que ayuda"
        )
        assert len(character.localizations) == 1
        assert character.localizations[0].language == "es"
        assert character.localizations[0].name == "RobotTest"

    def test_character_licensing(self) -> None:
        """Test character licensing functionality."""
        dimensions = CharacterDimensions(
            species=Species.ROBOT,
            archetype=Archetype.COMPANION,
            personality_traits=[PersonalityTrait.FRIENDLY],
            abilities=[Ability.TEACHING],
            topics=[Topic.FRIENDSHIP],
            backstory="A friendly robot",
            goals=["Help people"],
            fears=["Being unhelpful"],
            likes=["Teaching"],
            dislikes=["Rudeness"],
        )

        character = Character(name="TestBot", dimensions=dimensions)

        # Test setting licensing
        character.set_licensing(
            owner="ToyCorp",
            rights=["toy", "media"],
            restrictions=["no-violence"],
            expiration="2030-12-31",
            territories=["US", "EU"],
            license_type="exclusive",
        )

        assert character.licensing is not None
        assert character.licensing.owner == "ToyCorp"
        assert "toy" in character.licensing.rights
        assert character.has_right("media") is True
        assert character.is_restricted_by("no-violence") is True

    def test_character_voice_characteristics(self) -> None:
        """Test character voice characteristics."""
        dimensions = CharacterDimensions(
            species=Species.ROBOT,
            archetype=Archetype.COMPANION,
            personality_traits=[PersonalityTrait.FRIENDLY],
            abilities=[Ability.TEACHING],
            topics=[Topic.FRIENDSHIP],
            backstory="A friendly robot",
            goals=["Help people"],
            fears=["Being unhelpful"],
            likes=["Teaching"],
            dislikes=["Rudeness"],
        )

        character = Character(
            name="TestBot", dimensions=dimensions, voice_style="robotic"
        )

        voice_chars = character.get_voice_characteristics()
        assert voice_chars["voice_style"] == "robotic"
        assert voice_chars["species"] == "robot"
        assert "friendly" in voice_chars["personality"]

    def test_character_relationship_methods(self) -> None:
        """Test character relationship methods."""
        dimensions = CharacterDimensions(
            species=Species.ROBOT,
            archetype=Archetype.COMPANION,
            personality_traits=[PersonalityTrait.FRIENDLY],
            abilities=[Ability.TEACHING],
            topics=[Topic.FRIENDSHIP],
            backstory="A friendly robot",
            goals=["Help people"],
            fears=["Being unhelpful"],
            likes=["Teaching"],
            dislikes=["Rudeness"],
        )

        character = Character(name="TestBot", dimensions=dimensions)

        # Add multiple relationships
        character.add_relationship("Sidekick", "friend", 0.8, "A loyal companion")
        character.add_relationship("Mentor", "teacher", 0.9, "A wise guide")
        character.add_relationship("Rival", "competitor", 0.3, "A friendly rival")

        # Test getting relationships by type
        friends = character.get_relationships_by_type("friend")
        assert len(friends) == 1
        assert friends[0].character == "Sidekick"

        teachers = character.get_relationships_by_type("teacher")
        assert len(teachers) == 1
        assert teachers[0].character == "Mentor"

    def test_character_localization_methods(self) -> None:
        """Test character localization methods."""
        dimensions = CharacterDimensions(
            species=Species.ROBOT,
            archetype=Archetype.COMPANION,
            personality_traits=[PersonalityTrait.FRIENDLY],
            abilities=[Ability.TEACHING],
            topics=[Topic.FRIENDSHIP],
            backstory="A friendly robot",
            goals=["Help people"],
            fears=["Being unhelpful"],
            likes=["Teaching"],
            dislikes=["Rudeness"],
        )

        character = Character(name="TestBot", dimensions=dimensions)

        # Add multiple localizations
        character.add_localization(
            "es", "RobotTest", "Un robot amigable", "Un robot que ayuda"
        )
        character.add_localization(
            "fr", "RobotTest", "Un robot amical", "Un robot qui aide"
        )

        # Test getting specific localization
        spanish = character.get_localization("es")
        assert spanish is not None
        assert spanish.name == "RobotTest"
        assert spanish.language == "es"

        french = character.get_localization("fr")
        assert french is not None
        assert french.name == "RobotTest"
        assert french.language == "fr"

        # Test getting non-existent localization
        german = character.get_localization("de")
        assert german is None
