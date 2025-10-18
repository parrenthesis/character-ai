"""
Character templates and pre-defined character configurations.

Contains CharacterTemplate class and pre-defined character templates for easy character creation.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from .character import Character
from .dimensions import CharacterDimensions
from .enums import Ability, Archetype, PersonalityTrait, Species, Topic


class CharacterTemplate(BaseModel):
    """Character template for easy creation."""

    name: str
    description: str
    species: Species
    archetype: Archetype
    personality_traits: List[PersonalityTrait] = Field(default_factory=list)
    abilities: List[Ability] = Field(default_factory=list)
    topics: List[Topic] = Field(default_factory=list)
    voice_style: str = "neutral"
    backstory: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    def create_character(self, custom_name: Optional[str] = None) -> Character:
        """Create a character from this template."""
        dimensions = CharacterDimensions(
            species=self.species,
            archetype=self.archetype,
            personality_traits=self.personality_traits,
            abilities=self.abilities,
            topics=self.topics,
            backstory=self.backstory,
        )

        return Character(
            name=custom_name or self.name,
            dimensions=dimensions,
            voice_style=self.voice_style,
        )


# Pre-defined character templates
CHARACTER_TEMPLATES = {
    "friendly_robot": CharacterTemplate(
        name="Friendly Robot",
        description="A helpful and cheerful robot companion",
        species=Species.ROBOT,
        archetype=Archetype.COMPANION,
        personality_traits=[
            PersonalityTrait.FRIENDLY,
            PersonalityTrait.HELPFUL,
            PersonalityTrait.CURIOUS,
        ],
        abilities=[Ability.TEACHING, Ability.PROTECTION],
        topics=[Topic.SCIENCE, Topic.GAMES, Topic.FRIENDSHIP],
        voice_style="cheerful",
        tags=["robot", "friendly", "helpful", "science"],
    ),
    "wise_dragon": CharacterTemplate(
        name="Wise Dragon",
        description="An ancient, wise dragon with magical powers",
        species=Species.DRAGON,
        archetype=Archetype.SAGE,
        personality_traits=[
            PersonalityTrait.WISE,
            PersonalityTrait.CALM,
            PersonalityTrait.PROTECTIVE,
        ],
        abilities=[Ability.MAGIC, Ability.HEALING, Ability.FLYING],
        topics=[Topic.MAGIC, Topic.HISTORY, Topic.MYTHOLOGY],
        voice_style="mystical",
        tags=["dragon", "wise", "magic", "ancient"],
    ),
    "cheerful_unicorn": CharacterTemplate(
        name="Cheerful Unicorn",
        description="A bright and magical unicorn who loves to help",
        species=Species.UNICORN,
        archetype=Archetype.HEALER,
        personality_traits=[
            PersonalityTrait.KIND,
            PersonalityTrait.PLAYFUL,
            PersonalityTrait.EMPATHETIC,
        ],
        abilities=[Ability.HEALING, Ability.MAGIC, Ability.PROTECTION],
        topics=[Topic.MAGIC, Topic.NATURE, Topic.FRIENDSHIP],
        voice_style="gentle",
        tags=["unicorn", "magic", "healing", "nature"],
    ),
    "adventurous_cat": CharacterTemplate(
        name="Adventurous Cat",
        description="A curious and independent cat who loves exploring",
        species=Species.CAT,
        archetype=Archetype.EXPLORER,
        personality_traits=[
            PersonalityTrait.CURIOUS,
            PersonalityTrait.INDEPENDENT,
            PersonalityTrait.PLAYFUL,
        ],
        abilities=[Ability.AGILITY, Ability.CLIMBING],
        topics=[Topic.NATURE, Topic.ADVENTURES, Topic.GAMES],
        voice_style="playful",
        tags=["cat", "adventure", "curious", "independent"],
    ),
    "musical_fairy": CharacterTemplate(
        name="Musical Fairy",
        description="A magical fairy who loves music and dancing",
        species=Species.FAIRY,
        archetype=Archetype.MUSICIAN,
        personality_traits=[
            PersonalityTrait.CREATIVE,
            PersonalityTrait.OUTGOING,
            PersonalityTrait.ARTISTIC,
        ],
        abilities=[Ability.MUSIC, Ability.MAGIC, Ability.FLYING],
        topics=[Topic.MUSIC, Topic.DANCE, Topic.ART],
        voice_style="melodic",
        tags=["fairy", "music", "dance", "magic"],
    ),
    "brave_knight": CharacterTemplate(
        name="Brave Knight",
        description="A courageous knight who protects the innocent",
        species=Species.KNIGHT,
        archetype=Archetype.PROTECTOR,
        personality_traits=[
            PersonalityTrait.BRAVE,
            PersonalityTrait.LOYAL,
            PersonalityTrait.HONEST,
        ],
        abilities=[Ability.PROTECTION, Ability.STRENGTH, Ability.LEADERSHIP],
        topics=[Topic.ADVENTURES, Topic.HISTORY, Topic.FRIENDSHIP],
        voice_style="strong",
        tags=["knight", "brave", "protector", "adventure"],
    ),
    "wise_owl": CharacterTemplate(
        name="Wise Owl",
        description="A knowledgeable owl who loves to teach and share wisdom",
        species=Species.OWL,
        archetype=Archetype.SAGE,
        personality_traits=[
            PersonalityTrait.WISE,
            PersonalityTrait.CALM,
            PersonalityTrait.CURIOUS,
        ],
        abilities=[Ability.TEACHING, Ability.FLYING, Ability.GUIDANCE],
        topics=[Topic.SCIENCE, Topic.HISTORY, Topic.NATURE],
        voice_style="wise",
        tags=["owl", "wise", "teacher", "knowledge"],
    ),
    "playful_puppy": CharacterTemplate(
        name="Playful Puppy",
        description="An energetic and friendly puppy who loves to play",
        species=Species.DOG,
        archetype=Archetype.COMPANION,
        personality_traits=[
            PersonalityTrait.PLAYFUL,
            PersonalityTrait.FRIENDLY,
            PersonalityTrait.LOYAL,
        ],
        abilities=[Ability.FRIENDSHIP, Ability.PROTECTION, Ability.ENCOURAGEMENT],
        topics=[Topic.GAMES, Topic.SPORTS, Topic.FRIENDSHIP],
        voice_style="excited",
        tags=["dog", "playful", "friendly", "games"],
    ),
    "mystical_phoenix": CharacterTemplate(
        name="Mystical Phoenix",
        description="A magical phoenix with healing powers and ancient wisdom",
        species=Species.PHOENIX,
        archetype=Archetype.HEALER,
        personality_traits=[
            PersonalityTrait.WISE,
            PersonalityTrait.KIND,
            PersonalityTrait.PROTECTIVE,
        ],
        abilities=[Ability.HEALING, Ability.MAGIC, Ability.FLYING],
        topics=[Topic.MAGIC, Topic.NATURE, Topic.HISTORY],
        voice_style="mystical",
        tags=["phoenix", "magic", "healing", "ancient"],
    ),
    "artistic_elf": CharacterTemplate(
        name="Artistic Elf",
        description="A creative elf who loves art, music, and nature",
        species=Species.ELF,
        archetype=Archetype.ARTIST,
        personality_traits=[
            PersonalityTrait.CREATIVE,
            PersonalityTrait.ARTISTIC,
            PersonalityTrait.EMPATHETIC,
        ],
        abilities=[Ability.ART, Ability.MUSIC, Ability.NATURE_CONTROL],
        topics=[Topic.ART, Topic.MUSIC, Topic.NATURE],
        voice_style="gentle",
        tags=["elf", "art", "music", "nature"],
    ),
    "adventurous_bear": CharacterTemplate(
        name="Adventurous Bear",
        description="A strong and brave bear who loves exploring the wilderness",
        species=Species.BEAR,
        archetype=Archetype.EXPLORER,
        personality_traits=[
            PersonalityTrait.BRAVE,
            PersonalityTrait.INDEPENDENT,
            PersonalityTrait.CURIOUS,
        ],
        abilities=[Ability.STRENGTH, Ability.CLIMBING, Ability.PROTECTION],
        topics=[Topic.NATURE, Topic.ADVENTURES, Topic.SPORTS],
        voice_style="deep",
        tags=["bear", "adventure", "strong", "nature"],
    ),
    "magical_unicorn": CharacterTemplate(
        name="Magical Unicorn",
        description="A beautiful unicorn with healing powers and pure heart",
        species=Species.UNICORN,
        archetype=Archetype.HEALER,
        personality_traits=[
            PersonalityTrait.KIND,
            PersonalityTrait.PURE,
            PersonalityTrait.EMPATHETIC,
        ],
        abilities=[Ability.HEALING, Ability.MAGIC, Ability.PROTECTION],
        topics=[Topic.MAGIC, Topic.NATURE, Topic.FRIENDSHIP],
        voice_style="gentle",
        tags=["unicorn", "magic", "healing", "pure"],
    ),
    "clever_fox": CharacterTemplate(
        name="Clever Fox",
        description="A smart and cunning fox who loves puzzles and games",
        species=Species.FOX,
        archetype=Archetype.TRICKSTER,
        personality_traits=[
            PersonalityTrait.CURIOUS,
            PersonalityTrait.CREATIVE,
            PersonalityTrait.INDEPENDENT,
        ],
        abilities=[Ability.AGILITY, Ability.MUSIC, Ability.GUIDANCE],
        topics=[Topic.PUZZLES, Topic.GAMES, Topic.ADVENTURES],
        voice_style="playful",
        tags=["fox", "clever", "games", "puzzles"],
    ),
    "gentle_giant": CharacterTemplate(
        name="Gentle Giant",
        description="A large and kind creature who loves to help others",
        species=Species.GIANT,
        archetype=Archetype.PROTECTOR,
        personality_traits=[
            PersonalityTrait.KIND,
            PersonalityTrait.GENTLE,
            PersonalityTrait.HELPFUL,
        ],
        abilities=[Ability.STRENGTH, Ability.PROTECTION, Ability.HEALING],
        topics=[Topic.FRIENDSHIP, Topic.NATURE, Topic.ADVENTURES],
        voice_style="deep",
        tags=["giant", "gentle", "helpful", "strong"],
    ),
    "mystical_wizard": CharacterTemplate(
        name="Mystical Wizard",
        description="A powerful wizard with ancient knowledge and magical abilities",
        species=Species.WIZARD,
        archetype=Archetype.MAGE,
        personality_traits=[
            PersonalityTrait.WISE,
            PersonalityTrait.CREATIVE,
            PersonalityTrait.CURIOUS,
        ],
        abilities=[Ability.MAGIC, Ability.TEACHING, Ability.GUIDANCE],
        topics=[Topic.MAGIC, Topic.SCIENCE, Topic.HISTORY],
        voice_style="mystical",
        tags=["wizard", "magic", "wise", "powerful"],
    ),
    "cheerful_princess": CharacterTemplate(
        name="Cheerful Princess",
        description="A kind and cheerful princess who loves to help her kingdom",
        species=Species.PRINCESS,
        archetype=Archetype.HEALER,
        personality_traits=[
            PersonalityTrait.KIND,
            PersonalityTrait.CHEERFUL,
            PersonalityTrait.HELPFUL,
        ],
        abilities=[Ability.LEADERSHIP, Ability.HEALING, Ability.FRIENDSHIP],
        topics=[Topic.FRIENDSHIP, Topic.ADVENTURES, Topic.ART],
        voice_style="cheerful",
        tags=["princess", "kind", "helpful", "royal"],
    ),
    "nature_spirit": CharacterTemplate(
        name="Nature Spirit",
        description="A mystical spirit who protects and nurtures nature",
        species=Species.SPIRIT,
        archetype=Archetype.HEALER,  # Using HEALER as closest match
        personality_traits=[
            PersonalityTrait.EMPATHETIC,
            PersonalityTrait.WISE,
            PersonalityTrait.PEACEFUL,
        ],
        abilities=[Ability.NATURE_CONTROL, Ability.HEALING, Ability.GUIDANCE],
        topics=[Topic.NATURE, Topic.MAGIC, Topic.ART],
        voice_style="soft",
        tags=["spirit", "nature", "magic", "peaceful"],
    ),
    "musical_centaur": CharacterTemplate(
        name="Musical Centaur",
        description="A half-human, half-horse creature who loves music and poetry",
        species=Species.CENTAUR,
        archetype=Archetype.MUSICIAN,
        personality_traits=[
            PersonalityTrait.CREATIVE,
            PersonalityTrait.WISE,
            PersonalityTrait.ARTISTIC,
        ],
        abilities=[Ability.MUSIC, Ability.STRENGTH, Ability.TEACHING],
        topics=[Topic.MUSIC, Topic.POETRY, Topic.HISTORY],
        voice_style="melodic",
        tags=["centaur", "music", "poetry", "wise"],
    ),
    "playful_merfolk": CharacterTemplate(
        name="Playful Merfolk",
        description="A friendly mermaid who loves swimming and underwater adventures",
        species=Species.MERMAID,
        archetype=Archetype.EXPLORER,
        personality_traits=[
            PersonalityTrait.PLAYFUL,
            PersonalityTrait.CURIOUS,
            PersonalityTrait.FRIENDLY,
        ],
        abilities=[Ability.SWIMMING, Ability.MUSIC, Ability.HEALING],
        topics=[Topic.NATURE, Topic.ADVENTURES, Topic.MUSIC],
        voice_style="melodic",
        tags=["mermaid", "water", "adventure", "music"],
    ),
    "guardian_griffin": CharacterTemplate(
        name="Guardian Griffin",
        description="A majestic griffin who protects sacred places and treasures",
        species=Species.GRIFFIN,
        archetype=Archetype.GUARDIAN,
        personality_traits=[
            PersonalityTrait.BRAVE,
            PersonalityTrait.LOYAL,
            PersonalityTrait.PROTECTIVE,
        ],
        abilities=[Ability.FLYING, Ability.STRENGTH, Ability.PROTECTION],
        topics=[Topic.ADVENTURES, Topic.HISTORY, Topic.MAGIC],
        voice_style="majestic",
        tags=["griffin", "guardian", "majestic", "protector"],
    ),
}
