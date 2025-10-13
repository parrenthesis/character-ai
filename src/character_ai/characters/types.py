"""
Multi-dimensional character system for the Character AI.

Provides flexible, data-driven character creation with species, archetype, personality,
abilities, and topics dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CharacterType(Enum):
    """Basic character types for compatibility."""

    robot = "robot"
    dragon = "dragon"
    pony = "pony"
    cat = "cat"
    dog = "dog"
    bear = "bear"
    wolf = "wolf"
    fox = "fox"
    rabbit = "rabbit"
    owl = "owl"
    eagle = "eagle"
    human = "human"
    fairy = "fairy"
    elf = "elf"
    dwarf = "dwarf"
    unicorn = "unicorn"
    phoenix = "phoenix"
    griffin = "griffin"
    pegasus = "pegasus"
    centaur = "centaur"
    mermaid = "mermaid"
    satyr = "satyr"
    android = "android"
    cyborg = "cyborg"
    ai = "ai"


class Species(Enum):
    """Character species/creature types."""

    # Fantasy Creatures
    DRAGON = "dragon"
    UNICORN = "unicorn"
    PHOENIX = "phoenix"
    GRIFFIN = "griffin"
    FAIRY = "fairy"
    ELF = "elf"
    DWARF = "dwarf"

    # Animals
    CAT = "cat"
    DOG = "dog"
    BEAR = "bear"
    WOLF = "wolf"
    FOX = "fox"
    RABBIT = "rabbit"
    OWL = "owl"
    EAGLE = "eagle"

    # Mythical Animals
    PEGASUS = "pegasus"
    CENTAUR = "centaur"
    MERMAID = "mermaid"
    SATYR = "satyr"

    # Robots & Technology
    ROBOT = "robot"
    ANDROID = "android"
    CYBORG = "cyborg"
    AI = "ai"

    # Human-like
    HUMAN = "human"
    KNIGHT = "knight"
    WIZARD = "wizard"
    PRINCESS = "princess"
    PRINCE = "prince"
    GIANT = "giant"

    # Abstract/Conceptual
    SPIRIT = "spirit"
    GHOST = "ghost"
    ANGEL = "angel"
    DEMON = "demon"


class Archetype(Enum):
    """Character archetypes based on personality and role."""

    # Heroic Archetypes
    HERO = "hero"
    GUARDIAN = "guardian"
    MENTOR = "mentor"
    PROTECTOR = "protector"
    CHAMPION = "champion"

    # Wisdom & Knowledge
    SAGE = "sage"
    SCHOLAR = "scholar"
    TEACHER = "teacher"
    WISE_ELDER = "wise_elder"
    ORACLE = "oracle"

    # Magic & Mystery
    MAGE = "mage"
    SORCERER = "sorcerer"
    ENCHANTER = "enchanter"
    MYSTIC = "mystic"
    SHAMAN = "shaman"

    # Nature & Growth
    NATURE_SPIRIT = "nature_spirit"
    GARDENER = "gardener"
    HEALER = "healer"
    ANIMAL_FRIEND = "animal_friend"
    EARTH_KEEPER = "earth_keeper"

    # Adventure & Exploration
    EXPLORER = "explorer"
    ADVENTURER = "adventurer"
    TRAVELER = "traveler"
    SCOUT = "scout"
    RANGER = "ranger"

    # Creativity & Art
    ARTIST = "artist"
    MUSICIAN = "musician"
    STORYTELLER = "storyteller"
    DANCER = "dancer"
    POET = "poet"

    # Friendship & Social
    COMPANION = "companion"
    FRIEND = "friend"
    HELPER = "helper"
    CHEERLEADER = "cheerleader"
    MEDIATOR = "mediator"

    # Play & Fun
    JESTER = "jester"
    TRICKSTER = "trickster"
    CLOWN = "clown"
    ENTERTAINER = "entertainer"
    GAME_MASTER = "game_master"

    # Antagonistic
    ANTAGONIST = "antagonist"
    VILLAIN = "villain"
    RIVAL = "rival"

    # Observer & Learner
    INNOCENT_OBSERVER = "innocent_observer"
    SEEKER = "seeker"


class PersonalityTrait(Enum):
    """Core personality traits."""

    # Energy & Activity
    ENERGETIC = "energetic"
    CALM = "calm"
    ADVENTUROUS = "adventurous"
    CAUTIOUS = "cautious"
    PLAYFUL = "playful"
    SERIOUS = "serious"

    # Social & Emotional
    FRIENDLY = "friendly"
    HELPFUL = "helpful"
    SHY = "shy"
    OUTGOING = "outgoing"
    EMPATHETIC = "empathetic"
    COMPASSIONATE = "compassionate"
    INDEPENDENT = "independent"

    # Intelligence & Learning
    CURIOUS = "curious"
    WISE = "wise"
    CREATIVE = "creative"
    ARTISTIC = "artistic"
    ANALYTICAL = "analytical"
    INTUITIVE = "intuitive"
    LOGICAL = "logical"

    # Values & Ethics
    HONEST = "honest"
    LOYAL = "loyal"
    BRAVE = "brave"
    KIND = "kind"
    GENEROUS = "generous"
    FAIR = "fair"
    PROTECTIVE = "protective"
    PURE = "pure"
    GENTLE = "gentle"
    PEACEFUL = "peaceful"
    CHEERFUL = "cheerful"

    # Negative traits
    AGGRESSIVE = "aggressive"
    STUBBORN = "stubborn"
    IMPATIENT = "impatient"

    # Cognitive & Behavioral
    LITERAL_MINDED = "literal_minded"
    EARNEST = "earnest"
    SINCERE = "sincere"
    CHILDLIKE_WONDER = "childlike_wonder"
    PATIENT = "patient"
    NON_JUDGMENTAL = "non_judgmental"


class Ability(Enum):
    """Character abilities and powers."""

    # Physical Abilities
    FLYING = "flying"
    SWIMMING = "swimming"
    RUNNING = "running"
    CLIMBING = "climbing"
    STRENGTH = "strength"
    AGILITY = "agility"

    # Magical Abilities
    MAGIC = "magic"
    HEALING = "healing"
    TELEPATHY = "telepathy"
    TELEPORTATION = "teleportation"
    SHAPESHIFTING = "shapeshifting"
    ELEMENTAL_CONTROL = "elemental_control"

    # Knowledge & Skills
    TEACHING = "teaching"
    STORYTELLING = "storytelling"
    MUSIC = "music"
    ART = "art"
    COOKING = "cooking"
    GARDENING = "gardening"

    # Social Abilities
    LEADERSHIP = "leadership"
    MEDIATION = "mediation"
    ENCOURAGEMENT = "encouragement"
    FRIENDSHIP = "friendship"
    PROTECTION = "protection"
    GUIDANCE = "guidance"

    # Magical Abilities
    NATURE_CONTROL = "nature_control"

    # Cognitive & Technical
    PERFECT_MEMORY = "perfect_memory"
    RAPID_COMPUTATION = "rapid_computation"
    SYSTEM_INTERFACE = "system_interface"
    MULTITASKING = "multitasking"
    INSTANT_LEARNING = "instant_learning"


class Topic(Enum):
    """Conversation topics and interests."""

    # Learning & Education
    SCIENCE = "science"
    TECHNOLOGY = "technology"
    MATH = "math"
    HISTORY = "history"
    LITERATURE = "literature"
    LANGUAGES = "languages"
    NATURE = "nature"

    # Arts & Creativity
    MUSIC = "music"
    ART = "art"
    DANCE = "dance"
    THEATER = "theater"
    WRITING = "writing"
    CRAFTS = "crafts"

    # Activities & Hobbies
    SPORTS = "sports"
    GAMES = "games"
    PUZZLES = "puzzles"
    READING = "reading"
    COOKING = "cooking"
    GARDENING = "gardening"

    # Social & Emotional
    FRIENDSHIP = "friendship"
    FAMILY = "family"
    EMOTIONS = "emotions"
    DREAMS = "dreams"
    ADVENTURES = "adventures"
    STORIES = "stories"

    # Fantasy & Imagination
    MAGIC = "magic"
    FAIRY_TALES = "fairy_tales"
    MYTHOLOGY = "mythology"
    SPACE = "space"
    TIME_TRAVEL = "time_travel"
    SUPERPOWERS = "superpowers"
    POETRY = "poetry"

    # Philosophy & Abstract Concepts
    PHILOSOPHY = "philosophy"
    LOGIC = "logic"
    HUMANITY = "humanity"
    EXPLORATION = "exploration"


@dataclass
class CharacterDimensions:
    """Multi-dimensional character definition."""

    species: Species
    archetype: Archetype
    personality_traits: List[PersonalityTrait] = field(default_factory=list)
    abilities: List[Ability] = field(default_factory=list)
    topics: List[Topic] = field(default_factory=list)

    # Additional flexible attributes
    custom_traits: Dict[str, Any] = field(default_factory=dict)
    backstory: Optional[str] = None
    goals: List[str] = field(default_factory=list)
    fears: List[str] = field(default_factory=list)
    likes: List[str] = field(default_factory=list)
    dislikes: List[str] = field(default_factory=list)


@dataclass
class CharacterRelationship:
    """Character relationship data."""

    character: str
    relationship: str
    strength: float = 1.0
    description: Optional[str] = None


@dataclass
class CharacterLocalization:
    """Character localization data."""

    language: str
    name: str
    backstory: Optional[str] = None
    description: Optional[str] = None


@dataclass
class CharacterLicensing:
    """Character licensing and rights data."""

    owner: str
    rights: List[str] = field(default_factory=list)
    restrictions: List[str] = field(default_factory=list)
    expiration: Optional[str] = None
    territories: List[str] = field(default_factory=list)
    license_type: str = "proprietary"


@dataclass
class Character:
    """Character with multi-dimensional attributes."""

    name: str
    dimensions: CharacterDimensions
    voice_style: str = "neutral"
    language: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Enterprise features
    relationships: List[CharacterRelationship] = field(default_factory=list)
    localizations: List[CharacterLocalization] = field(default_factory=list)
    licensing: Optional[CharacterLicensing] = None

    def get_voice_characteristics(self) -> Dict[str, Any]:
        """Get voice characteristics for TTS."""
        return {
            "voice_style": self.voice_style,
            "species": self.dimensions.species.value,
            "personality": [
                trait.value for trait in self.dimensions.personality_traits
            ],
        }

    # Enterprise feature methods
    def add_relationship(
        self,
        character: str,
        relationship: str,
        strength: float = 1.0,
        description: Optional[str] = None,
    ) -> None:
        """Add a relationship to another character."""
        rel = CharacterRelationship(
            character=character,
            relationship=relationship,
            strength=strength,
            description=description,
        )
        self.relationships.append(rel)

    def get_relationships_by_type(
        self, relationship_type: str
    ) -> List[CharacterRelationship]:
        """Get all relationships of a specific type."""
        return [
            rel for rel in self.relationships if rel.relationship == relationship_type
        ]

    def add_localization(
        self,
        language: str,
        name: str,
        backstory: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Add a localization for a specific language."""
        loc = CharacterLocalization(
            language=language, name=name, backstory=backstory, description=description
        )
        self.localizations.append(loc)

    def get_localization(self, language: str) -> Optional[CharacterLocalization]:
        """Get localization for a specific language."""
        for loc in self.localizations:
            if loc.language == language:
                return loc
        return None

    def set_licensing(
        self,
        owner: str,
        rights: List[str],
        restrictions: Optional[List[str]] = None,
        expiration: Optional[str] = None,
        territories: Optional[List[str]] = None,
        license_type: str = "proprietary",
    ) -> None:
        """Set licensing information for the character."""
        self.licensing = CharacterLicensing(
            owner=owner,
            rights=rights,
            restrictions=restrictions or [],
            expiration=expiration,
            territories=territories or [],
            license_type=license_type,
        )

    def has_right(self, right: str) -> bool:
        """Check if character has a specific right."""
        if not self.licensing:
            return False
        return right in self.licensing.rights

    def is_restricted_by(self, restriction: str) -> bool:
        """Check if character is restricted by a specific rule."""
        if not self.licensing:
            return False
        return restriction in self.licensing.restrictions


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
        archetype=Archetype.NATURE_SPIRIT,
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
