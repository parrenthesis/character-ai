"""
Character enums and enumeration types.

Contains all character-related enums for species, archetypes, personality traits, abilities, and topics.
"""

from enum import Enum


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
    WITCH = "witch"
    WIZARD = "wizard"
    NECROMANCER = "necromancer"

    # Adventure & Exploration
    EXPLORER = "explorer"
    ADVENTURER = "adventurer"
    TRAVELER = "traveler"
    PIONEER = "pioneer"
    DISCOVERER = "discoverer"

    # Social & Leadership
    LEADER = "leader"
    DIPLOMAT = "diplomat"
    MEDIATOR = "mediator"
    COUNSELOR = "counselor"
    FRIEND = "friend"

    # Creative & Artistic
    ARTIST = "artist"
    MUSICIAN = "musician"
    STORYTELLER = "storyteller"
    POET = "poet"
    DREAMER = "dreamer"

    # Mystical & Spiritual
    MYSTIC = "mystic"
    HEALER = "healer"
    SHAMAN = "shaman"
    PRIEST = "priest"
    MONK = "monk"

    # Technical & Scientific
    INVENTOR = "inventor"
    SCIENTIST = "scientist"
    ENGINEER = "engineer"
    ANALYST = "analyst"
    RESEARCHER = "researcher"

    # Playful & Fun
    TRICKSTER = "trickster"
    JESTER = "jester"
    CLOWN = "clown"
    ENTERTAINER = "entertainer"
    COMPANION = "companion"

    # Dark & Complex
    ANTI_HERO = "anti_hero"
    OUTCAST = "outcast"
    ROGUE = "rogue"
    REBEL = "rebel"
    LONER = "loner"

    # Observational & Analytical
    INNOCENT_OBSERVER = "innocent_observer"


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
