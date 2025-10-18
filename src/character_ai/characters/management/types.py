"""
Character management types and data structures.

Provides a unified interface for all character-related types and templates.
"""

# Import main character class
from .character import Character

# Import all dimension classes
from .dimensions import (
    CharacterDimensions,
    CharacterLicensing,
    CharacterLocalization,
    CharacterRelationship,
)

# Import all enums
from .enums import Ability, Archetype, CharacterType, PersonalityTrait, Species, Topic

# Import templates
from .templates import CHARACTER_TEMPLATES, CharacterTemplate

# Re-export everything for backward compatibility
__all__ = [
    # Enums
    "Ability",
    "Archetype",
    "CharacterType",
    "PersonalityTrait",
    "Species",
    "Topic",
    # Dimensions
    "CharacterDimensions",
    "CharacterLicensing",
    "CharacterLocalization",
    "CharacterRelationship",
    # Main classes
    "Character",
    "CharacterTemplate",
    # Templates
    "CHARACTER_TEMPLATES",
]
