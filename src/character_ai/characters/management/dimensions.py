"""
Character dimensions and relationship data structures.

Contains multi-dimensional character definitions, relationships, localization, and licensing.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .enums import Ability, Archetype, PersonalityTrait, Species, Topic


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
