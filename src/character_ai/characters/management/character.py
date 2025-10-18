"""
Main Character class and related functionality.

Contains the core Character class with multi-dimensional attributes and enterprise features.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .dimensions import (
    CharacterDimensions,
    CharacterLicensing,
    CharacterLocalization,
    CharacterRelationship,
)


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
