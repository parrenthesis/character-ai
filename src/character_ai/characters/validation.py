"""
Character validation system for safety and quality checks.

Ensures generated characters meet platform standards and are appropriate for the
target audience.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .types import Ability, Archetype, Character, PersonalityTrait, Species

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of character validation."""

    is_valid: bool
    level: ValidationLevel
    message: str
    field: Optional[str] = None
    suggestions: Optional[List[str]] = None


class CharacterValidator:
    """Validates characters for safety, quality, and appropriateness."""

    def __init__(self) -> None:
        """Initialize the character validator."""
        self.safety_keywords = self._load_safety_keywords()
        self.quality_standards = self._load_quality_standards()

    def _load_safety_keywords(self) -> Dict[str, List[str]]:
        """Load safety keywords for content filtering."""
        return {
            "inappropriate": [
                "violence",
                "weapon",
                "fight",
                "battle",
                "war",
                "kill",
                "hurt",
                "harm",
                "scary",
                "frightening",
                "terrifying",
                "monster",
                "demon",
                "evil",
                "inappropriate",
                "adult",
                "mature",
                "explicit",
                "sexual",
            ],
            "negative_emotions": [
                "angry",
                "mad",
                "furious",
                "hate",
                "despise",
                "revenge",
                "bitter",
                "sad",
                "depressed",
                "lonely",
                "isolated",
                "abandoned",
            ],
            "dangerous_activities": [
                "dangerous",
                "risky",
                "unsafe",
                "harmful",
                "toxic",
                "poison",
                "explosive",
                "flammable",
                "sharp",
                "cut",
                "stab",
            ],
        }

    def _load_quality_standards(self) -> Dict[str, Any]:
        """Load quality standards for character validation."""
        return {
            "min_backstory_length": 20,
            "max_backstory_length": 500,
            "required_fields": [
                "name",
                "species",
                "archetype",
                "personality_traits",
                "backstory",
            ],
            "min_personality_traits": 1,
            "max_personality_traits": 5,
            "min_abilities": 1,
            "max_abilities": 4,
            "min_topics": 1,
            "max_topics": 4,
        }

    def validate_character(self, character: Character) -> List[ValidationResult]:
        """Validate a character for safety, quality, and appropriateness."""
        results = []

        # Safety validation
        safety_results = self._validate_safety(character)
        results.extend(safety_results)

        # Quality validation
        quality_results = self._validate_quality(character)
        results.extend(quality_results)

        # Consistency validation
        consistency_results = self._validate_consistency(character)
        results.extend(consistency_results)

        return results

    def _validate_safety(self, character: Character) -> List[ValidationResult]:
        """Validate character for safety and appropriateness."""
        results = []

        # Check name for inappropriate content
        name_result = self._check_safety_keywords(character.name, "name")
        if name_result:
            results.append(name_result)

        # Check backstory for inappropriate content
        if character.dimensions.backstory:
            backstory_result = self._check_safety_keywords(
                character.dimensions.backstory, "backstory"
            )
            if backstory_result:
                results.append(backstory_result)

        # Check personality traits for negative traits
        personality_result = self._check_negative_personality_traits(
            character.dimensions.personality_traits
        )
        if personality_result:
            results.append(personality_result)

        # Check goals for inappropriate content
        for goal in character.dimensions.goals:
            goal_result = self._check_safety_keywords(goal, "goals")
            if goal_result:
                results.append(goal_result)

        return results

    def _validate_quality(self, character: Character) -> List[ValidationResult]:
        """Validate character quality and completeness."""
        results = []

        # Check required fields
        required_fields = self.quality_standards["required_fields"]
        for field in required_fields:
            if not self._has_required_field(character, field):
                results.append(
                    ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message=f"Missing required field: {field}",
                        field=field,
                        suggestions=[f"Add {field} to complete the character"],
                    )
                )

        # Check backstory length
        if character.dimensions.backstory:
            backstory_length = len(character.dimensions.backstory)
            min_length = self.quality_standards["min_backstory_length"]
            max_length = self.quality_standards["max_backstory_length"]

            if backstory_length < min_length:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.WARNING,
                        message=f"Backstory too short ({backstory_length} chars, "
                        f"minimum {min_length})",
                        field="backstory",
                        suggestions=["Add more detail to the character's backstory"],
                    )
                )
            elif backstory_length > max_length:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.WARNING,
                        message=f"Backstory too long ({backstory_length} chars, "
                        f"maximum {max_length})",
                        field="backstory",
                        suggestions=["Shorten the backstory to be more concise"],
                    )
                )

        # Check personality traits count
        traits_count = len(character.dimensions.personality_traits)
        min_traits = self.quality_standards["min_personality_traits"]
        max_traits = self.quality_standards["max_personality_traits"]

        if traits_count < min_traits:
            results.append(
                ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message=f"Too few personality traits ({traits_count}, "
                    f"minimum {min_traits})",
                    field="personality_traits",
                    suggestions=[
                        "Add more personality traits to make the character more "
                        "interesting"
                    ],
                )
            )
        elif traits_count > max_traits:
            results.append(
                ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message=f"Too many personality traits ({traits_count}, "
                    f"maximum {max_traits})",
                    field="personality_traits",
                    suggestions=["Reduce the number of personality traits for "
                                "clarity"],
                )
            )

        # Check abilities count
        abilities_count = len(character.dimensions.abilities)
        min_abilities = self.quality_standards["min_abilities"]
        max_abilities = self.quality_standards["max_abilities"]

        if abilities_count < min_abilities:
            results.append(
                ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message=f"Too few abilities ({abilities_count}, "
                    f"minimum {min_abilities})",
                    field="abilities",
                    suggestions=[
                        "Add more abilities to make the character more capable"
                    ],
                )
            )
        elif abilities_count > max_abilities:
            results.append(
                ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message=f"Too many abilities ({abilities_count}, "
                    f"maximum {max_abilities})",
                    field="abilities",
                    suggestions=["Reduce the number of abilities for focus"],
                )
            )

        return results

    def _validate_consistency(self, character: Character) -> List[ValidationResult]:
        """Validate character consistency and coherence."""
        results = []

        # Check species-archetype consistency
        species_archetype_result = self._check_species_archetype_consistency(
            character.dimensions.species, character.dimensions.archetype
        )
        if species_archetype_result:
            results.append(species_archetype_result)

        # Check personality-abilities consistency
        personality_abilities_result = self._check_personality_abilities_consistency(
            character.dimensions.personality_traits, character.dimensions.abilities
        )
        if personality_abilities_result:
            results.append(personality_abilities_result)

        return results

    def _check_safety_keywords(
        self, text: str, field: str
    ) -> Optional[ValidationResult]:
        """Check text for safety keywords."""
        text_lower = text.lower()

        for category, keywords in self.safety_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.CRITICAL,
                        message=f"Inappropriate content detected in {field}: "
                        f"'{keyword}'",
                        field=field,
                        suggestions=[
                            f"Remove or replace content containing '{keyword}'"
                        ],
                    )

        return None

    def _check_negative_personality_traits(
        self, traits: List[PersonalityTrait]
    ) -> Optional[ValidationResult]:
        """Check for negative personality traits."""
        # Define negative traits that exist in the enum

        # Check if any of the traits are negative (we'll define this based on the
        # actual enum values)
        negative_found = []
        for trait in traits:
            # Check for traits that might be considered negative
            if trait.value.lower() in [
                "aggressive",
                "mean",
                "cruel",
                "angry",
                "hostile",
                "violent",
            ]:
                negative_found.append(trait)

        if negative_found:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Negative personality traits detected: "
                f"{[t.value for t in negative_found]}",
                field="personality_traits",
                suggestions=[
                    "Replace negative traits with positive ones for "
                    "child-friendly characters"
                ],
            )

        return None

    def _has_required_field(self, character: Character, field: str) -> bool:
        """Check if character has a required field."""
        if field == "name":
            return bool(character.name and character.name.strip())
        elif field == "species":
            return character.dimensions.species is not None
        elif field == "archetype":
            return character.dimensions.archetype is not None
        elif field == "personality_traits":
            return len(character.dimensions.personality_traits) > 0
        elif field == "backstory":
            return bool(
                character.dimensions.backstory
                and character.dimensions.backstory.strip()
            )
        return False

    def _check_species_archetype_consistency(
        self, species: Species, archetype: Archetype
    ) -> Optional[ValidationResult]:
        """Check if species and archetype are consistent."""
        # Define consistent combinations based on actual enum values

        # Only check if we have valid combinations for the species
        species_value = species.value.lower()
        if species_value == "robot":
            consistent_archetypes = ["companion", "mentor", "musician", "scholar"]
        elif species_value == "dragon":
            consistent_archetypes = ["mentor", "sage", "protector"]
        elif species_value == "fairy":
            consistent_archetypes = ["companion", "helper", "musician"]
        elif species_value == "human":
            consistent_archetypes = ["hero", "mentor", "scholar", "protector"]
        elif species_value == "alien":
            consistent_archetypes = ["sage", "scholar", "mentor"]
        else:
            # For unknown species, don't validate
            return None

        archetype_value = archetype.value.lower()
        if archetype_value not in consistent_archetypes:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Species {species.value} and archetype {archetype.value} "
                f"may not be consistent",
                field="archetype",
                suggestions=[
                    f"Consider changing archetype to: "
                    f"{', '.join(consistent_archetypes)}"
                ],
            )

        return None

    def _check_personality_abilities_consistency(
        self, traits: List[PersonalityTrait], abilities: List[Ability]
    ) -> Optional[ValidationResult]:
        """Check if personality traits and abilities are consistent."""
        # Define consistent combinations based on actual enum values

        # Only check if we have valid combinations
        for trait in traits:
            trait_value = trait.value.lower()
            if trait_value == "friendly":
                consistent_abilities = ["teaching", "music", "healing"]
            elif trait_value == "wise":
                consistent_abilities = ["teaching", "magic", "scholarship"]
            elif trait_value == "brave":
                consistent_abilities = ["protection", "leadership", "combat"]
            elif trait_value == "creative":
                consistent_abilities = ["music", "art", "magic"]
            elif trait_value == "helpful":
                consistent_abilities = ["healing", "teaching", "protection"]
            else:
                continue

            # Check if any abilities match the consistent ones
            ability_values = [ability.value.lower() for ability in abilities]
            if not any(ability in consistent_abilities for ability in ability_values):
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.INFO,
                    message=f"Personality trait {trait.value} may not align with "
                    f"abilities",
                    field="abilities",
                    suggestions=[
                        f"Consider adding abilities like: "
                        f"{', '.join(consistent_abilities)}"
                    ],
                )

        return None

    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:

        """Get a summary of validation results."""
        total_results = len(results)
        critical_errors = len(
            [r for r in results if r.level == ValidationLevel.CRITICAL]
        )
        errors = len([r for r in results if r.level == ValidationLevel.ERROR])
        warnings = len([r for r in results if r.level == ValidationLevel.WARNING])
        info = len([r for r in results if r.level == ValidationLevel.INFO])

        is_valid = critical_errors == 0 and errors == 0

        return {
            "is_valid": is_valid,
            "total_issues": total_results,
            "critical_errors": critical_errors,
            "errors": errors,
            "warnings": warnings,
            "info": info,
            "can_use": critical_errors == 0,  # Can use if no critical errors
            "needs_improvement": errors > 0 or warnings > 0,
        }
