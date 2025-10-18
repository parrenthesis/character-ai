"""
AI-powered character generation system.

Uses LLMs to generate characters from natural language descriptions.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..core.config.main import Config

from ..core.llm.config import LLMType
from ..core.llm.factory import LLMFactory
from ..core.llm.manager import OpenModelService
from .management.types import (
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
from .management.validation import CharacterValidator, ValidationLevel, ValidationResult

logger = logging.getLogger(__name__)


class AICharacterGenerator:
    """AI-powered character generation using LLMs."""

    def __init__(self, config: Optional["Config"] = None) -> None:
        """Initialize the AI character generator."""
        if config is None:
            from ..core.config.main import Config

            config = Config()

        self.config_manager = config.create_llm_config_manager()
        self.model_manager = OpenModelService()
        self.factory = LLMFactory(self.config_manager, self.model_manager)
        self.validator = CharacterValidator()
        self.llm: Optional[Any] = None
        self._generation_count = 0
        self._max_generations_before_refresh = 5  # Refresh LLM after 5 generations
        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize the character creation LLM."""
        try:
            # Try to get the character creation LLM from the factory
            self.llm = self.factory.get_character_creation_llm()
            logger.info("Character creation LLM initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize character creation LLM: {e}")
            logger.info("Falling back to mock character generation")
            self.llm = None

    def _refresh_llm_if_needed(self) -> None:
        """Refresh LLM if it has been used too many times (to prevent state issues)."""
        self._generation_count += 1
        if self._generation_count >= self._max_generations_before_refresh:
            logger.info("Refreshing LLM to prevent state issues")
            try:
                self.factory.refresh_llm(LLMType.CHARACTER_CREATION)
                self.llm = self.factory.get_character_creation_llm()
                self._generation_count = 0
                logger.info("LLM refreshed successfully")
            except Exception as e:
                logger.error(f"Failed to refresh LLM: {e}")
                self.llm = None

    async def generate_from_description(
        self, description: str, custom_name: Optional[str] = None
    ) -> Optional[Character]:
        """Generate a character from a natural language description."""
        # Refresh LLM if needed to prevent state issues
        self._refresh_llm_if_needed()

        if not self.llm:
            # Fall back to mock character generation
            logger.info("Using mock character generation (LLM not available)")
            return self._generate_mock_character(description, custom_name)

        try:
            # Create generation prompt
            prompt = self._create_generation_prompt(description)

            # Generate character data using real LLM with retry logic
            response = await self._generate_character_data(prompt)

            # If LLM generation failed, fall back to mock
            if not response:
                logger.warning("LLM generation failed, falling back to mock generation")

                return self._generate_mock_character(description, custom_name)

            # Parse and create character
            character = self._parse_character_response(response, custom_name)

            if character:
                # Validate the generated character
                validation_results = self.validator.validate_character(character)
                validation_summary = self.validator.get_validation_summary(
                    validation_results
                )

                if validation_summary["is_valid"]:
                    logger.info(
                        f"Successfully generated and validated character: "
                        f"{character.name}"
                    )
                    self._log_validation_results(validation_results)
                    return character
                elif validation_summary["can_use"]:
                    logger.warning(f"Character generated with issues: {character.name}")

                    self._log_validation_results(validation_results)
                    return character
                else:
                    logger.error(f"Character failed validation: {character.name}")
                    self._log_validation_results(validation_results)
                    # Try to generate a new character or fall back to mock
                    logger.info("Attempting to generate alternative character...")
                    return await self._generate_alternative_character(
                        description, custom_name
                    )
            else:
                # If LLM parsing fails, fall back to mock generation
                logger.warning(
                    "LLM character parsing failed, falling back to mock generation"
                )
                return self._generate_mock_character(description, custom_name)

        except Exception as e:
            logger.error(f"Error generating character with LLM: {e}")
            logger.info("Falling back to mock character generation")
            return self._generate_mock_character(description, custom_name)

    def _create_generation_prompt(self, description: str) -> str:
        """Create a structured prompt for character generation using best practices."""
        return f"""You are a character creation assistant. Create a character based on this
 description: "{description}"

IMPORTANT: Respond with ONLY valid JSON. No explanations, no additional text.

Format:
{{
    "name": "Character name",
    "species": "robot",
    "archetype": "companion",
    "personality_traits": ["friendly"],
    "abilities": ["music"],
    "topics": ["music"],
    "voice_style": "cheerful",
    "backstory": "Brief backstory"
}}

Example:
{{
    "name": "SparkleBot",
    "species": "robot",
    "archetype": "companion",
    "personality_traits": ["friendly", "helpful"],
    "abilities": ["music", "teaching"],
    "topics": ["music", "learning"],
    "voice_style": "cheerful",
    "backstory": "A friendly robot who loves to help children learn music"
}}

Response:"""

    async def _generate_character_data(self, prompt: str, max_retries: int = 3) -> str:
        """Generate character data using the LLM with retry logic and validation."""
        for attempt in range(max_retries):
            try:
                logger.info(f"LLM generation attempt {attempt + 1}/{max_retries}")

                # Use the character creation LLM with optimized parameters
                if not self.llm:
                    raise RuntimeError("LLM not initialized")
                response = await self.llm.generate(
                    prompt=prompt,
                    max_tokens=500,  # Reduced for more focused responses
                    temperature=0.7,  # Slightly lower for more consistent output
                )

                # Validate response quality
                if self._validate_response(response):
                    logger.info(f"Valid response received on attempt {attempt + 1}")
                    return str(response)
                else:
                    logger.warning(
                        f"Invalid response on attempt {attempt + 1}: {repr(response)}"
                    )
                    if attempt < max_retries - 1:
                        # Wait before retry
                        import asyncio

                        await asyncio.sleep(1)
                        continue

            except Exception as e:
                logger.error(
                    f"Error generating character data on attempt {attempt + 1}: {e}"
                )
                if attempt < max_retries - 1:
                    import asyncio

                    await asyncio.sleep(1)
                    continue
                else:
                    raise

        # If all retries failed, return empty string to trigger fallback
        logger.error("All LLM generation attempts failed")
        return ""

    def _validate_response(self, response: str) -> bool:
        """Validate LLM response quality."""
        if not response or len(response.strip()) == 0:
            return False

        # Check if response contains JSON-like structure
        if "{" not in response or "}" not in response:
            return False

        # Check if response is too short (likely incomplete)
        if len(response.strip()) < 50:
            return False

        # Check if response contains required fields
        required_fields = ["name", "species", "archetype"]
        response_lower = response.lower()
        for field in required_fields:
            if f'"{field}"' not in response_lower:
                return False

        return True

    def _log_validation_results(
        self, validation_results: List[ValidationResult]
    ) -> None:
        """Log validation results for debugging and monitoring."""
        for result in validation_results:
            if result.level == ValidationLevel.CRITICAL:
                logger.error(f"CRITICAL: {result.message}")
            elif result.level == ValidationLevel.ERROR:
                logger.error(f"ERROR: {result.message}")
            elif result.level == ValidationLevel.WARNING:
                logger.warning(f"WARNING: {result.message}")
            elif result.level == ValidationLevel.INFO:
                logger.info(f"INFO: {result.message}")

    async def _generate_alternative_character(
        self, description: str, custom_name: Optional[str] = None
    ) -> Optional[Character]:
        """Generate an alternative character when validation fails."""
        try:
            # Try to generate with a safer, more constrained prompt
            safe_prompt = self._create_safe_generation_prompt(description)
            response = await self._generate_character_data(safe_prompt)

            if response:
                character = self._parse_character_response(response, custom_name)
                if character:
                    # Validate the alternative character
                    validation_results = self.validator.validate_character(character)
                    validation_summary = self.validator.get_validation_summary(
                        validation_results
                    )

                    if validation_summary["can_use"]:
                        logger.info(
                            f"Alternative character generated: {character.name}"
                        )
                        return character

            # If alternative generation fails, fall back to mock
            logger.warning(
                "Alternative character generation failed, using mock character"
            )
            return self._generate_mock_character(description, custom_name)

        except Exception as e:
            logger.error(f"Error generating alternative character: {e}")
            return self._generate_mock_character(description, custom_name)

    def _create_safe_generation_prompt(self, description: str) -> str:
        """Create a safer prompt for character generation with more constraints."""
        return f"""Create a child-friendly character: "{description}"

IMPORTANT: Make the character safe, positive, and appropriate for children.

Format:
{{
    "name": "Character name",
    "species": "robot",
    "archetype": "companion",
    "personality_traits": ["friendly", "helpful"],
    "abilities": ["music", "teaching"],
    "topics": ["music", "learning"],
    "voice_style": "cheerful",
    "backstory": "Positive, child-friendly backstory"
}}

Example:
{{
    "name": "HappyBot",
    "species": "robot",
    "archetype": "companion",
    "personality_traits": ["friendly", "helpful", "cheerful"],
    "abilities": ["music", "teaching", "healing"],
    "topics": ["music", "learning", "friendship"],
    "voice_style": "cheerful",
    "backstory": "A friendly robot who loves to help children learn and play music"
}}

Response:"""

    def _parse_character_response(
        self, response: str, custom_name: Optional[str] = None
    ) -> Optional[Character]:
        """Parse LLM response and create character."""
        try:
            # Ensure response is a string
            if not isinstance(response, str):
                response = str(response)  # type: ignore

            # Debug: Log the raw response
            logger.debug(f"Raw LLM response: {response}")

            # Extract JSON from response - try multiple patterns
            json_match = None

            # Try to find JSON in various formats
            patterns = [
                r'\{[^{}]*"name"[^{}]*\}',  # Simple JSON with name field
                r"\{.*?\}",  # Any JSON object
                r"\{[^{}]*\}",  # JSON without nested objects
                r"```json\s*(\{.*?\})\s*```",  # JSON in code blocks
                r"```\s*(\{.*?\})\s*```",  # JSON in generic code blocks
                r"===\s*(\{.*?\})",  # JSON after === separator
                r"===\s*\n\s*(\{.*?\})",  # JSON after === separator with newline
            ]

            for pattern in patterns:
                json_match = re.search(pattern, response, re.DOTALL)
                if json_match:
                    break

            if not json_match:
                logger.error(f"No JSON found in LLM response. Response was: {response}")

                return None

            # Get the JSON text (handle capture groups)
            json_text = (
                json_match.group(1) if json_match.groups() else json_match.group()
            )

            # Try to parse the JSON
            try:
                character_data = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}. Trying to clean JSON...")
                # Try to clean the JSON by removing extra characters
                json_text = (
                    json_match.group(1) if json_match.groups() else json_match.group()
                )
                # Remove common artifacts
                json_text = re.sub(
                    r"<\|.*?\|>", "", json_text
                )  # Remove <|assistant|> tags
                json_text = re.sub(r"==+.*", "", json_text)  # Remove === separators
                json_text = re.sub(r"<hr>.*", "", json_text)  # Remove <hr> tags
                json_text = re.sub(
                    r"```json\s*", "", json_text
                )  # Remove ```json markers
                json_text = re.sub(r"```\s*", "", json_text)  # Remove ``` markers
                json_text = json_text.strip()

                try:
                    character_data = json.loads(json_text)
                except json.JSONDecodeError as e2:
                    logger.error(f"Still can't parse JSON after cleaning: {e2}")
                    return None

            # Parse species
            species_str = character_data.get("species", "robot")
            try:
                species = Species(species_str)
            except ValueError:
                species = Species.ROBOT  # Default fallback

            # Parse archetype
            archetype_str = character_data.get("archetype", "companion")
            try:
                archetype = Archetype(archetype_str)
            except ValueError:
                archetype = Archetype.COMPANION  # Default fallback

            # Parse personality traits
            personality_traits = []
            for trait_str in character_data.get("personality_traits", []):
                try:
                    trait = PersonalityTrait(trait_str)
                    personality_traits.append(trait)
                except ValueError:
                    continue  # Skip invalid traits

            # Parse abilities
            abilities = []
            for ability_str in character_data.get("abilities", []):
                try:
                    ability = Ability(ability_str)
                    abilities.append(ability)
                except ValueError:
                    continue  # Skip invalid abilities

            # Parse topics
            topics = []
            for topic_str in character_data.get("topics", []):
                try:
                    topic = Topic(topic_str)
                    topics.append(topic)
                except ValueError:
                    continue  # Skip invalid topics

            # Create character dimensions
            dimensions = CharacterDimensions(
                species=species,
                archetype=archetype,
                personality_traits=personality_traits,
                abilities=abilities,
                topics=topics,
                backstory=character_data.get("backstory"),
                goals=character_data.get("goals", []),
                fears=character_data.get("fears", []),
                likes=character_data.get("likes", []),
                dislikes=character_data.get("dislikes", []),
            )

            # Create enhanced character
            character = Character(
                name=custom_name or character_data.get("name", "Generated Character"),
                dimensions=dimensions,
                voice_style=character_data.get("voice_style", "neutral"),
            )

            return character

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating character: {e}")
            return None

    def suggest_character_variations(
        self, base_character: Character, count: int = 3
    ) -> List[Character]:
        """Suggest variations of a base character."""
        variations = []

        # Create variations by modifying different aspects
        for i in range(count):
            variation = self._create_character_variation(base_character, i)
            if variation:
                variations.append(variation)

        return variations

    def _create_character_variation(
        self, base_character: Character, variation_index: int
    ) -> Optional[Character]:
        """Create a variation of a base character."""
        try:
            # Create a copy of the base character
            variation = Character(
                name=f"{base_character.name} Variant {variation_index + 1}",
                dimensions=CharacterDimensions(
                    species=base_character.dimensions.species,
                    archetype=base_character.dimensions.archetype,
                    personality_traits=base_character.dimensions.personality_traits.copy(),
                    abilities=base_character.dimensions.abilities.copy(),
                    topics=base_character.dimensions.topics.copy(),
                    backstory=base_character.dimensions.backstory,
                    goals=base_character.dimensions.goals.copy(),
                    fears=base_character.dimensions.fears.copy(),
                    likes=base_character.dimensions.likes.copy(),
                    dislikes=base_character.dimensions.dislikes.copy(),
                ),
                voice_style=base_character.voice_style,
            )

            # Modify based on variation index
            if variation_index == 0:
                # Add a new personality trait
                if (
                    PersonalityTrait.CREATIVE
                    not in variation.dimensions.personality_traits
                ):
                    variation.dimensions.personality_traits.append(
                        PersonalityTrait.CREATIVE
                    )
            elif variation_index == 1:
                # Add a new ability
                if Ability.MUSIC not in variation.dimensions.abilities:
                    variation.dimensions.abilities.append(Ability.MUSIC)
            elif variation_index == 2:
                # Add a new topic
                if Topic.ART not in variation.dimensions.topics:
                    variation.dimensions.topics.append(Topic.ART)

            return variation

        except Exception as e:
            logger.error(f"Error creating character variation: {e}")
            return None

    def get_character_suggestions(
        self, user_preferences: Dict[str, Any]
    ) -> List[CharacterTemplate]:
        """Get character suggestions based on user preferences."""
        suggestions = []

        # Filter templates based on preferences
        for template in CHARACTER_TEMPLATES.values():
            if self._matches_preferences(template, user_preferences):
                suggestions.append(template)

        # Sort by relevance
        suggestions.sort(
            key=lambda t: self._calculate_relevance_score(t, user_preferences),
            reverse=True,
        )

        return suggestions[:5]  # Return top 5 suggestions

    def _matches_preferences(
        self, template: CharacterTemplate, preferences: Dict[str, Any]
    ) -> bool:
        """Check if template matches user preferences."""
        # Check species preference
        if "species" in preferences:
            if template.species.value != preferences["species"]:
                return False

        # Check archetype preference
        if "archetype" in preferences:
            if template.archetype.value != preferences["archetype"]:
                return False

        # Check personality preference
        if "personality" in preferences:
            preferred_traits = set(preferences["personality"])
            template_traits = set(trait.value for trait in template.personality_traits)
            if not preferred_traits.intersection(template_traits):
                return False

        return True

    def _calculate_relevance_score(
        self, template: CharacterTemplate, preferences: Dict[str, Any]
    ) -> float:
        """Calculate relevance score for template matching."""
        score = 0.0

        # Species match
        if "species" in preferences:
            if template.species.value == preferences["species"]:
                score += 1.0

        # Archetype match
        if "archetype" in preferences:
            if template.archetype.value == preferences["archetype"]:
                score += 1.0

        # Personality match
        if "personality" in preferences:
            preferred_traits = set(preferences["personality"])
            template_traits = set(trait.value for trait in template.personality_traits)
            overlap = len(preferred_traits.intersection(template_traits))
            score += overlap * 0.5

        # Topic match
        if "topics" in preferences:
            preferred_topics = set(preferences["topics"])
            template_topics = set(topic.value for topic in template.topics)
            overlap = len(preferred_topics.intersection(template_topics))
            score += overlap * 0.3

        return score

    def _generate_mock_character(
        self, description: str, custom_name: Optional[str] = None
    ) -> Optional[Character]:
        """Generate a mock character based on description keywords."""
        try:
            description_lower = description.lower()

            # Determine species based on keywords
            species = Species.ROBOT  # Default
            if any(
                word in description_lower
                for word in ["dragon", "fire", "magic", "ancient"]
            ):
                species = Species.DRAGON
            elif any(
                word in description_lower
                for word in ["unicorn", "magic", "healing", "rainbow"]
            ):
                species = Species.UNICORN
            elif any(
                word in description_lower
                for word in ["cat", "feline", "curious", "independent"]
            ):
                species = Species.CAT
            elif any(
                word in description_lower
                for word in ["fairy", "magic", "music", "dance"]
            ):
                species = Species.FAIRY
            elif any(
                word in description_lower
                for word in ["dog", "loyal", "friendly", "playful"]
            ):
                species = Species.DOG

            # Determine archetype
            archetype = Archetype.COMPANION  # Default
            if any(
                word in description_lower
                for word in ["wise", "sage", "ancient", "knowledge"]
            ):
                archetype = Archetype.SAGE
            elif any(
                word in description_lower
                for word in ["healer", "healing", "helpful", "kind"]
            ):
                archetype = Archetype.HEALER
            elif any(
                word in description_lower
                for word in ["explorer", "adventure", "curious", "travel"]
            ):
                archetype = Archetype.EXPLORER
            elif any(
                word in description_lower
                for word in ["musician", "music", "dance", "art"]
            ):
                archetype = Archetype.MUSICIAN

            # Determine personality traits
            personality_traits = [PersonalityTrait.FRIENDLY]  # Default
            if any(
                word in description_lower for word in ["wise", "smart", "intelligent"]
            ):
                personality_traits.append(PersonalityTrait.WISE)
            if any(
                word in description_lower
                for word in ["playful", "fun", "happy", "cheerful"]
            ):
                personality_traits.append(PersonalityTrait.PLAYFUL)
            if any(
                word in description_lower
                for word in ["curious", "explorer", "adventure"]
            ):
                personality_traits.append(PersonalityTrait.CURIOUS)
            if any(word in description_lower for word in ["kind", "helpful", "caring"]):
                personality_traits.append(PersonalityTrait.KIND)

            # Determine abilities
            abilities = [Ability.PROTECTION]  # Default
            if any(word in description_lower for word in ["magic", "magical", "spell"]):
                abilities.append(Ability.MAGIC)
            if any(word in description_lower for word in ["healing", "healer", "help"]):
                abilities.append(Ability.HEALING)
            if any(word in description_lower for word in ["flying", "fly", "wing"]):
                abilities.append(Ability.FLYING)
            if any(word in description_lower for word in ["music", "song", "dance"]):
                abilities.append(Ability.MUSIC)

            # Determine topics
            topics = [Topic.FRIENDSHIP]  # Default
            if any(word in description_lower for word in ["magic", "magical", "spell"]):
                topics.append(Topic.MAGIC)
            if any(
                word in description_lower for word in ["nature", "garden", "forest"]
            ):
                topics.append(Topic.NATURE)
            if any(word in description_lower for word in ["music", "song", "dance"]):
                topics.append(Topic.MUSIC)
            if any(
                word in description_lower for word in ["adventure", "explore", "travel"]
            ):
                topics.append(Topic.ADVENTURES)

            # Create character dimensions
            dimensions = CharacterDimensions(
                species=species,
                archetype=archetype,
                personality_traits=personality_traits,
                abilities=abilities,
                topics=topics,
                backstory=f"A character created from the description: '{description}'",
            )

            # Create character
            character = Character(
                name=custom_name or f"Generated {species.value.title()}",
                dimensions=dimensions,
                voice_style="friendly",
            )

            return character

        except Exception as e:
            logger.error(f"Error generating mock character: {e}")
            return None
