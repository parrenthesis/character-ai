"""
Catalog storage system for enterprise character management.

Provides hierarchical franchise-based organization with advanced search and analytics.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .manager import CharacterManager
from .types import Archetype, Character, Species

logger = logging.getLogger(__name__)


class CatalogStorage:
    """Enterprise catalog storage with franchise-based organization."""

    def __init__(self, catalog_dir: Path = Path.cwd() / "catalog"):
        self.catalog_dir = catalog_dir
        self.characters_dir = catalog_dir / "characters"
        self.voices_dir = catalog_dir / "voices"
        self.metadata_dir = catalog_dir / "metadata"
        self.exports_dir = catalog_dir / "exports"
        self.index_file = catalog_dir / "index.json"

        # Don't create directory structure during instantiation - create when actually needed

        # Initialize character manager
        self.character_manager = CharacterManager()

        # Load or create index
        self.index = self._load_or_create_index()

    def _load_or_create_index(self) -> Dict[str, Any]:
        """Load existing index or create new one."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    return dict(json.load(f))
            except Exception as e:
                logger.warning(f"Error loading index: {e}, creating new index")

        # Create new index
        return {
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_characters": 0,
            "franchises": {},
            "voice_stats": {"total_with_voice": 0, "voice_availability": 0.0},
            "usage_stats": {"most_used": [], "recent_adaptations": 0},
        }

    async def store_character(
        self, character: Character, franchise: str = "original"
    ) -> str:
        """Store character with franchise organization."""
        try:
            # Create directories if they don't exist
            if not self.characters_dir.exists():
                self.characters_dir.mkdir(parents=True, exist_ok=True)

            # Create franchise directory
            franchise_dir = self.characters_dir / f"franchise={franchise}"
            franchise_dir.mkdir(exist_ok=True)

            # Save character file
            char_file = (
                franchise_dir / f"{character.name.lower().replace(' ', '_')}.yaml"
            )
            char_data = self._character_to_dict(character, franchise)

            logger.debug(f"Storing character data: {char_data}")

            with open(char_file, "w") as f:
                yaml.dump(char_data, f, default_flow_style=False)

            # Update index
            await self._update_index(character.name, franchise, str(char_file))

            logger.info(
                f"Stored character '{character.name}' in franchise '{franchise}'"
            )
            return str(char_file)

        except Exception as e:
            logger.error(f"Error storing character: {e}")
            raise

    def _character_to_dict(
        self, character: Character, franchise: str
    ) -> Dict[str, Any]:
        """Convert character to dictionary with catalog metadata."""
        return {
            "name": character.name,
            "franchise": franchise,
            "catalog_id": f"{franchise}_{character.name.lower().replace(' ', '_')}",
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "voice_style": character.voice_style,
            "language": character.language,
            "metadata": {
                **character.metadata,
                "source": "catalog_storage",
                "tags": self._extract_tags(character),
                "usage_count": 0,
            },
            "dimensions": {
                "species": character.dimensions.species.value,
                "archetype": character.dimensions.archetype.value,
                "personality_traits": [
                    trait.value for trait in character.dimensions.personality_traits
                ],
                "abilities": [
                    ability.value for ability in character.dimensions.abilities
                ],
                "topics": [topic.value for topic in character.dimensions.topics],
                "backstory": character.dimensions.backstory,
                "goals": character.dimensions.goals,
                "fears": character.dimensions.fears,
                "likes": character.dimensions.likes,
                "dislikes": character.dimensions.dislikes,
            },
            "voice": {
                "available": False,
                "file_path": None,
                "quality_score": None,
                "cloned_at": None,
            },
            "analytics": {
                "total_usage": 0,
                "last_used": None,
                "adaptations": 0,
                "contexts": [],
            },
            # Enterprise features
            "relationships": [
                {
                    "character": rel.character,
                    "relationship": rel.relationship,
                    "strength": rel.strength,
                    "description": rel.description,
                }
                for rel in character.relationships
            ],
            "localizations": [
                {
                    "language": loc.language,
                    "name": loc.name,
                    "backstory": loc.backstory,
                    "description": loc.description,
                }
                for loc in character.localizations
            ],
            "licensing": (
                {
                    "owner": character.licensing.owner,
                    "rights": character.licensing.rights,
                    "restrictions": character.licensing.restrictions,
                    "expiration": character.licensing.expiration,
                    "territories": character.licensing.territories,
                    "license_type": character.licensing.license_type,
                }
                if character.licensing
                else None
            ),
        }

    def _extract_tags(self, character: Character) -> List[str]:
        """Extract tags from character for search indexing."""
        tags = []

        # Add species and archetype as tags
        tags.append(character.dimensions.species.value)
        tags.append(character.dimensions.archetype.value)

        # Add personality traits as tags
        for trait in character.dimensions.personality_traits:
            tags.append(trait.value)

        # Add abilities as tags
        for ability in character.dimensions.abilities:
            tags.append(ability.value)

        return list(set(tags))  # Remove duplicates

    async def _update_index(
        self, character_name: str, franchise: str, file_path: str
    ) -> None:
        """Update catalog index with new character."""
        try:
            # Update franchise info
            if franchise not in self.index["franchises"]:
                self.index["franchises"][franchise] = {"count": 0, "characters": []}

            if character_name not in self.index["franchises"][franchise]["characters"]:
                self.index["franchises"][franchise]["characters"].append(character_name)

                self.index["franchises"][franchise]["count"] += 1
                self.index["total_characters"] += 1

            # Update timestamp
            self.index["last_updated"] = datetime.now(timezone.utc).isoformat()

            # Save index
            with open(self.index_file, "w") as f:
                json.dump(self.index, f, indent=2)

        except Exception as e:
            logger.error(f"Error updating index: {e}")

    async def load_character(
        self, character_name: str, franchise: str = "original"
    ) -> Optional[Character]:
        """Load character from catalog storage."""
        try:
            franchise_dir = self.characters_dir / f"franchise={franchise}"
            char_file = (
                franchise_dir / f"{character_name.lower().replace(' ', '_')}.yaml"
            )

            if not char_file.exists():
                logger.warning(f"Character file not found: {char_file}")
                return None

            with open(char_file, "r") as f:
                raw_content = f.read()
                data = yaml.safe_load(raw_content)

            if data is None:
                logger.error(f"YAML file is empty or invalid: {char_file}")
                return None

            if "dimensions" not in data:
                logger.error(f"Character data missing 'dimensions' field: {data}")
                return None

            # Convert back to Character
            return self._dict_to_character(data)

        except Exception as e:
            logger.error(f"Error loading character: {e}")
            return None

    def _dict_to_character(self, data: Dict[str, Any]) -> Character:
        """Convert dictionary back to Character."""
        from .types import Ability, Archetype, PersonalityTrait, Species, Topic

        if data is None:
            raise ValueError("Data is None in _dict_to_character")

        # Parse dimensions
        dimensions_data = data.get("dimensions", {})
        if not dimensions_data:
            raise ValueError(f"Missing or empty dimensions data: {data}")

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

        from .types import CharacterDimensions

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

        # Parse enterprise features
        relationships = []
        for rel_data in data.get("relationships", []):
            if rel_data is None:
                continue
            from .types import CharacterRelationship

            relationships.append(
                CharacterRelationship(
                    character=rel_data["character"],
                    relationship=rel_data["relationship"],
                    strength=rel_data.get("strength", 1.0),
                    description=rel_data.get("description"),
                )
            )

        localizations = []
        for loc_data in data.get("localizations", []):
            if loc_data is None:
                continue
            from .types import CharacterLocalization

            localizations.append(
                CharacterLocalization(
                    language=loc_data["language"],
                    name=loc_data["name"],
                    backstory=loc_data.get("backstory"),
                    description=loc_data.get("description"),
                )
            )

        licensing = None
        if "licensing" in data and data["licensing"] is not None:
            from .types import CharacterLicensing

            lic_data = data["licensing"]
            licensing = CharacterLicensing(
                owner=lic_data["owner"],
                rights=lic_data.get("rights", []),
                restrictions=lic_data.get("restrictions", []),
                expiration=lic_data.get("expiration"),
                territories=lic_data.get("territories", []),
                license_type=lic_data.get("license_type", "proprietary"),
            )

        return Character(
            name=data["name"],
            dimensions=dimensions,
            voice_style=data.get("voice_style", "neutral"),
            language=data.get("language", "en"),
            metadata=data.get("metadata", {}),
            relationships=relationships,
            localizations=localizations,
            licensing=licensing,
        )

    async def search_characters(self, query: Dict[str, Any]) -> List[Character]:
        """Advanced search characters by criteria with text search and fuzzy
        matching."""
        results = []

        try:
            # Load search index
            search_index = await self._load_search_index()

            # Handle text search
            if "text" in query:
                text_results = await self._text_search(query["text"], search_index)
                for character_data in text_results:
                    if self._matches_search_criteria(character_data, query):
                        character = await self.load_character(
                            character_data["name"], character_data["franchise"]
                        )
                        if character:
                            results.append(character)
            else:
                # Standard criteria search
                for character_data in search_index.get("characters", {}).values():
                    if self._matches_search_criteria(character_data, query):
                        character = await self.load_character(
                            character_data["name"], character_data["franchise"]
                        )
                        if character:
                            results.append(character)

            # Sort results by relevance if text search
            if "text" in query:
                results = await self._sort_by_relevance(results, query["text"])

            return results

        except Exception as e:
            logger.error(f"Error searching characters: {e}")
            return []

    def _matches_search_criteria(
        self, character_data: Dict[str, Any], criteria: Dict[str, Any]
    ) -> bool:
        """Check if character matches search criteria."""
        # Franchise filter
        if "franchise" in criteria:
            if character_data.get("franchise") != criteria["franchise"]:
                return False

        # Species filter
        if "species" in criteria:
            if character_data.get("species") != criteria["species"]:
                return False

        # Archetype filter
        if "archetype" in criteria:
            if character_data.get("archetype") != criteria["archetype"]:
                return False

        # Personality traits filter
        if "personality_traits" in criteria:
            required_traits = set(criteria["personality_traits"])
            character_traits = set(character_data.get("personality_traits", []))
            if not required_traits.issubset(character_traits):
                return False

        # Abilities filter
        if "abilities" in criteria:
            required_abilities = set(criteria["abilities"])
            character_abilities = set(character_data.get("abilities", []))
            if not required_abilities.issubset(character_abilities):
                return False

        # Topics filter
        if "topics" in criteria:
            required_topics = set(criteria["topics"])
            character_topics = set(character_data.get("topics", []))
            if not required_topics.issubset(character_topics):
                return False

        # Voice availability filter
        if "voice_available" in criteria:
            if character_data.get("voice_available") != criteria["voice_available"]:
                return False

        # Name filter
        if "name" in criteria:
            if criteria["name"].lower() not in character_data.get("name", "").lower():
                return False

        return True

    async def _load_search_index(self) -> Dict[str, Any]:
        """Load or create search index."""
        # Create metadata directory if it doesn't exist
        if not self.metadata_dir.exists():
            self.metadata_dir.mkdir(parents=True, exist_ok=True)

        search_index_file = self.metadata_dir / "search_index.json"

        if search_index_file.exists():
            try:
                with open(search_index_file, "r") as f:
                    return dict(json.load(f))
            except Exception as e:
                logger.warning(f"Error loading search index: {e}")

        # Create search index
        return await self._build_search_index()

    async def _build_search_index(self) -> Dict[str, Any]:
        """Build search index from all characters."""
        search_index: Dict[str, Any] = {
            "characters": {},
            "franchises": {},
            "search_terms": {},
        }

        try:
            # Create directories if they don't exist
            if not self.characters_dir.exists():
                self.characters_dir.mkdir(parents=True, exist_ok=True)
            if not self.metadata_dir.exists():
                self.metadata_dir.mkdir(parents=True, exist_ok=True)

            # Scan all franchise directories
            for franchise_dir in self.characters_dir.glob("franchise=*"):
                franchise_name = franchise_dir.name.replace("franchise=", "")

                for char_file in franchise_dir.glob("*.yaml"):
                    with open(char_file, "r") as f:
                        data = yaml.safe_load(f)

                    character_name = data["name"]
                    search_index["characters"][character_name] = {
                        "name": character_name,
                        "franchise": franchise_name,
                        "species": data["dimensions"]["species"],
                        "archetype": data["dimensions"]["archetype"],
                        "personality_traits": data["dimensions"]["personality_traits"],
                        "abilities": data["dimensions"]["abilities"],
                        "topics": data["dimensions"]["topics"],
                        "backstory": data["dimensions"].get("backstory", ""),
                        "goals": data["dimensions"].get("goals", []),
                        "fears": data["dimensions"].get("fears", []),
                        "likes": data["dimensions"].get("likes", []),
                        "dislikes": data["dimensions"].get("dislikes", []),
                        "voice_available": data.get("voice", {}).get(
                            "available", False
                        ),
                        "usage_count": data.get("analytics", {}).get("total_usage", 0),
                        "tags": data.get("metadata", {}).get("tags", []),
                        "file_path": str(char_file),
                    }

                # Update franchise info
                search_index["franchises"][franchise_name] = {
                    "count": len(list(franchise_dir.glob("*.yaml"))),
                    "characters": [f.stem for f in franchise_dir.glob("*.yaml")],
                }

            # Build search terms index
            for character_name, char_data in search_index["characters"].items():
                for term in char_data.get("tags", []):
                    if term not in search_index["search_terms"]:
                        search_index["search_terms"][term] = []
                    search_index["search_terms"][term].append(character_name)

            # Save search index
            search_index_file = self.metadata_dir / "search_index.json"
            with open(search_index_file, "w") as f:
                json.dump(search_index, f, indent=2)

            return search_index

        except Exception as e:
            logger.error(f"Error building search index: {e}")
            return search_index

    async def get_catalog_statistics(self) -> Dict[str, Any]:
        """Get comprehensive catalog statistics."""
        try:
            search_index = await self._load_search_index()

            stats: Dict[str, Any] = {
                "total_characters": len(search_index.get("characters", {})),
                "franchises": {},
                "species_distribution": {},
                "archetype_distribution": {},
                "personality_traits_distribution": {},
                "abilities_distribution": {},
                "topics_distribution": {},
                "voice_stats": {"total_with_voice": 0, "voice_availability": 0.0},
                "usage_stats": {"most_used": [], "total_usage": 0},
            }

            # Analyze characters
            for char_data in search_index.get("characters", {}).values():
                franchise = char_data.get("franchise", "unknown")
                if franchise not in stats["franchises"]:
                    stats["franchises"][franchise] = 0
                stats["franchises"][franchise] += 1

                # Species distribution
                species = char_data.get("species", "unknown")
                stats["species_distribution"][species] = (
                    stats["species_distribution"].get(species, 0) + 1
                )

                # Archetype distribution
                archetype = char_data.get("archetype", "unknown")
                stats["archetype_distribution"][archetype] = (
                    stats["archetype_distribution"].get(archetype, 0) + 1
                )

                # Personality traits distribution
                for trait in char_data.get("personality_traits", []):
                    stats["personality_traits_distribution"][trait] = (
                        stats["personality_traits_distribution"].get(trait, 0) + 1
                    )

                # Abilities distribution
                for ability in char_data.get("abilities", []):
                    stats["abilities_distribution"][ability] = (
                        stats["abilities_distribution"].get(ability, 0) + 1
                    )

                # Topics distribution
                for topic in char_data.get("topics", []):
                    stats["topics_distribution"][topic] = (
                        stats["topics_distribution"].get(topic, 0) + 1
                    )

                # Voice stats
                if char_data.get("voice_available", False):
                    stats["voice_stats"]["total_with_voice"] += 1

                # Usage stats
                usage_count = char_data.get("usage_count", 0)
                stats["usage_stats"]["total_usage"] += usage_count
                if usage_count > 0:
                    stats["usage_stats"]["most_used"].append(
                        {"name": char_data["name"], "usage_count": usage_count}
                    )

            # Calculate voice availability percentage
            total_chars = stats["total_characters"]
            if total_chars > 0:
                stats["voice_stats"]["voice_availability"] = (
                    stats["voice_stats"]["total_with_voice"] / total_chars
                )

            # Sort most used characters
            stats["usage_stats"]["most_used"].sort(
                key=lambda x: x["usage_count"], reverse=True
            )
            stats["usage_stats"]["most_used"] = stats["usage_stats"]["most_used"][
                :10
            ]  # Top 10

            return stats

        except Exception as e:
            logger.error(f"Error getting catalog statistics: {e}")
            return {}

    async def export_catalog(
        self, franchise: Optional[str] = None, output_file: Optional[Path] = None
    ) -> Path:
        """Export catalog to YAML file."""
        try:
            if output_file is None:
                # Create exports directory if it doesn't exist
                if not self.exports_dir.exists():
                    self.exports_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if franchise:
                    output_file = (
                        self.exports_dir / f"catalog_{franchise}_{timestamp}.yaml"
                    )
                else:
                    output_file = self.exports_dir / f"catalog_all_{timestamp}.yaml"

            # Collect characters
            characters = []
            search_index = await self._load_search_index()

            for char_data in search_index.get("characters", {}).values():
                if franchise is None or char_data.get("franchise") == franchise:
                    char_file = Path(char_data["file_path"])
                    if char_file.exists():
                        with open(char_file, "r") as f:
                            char_yaml = yaml.safe_load(f)
                        characters.append(char_yaml)

            # Create export data
            export_data = {
                "catalog_export": {
                    "version": "1.0",
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "franchise": franchise or "all",
                    "total_characters": len(characters),
                },
                "characters": characters,
            }

            # Write export file
            with open(output_file, "w") as f:
                yaml.dump(export_data, f, default_flow_style=False)

            logger.info(f"Exported catalog to {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error exporting catalog: {e}")
            raise

    async def import_catalog(
        self, catalog_file: Path, voice_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Import catalog from YAML file with optional voice file support."""
        try:
            with open(catalog_file, "r") as f:
                data = yaml.safe_load(f)

            if "catalog_export" not in data:
                raise ValueError(
                    "Invalid catalog format: missing 'catalog_export' section"
                )

            imported_count = 0
            voice_processed_count = 0
            errors = []

            # Initialize voice manager if voice directory provided
            voice_manager = None
            if voice_dir and voice_dir.exists():
                from .catalog_voice_manager import CatalogVoiceManager

                voice_manager = CatalogVoiceManager(self)
                logger.info(f"Voice directory provided: {voice_dir}")

            for char_data in data.get("characters", []):
                try:
                    # Convert to Character
                    character = self._dict_to_character(char_data)
                    franchise = char_data.get("franchise", "imported")

                    # Store character
                    await self.store_character(character, franchise)
                    imported_count += 1

                    # Process voice file if available
                    if voice_manager and voice_dir:
                        voice_processed = await self._process_character_voice(
                            character, franchise, voice_dir, voice_manager
                        )
                        if voice_processed:
                            voice_processed_count += 1

                except Exception as e:
                    errors.append(
                        f"Error importing {char_data.get('name', 'unknown')}: {e}"
                    )

            result = {
                "imported_count": imported_count,
                "voice_processed_count": voice_processed_count,
                "total_characters": len(data.get("characters", [])),
                "errors": errors,
            }

            logger.info(f"Imported {imported_count} characters from {catalog_file}")
            if voice_processed_count > 0:
                logger.info(f"Processed {voice_processed_count} voice files")
            return result

        except Exception as e:
            logger.error(f"Error importing catalog: {e}")
            raise

    async def _process_character_voice(
        self, character: Character, franchise: str, voice_dir: Path, voice_manager: Any
    ) -> bool:
        """Process voice file for a character during catalog import."""
        try:
            # Try to find matching voice file
            voice_file = self._find_voice_file(character.name, voice_dir)

            if voice_file:
                # Clone voice for character
                success = await voice_manager.clone_character_voice(
                    character_name=character.name,
                    voice_file_path=voice_file,
                    franchise=franchise,
                )
                if success:
                    logger.info(f"✓ Voice cloned for {character.name}")
                    return True
                else:
                    logger.warning(f"✗ Failed to clone voice for {character.name}")
                    return False
            else:
                # Set default voice profile
                await self._set_default_voice_profile(
                    character, franchise, voice_manager
                )
                logger.info(f"✓ Default voice profile set for {character.name}")
                return True

        except Exception as e:
            logger.warning(f"Error processing voice for {character.name}: {e}")
            return False

    def _find_voice_file(self, character_name: str, voice_dir: Path) -> Optional[Path]:
        """Find matching voice file for character."""
        # Try different naming patterns
        patterns = [
            f"{character_name.lower().replace(' ', '_')}_voice.wav",
            f"{character_name.lower().replace(' ', '_')}_voice.mp3",
            f"{character_name.lower().replace(' ', '_')}.wav",
            f"{character_name.lower().replace(' ', '_')}.mp3",
            f"{character_name}_voice.wav",
            f"{character_name}_voice.mp3",
            f"{character_name}.wav",
            f"{character_name}.mp3",
        ]

        for pattern in patterns:
            voice_file = voice_dir / pattern
            if voice_file.exists():
                return voice_file

        return None

    async def _set_default_voice_profile(
        self, character: Character, franchise: str, voice_manager: Any
    ) -> None:
        """Set default voice profile based on character type."""
        # Determine default voice characteristics based on character
        voice_style = self._get_default_voice_style(character)

        # Create a default voice profile entry
        voice_info = {
            "character_name": character.name,
            "franchise": franchise,
            "voice_style": voice_style,
            "quality_score": 0.5,  # Default quality
            "is_default": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Store default voice metadata
        await voice_manager._save_voice_metadata_entry(
            character.name, franchise, voice_info
        )

    def _get_default_voice_style(self, character: Character) -> str:
        """Get default voice style based on character attributes."""
        # Map character attributes to voice styles
        if character.dimensions.species in [Species.ROBOT, Species.ANDROID]:
            return "robotic"
        elif character.dimensions.species in [Species.UNICORN, Species.DRAGON]:
            return "mystical"
        elif character.dimensions.species in [Species.CAT, Species.DOG]:
            return "playful"
        elif character.dimensions.archetype == Archetype.SCHOLAR:
            return "wise"
        elif character.dimensions.archetype == Archetype.HERO:
            return "brave"
        elif character.dimensions.archetype == Archetype.MENTOR:
            return "calm"
        else:
            return "friendly"

    async def _text_search(
        self, search_text: str, search_index: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Perform text search across character data."""
        results = []
        search_text_lower = search_text.lower()

        for character_data in search_index.get("characters", {}).values():
            # Search in name
            if search_text_lower in character_data.get("name", "").lower():
                results.append(character_data)
                continue

            # Search in backstory
            if search_text_lower in character_data.get("backstory", "").lower():
                results.append(character_data)
                continue

            # Search in goals
            for goal in character_data.get("goals", []):
                if search_text_lower in goal.lower():
                    results.append(character_data)
                    break

            # Search in topics
            for topic in character_data.get("topics", []):
                if search_text_lower in topic.lower():
                    results.append(character_data)
                    break

            # Search in personality traits
            for trait in character_data.get("personality_traits", []):
                if search_text_lower in trait.lower():
                    results.append(character_data)
                    break

        return results

    async def _sort_by_relevance(
        self, characters: List[Character], search_text: str
    ) -> List[Character]:
        """Sort characters by relevance to search text."""

        def relevance_score(character: Character) -> int:
            score = 0
            search_text_lower = search_text.lower()

            # Name match (highest priority)
            if search_text_lower in character.name.lower():
                score += 100

            # Backstory match
            if (
                character.dimensions.backstory
                and search_text_lower in character.dimensions.backstory.lower()
            ):
                score += 50

            # Goals match
            for goal in character.dimensions.goals:
                if search_text_lower in goal.lower():
                    score += 30

            # Topics match
            for topic in character.dimensions.topics:
                if search_text_lower in topic.value.lower():
                    score += 20

            # Personality traits match
            for trait in character.dimensions.personality_traits:
                if search_text_lower in trait.value.lower():
                    score += 10

            return score

        return sorted(characters, key=relevance_score, reverse=True)

    async def create_franchise(
        self,
        franchise_name: str,
        description: str = "",
        owner: str = "",
        permissions: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create a new franchise with proper isolation."""
        try:
            # Validate franchise name
            if (
                not franchise_name
                or not franchise_name.replace("_", "").replace("-", "").isalnum()
            ):
                raise ValueError(
                    "Franchise name must be alphanumeric (with _ and - allowed)"
                )

            # Check if franchise already exists
            if await self.franchise_exists(franchise_name):
                raise ValueError(f"Franchise '{franchise_name}' already exists")

            # Create directories if they don't exist
            if not self.characters_dir.exists():
                self.characters_dir.mkdir(parents=True, exist_ok=True)

            # Create franchise directory
            franchise_dir = self.characters_dir / f"franchise={franchise_name}"
            franchise_dir.mkdir(exist_ok=True)

            # Create franchise metadata
            franchise_metadata = {
                "name": franchise_name,
                "description": description,
                "owner": owner,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "permissions": permissions
                or {"read": True, "write": True, "admin": False},
                "character_count": 0,
                "voice_availability": 0.0,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }

            # Create metadata directory if it doesn't exist
            if not self.metadata_dir.exists():
                self.metadata_dir.mkdir(parents=True, exist_ok=True)

            # Save franchise metadata
            franchise_meta_file = self.metadata_dir / f"franchise_{franchise_name}.json"

            with open(franchise_meta_file, "w") as f:
                json.dump(franchise_metadata, f, indent=2)

            # Update main index
            await self._update_franchise_index(franchise_name, franchise_metadata)

            logger.info(f"Created franchise '{franchise_name}' with owner '{owner}'")
            return True

        except Exception as e:
            logger.error(f"Error creating franchise '{franchise_name}': {e}")
            raise

    async def franchise_exists(self, franchise_name: str) -> bool:
        """Check if franchise exists."""
        franchise_dir = self.characters_dir / f"franchise={franchise_name}"
        return franchise_dir.exists()

    async def get_franchise_info(self, franchise_name: str) -> Optional[Dict[str, Any]]:
        """Get franchise information and statistics."""
        try:
            if not await self.franchise_exists(franchise_name):
                return None

            # Load franchise metadata
            franchise_meta_file = self.metadata_dir / f"franchise_{franchise_name}.json"

            if franchise_meta_file.exists():
                with open(franchise_meta_file, "r") as f:
                    franchise_metadata = json.load(f)
            else:
                # Create basic metadata if file doesn't exist
                franchise_metadata = {
                    "name": franchise_name,
                    "description": "",
                    "owner": "",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "permissions": {"read": True, "write": True, "admin": False},
                    "character_count": 0,
                    "voice_availability": 0.0,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }

            # Get character count
            franchise_dir = self.characters_dir / f"franchise={franchise_name}"
            character_count = len(list(franchise_dir.glob("*.yaml")))
            franchise_metadata["character_count"] = character_count

            # Get voice availability
            voice_count = 0
            for char_file in franchise_dir.glob("*.yaml"):
                with open(char_file, "r") as f:
                    char_data = yaml.safe_load(f)
                    if char_data.get("voice", {}).get("available", False):
                        voice_count += 1

            franchise_metadata["voice_availability"] = (
                voice_count / character_count if character_count > 0 else 0.0
            )
            franchise_metadata["last_updated"] = datetime.now(timezone.utc).isoformat()

            # Save updated metadata
            with open(franchise_meta_file, "w") as f:
                json.dump(franchise_metadata, f, indent=2)

            return dict(franchise_metadata)

        except Exception as e:
            logger.error(f"Error getting franchise info for '{franchise_name}': {e}")
            return None

    async def list_franchises(self) -> List[Dict[str, Any]]:
        """List all franchises with their information."""
        franchises = []

        try:
            # Scan franchise directories
            for franchise_dir in self.characters_dir.glob("franchise=*"):
                franchise_name = franchise_dir.name.replace("franchise=", "")
                franchise_info = await self.get_franchise_info(franchise_name)
                if franchise_info:
                    franchises.append(franchise_info)

            # Sort by creation date
            franchises.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return franchises

        except Exception as e:
            logger.error(f"Error listing franchises: {e}")
            return []

    async def delete_franchise(self, franchise_name: str, force: bool = False) -> bool:
        """Delete a franchise and all its characters."""
        try:
            if not await self.franchise_exists(franchise_name):
                raise ValueError(f"Franchise '{franchise_name}' does not exist")

            if not force:
                # Check if franchise has characters
                franchise_dir = self.characters_dir / f"franchise={franchise_name}"
                character_count = len(list(franchise_dir.glob("*.yaml")))
                if character_count > 0:
                    raise ValueError(
                        f"Franchise '{franchise_name}' has {character_count} characters. "
                        f"Use --force to delete anyway."
                    )

            # Remove franchise directory
            franchise_dir = self.characters_dir / f"franchise={franchise_name}"
            if franchise_dir.exists():
                shutil.rmtree(franchise_dir)

            # Remove franchise metadata
            franchise_meta_file = self.metadata_dir / f"franchise_{franchise_name}.json"

            if franchise_meta_file.exists():
                franchise_meta_file.unlink()

            # Update main index
            await self._remove_franchise_from_index(franchise_name)

            logger.info(f"Deleted franchise '{franchise_name}'")
            return True

        except Exception as e:
            logger.error(f"Error deleting franchise '{franchise_name}': {e}")
            raise

    async def _update_franchise_index(
        self, franchise_name: str, franchise_metadata: Dict[str, Any]
    ) -> None:
        """Update main index with franchise information."""
        try:
            self.index["franchises"][franchise_name] = {
                "name": franchise_name,
                "description": franchise_metadata.get("description", ""),
                "owner": franchise_metadata.get("owner", ""),
                "created_at": franchise_metadata.get("created_at", ""),
                "character_count": franchise_metadata.get("character_count", 0),
                "voice_availability": franchise_metadata.get("voice_availability", 0.0),
            }

            self.index["last_updated"] = datetime.now(timezone.utc).isoformat()

            # Save updated index
            with open(self.index_file, "w") as f:
                json.dump(self.index, f, indent=2)

        except Exception as e:
            logger.error(f"Error updating franchise index: {e}")

    async def _remove_franchise_from_index(self, franchise_name: str) -> None:
        """Remove franchise from main index."""
        try:
            if franchise_name in self.index.get("franchises", {}):
                del self.index["franchises"][franchise_name]
                self.index["last_updated"] = datetime.now(timezone.utc).isoformat()

                # Save updated index
                with open(self.index_file, "w") as f:
                    json.dump(self.index, f, indent=2)

        except Exception as e:
            logger.error(f"Error removing franchise from index: {e}")
