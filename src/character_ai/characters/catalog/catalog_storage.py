"""
Catalog storage system for enterprise character management.

Provides hierarchical franchise-based organization with advanced search and analytics.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..management.manager import CharacterService
from ..management.types import Character
from .analytics import AnalyticsService
from .franchise import FranchiseService
from .import_export import ImportExportService
from .search import SearchService
from .storage import CharacterRepository

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

        # Initialize character manager
        self.character_manager = CharacterService()

        # Initialize sub-managers
        self.storage_manager = CharacterRepository(
            self.catalog_dir, self.character_manager
        )
        self.search_manager = SearchService(self.catalog_dir, self.storage_manager)
        self.analytics_manager = AnalyticsService(self.search_manager)
        self.import_export_manager = ImportExportService(
            self.catalog_dir, self.storage_manager, self.search_manager
        )
        self.franchise_manager = FranchiseService(
            self.catalog_dir, self.storage_manager
        )

        # Load or create index
        self.index = self.storage_manager._load_or_create_index()

    # Storage operations
    async def store_character(
        self, character: Character, franchise: str = "original"
    ) -> str:
        """Store character with franchise organization."""
        return await self.storage_manager.store_character(
            character, self.index, franchise
        )

    async def load_character(
        self, character_name: str, franchise: str = "original"
    ) -> Optional[Character]:
        """Load character from catalog storage."""
        return await self.storage_manager.load_character(character_name, franchise)

    # Search operations
    async def search_characters(self, query: Dict[str, Any]) -> List[Character]:
        """Advanced search characters by criteria with text search and fuzzy matching."""
        return await self.search_manager.search_characters(query)

    # Analytics operations
    async def get_catalog_statistics(self) -> Dict[str, Any]:
        """Get comprehensive catalog statistics."""
        return await self.analytics_manager.get_catalog_statistics()

    # Import/Export operations
    async def export_catalog(
        self, franchise: Optional[str] = None, output_file: Optional[Path] = None
    ) -> Path:
        """Export catalog to YAML file."""
        return await self.import_export_manager.export_catalog(franchise, output_file)

    async def import_catalog(
        self,
        import_file: Path,
        franchise: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Import catalog from YAML file."""
        return await self.import_export_manager.import_catalog(
            import_file, franchise, overwrite
        )

    # Franchise management operations
    async def create_franchise(
        self,
        franchise_name: str,
        description: str = "",
        owner: str = "",
        permissions: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create a new franchise with proper isolation."""
        return await self.franchise_manager.create_franchise(
            franchise_name, description, owner, permissions
        )

    async def franchise_exists(self, franchise_name: str) -> bool:
        """Check if franchise exists."""
        return await self.franchise_manager.franchise_exists(franchise_name)

    async def get_franchise_info(self, franchise_name: str) -> Optional[Dict[str, Any]]:
        """Get franchise information and statistics."""
        return await self.franchise_manager.get_franchise_info(franchise_name)

    async def list_franchises(self) -> List[Dict[str, Any]]:
        """List all franchises with their information."""
        return await self.franchise_manager.list_franchises()

    async def delete_franchise(self, franchise_name: str, force: bool = False) -> bool:
        """Delete a franchise and all its characters."""
        return await self.franchise_manager.delete_franchise(franchise_name, force)

    def _find_voice_file(self, character_name: str, voice_dir: Path) -> Optional[Path]:
        """Find voice file for character."""
        if not voice_dir.exists():
            return None

        # Look for common voice file patterns
        patterns = [
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

    def _get_default_voice_style(self, character: Character) -> str:
        """Get default voice style based on character attributes."""
        # Simple voice style mapping based on character dimensions
        if character.dimensions.species.value == "robot":
            return "robotic"
        elif character.dimensions.species.value == "unicorn":
            return "mystical"
        elif character.dimensions.archetype.value == "hero":
            return "confident"
        elif character.dimensions.archetype.value == "villain":
            return "menacing"
        elif character.dimensions.archetype.value == "scholar":
            return "wise"
        else:
            return "neutral"

    async def _set_default_voice_profile(
        self, character: Character, franchise: str, voice_manager: Any
    ) -> None:
        """Set default voice profile for character."""
        # Get default voice style for this character
        voice_style = self._get_default_voice_style(character)

        # Create voice info dictionary
        voice_info = {
            "voice_style": voice_style,
            "is_default": True,
            "character_name": character.name,
            "franchise": franchise,
        }

        # Call voice manager to save metadata
        if hasattr(voice_manager, "_save_voice_metadata_entry"):
            await voice_manager._save_voice_metadata_entry(
                character.name,  # character_name
                franchise,  # franchise
                voice_info,  # voice_info dict
            )
