"""
Franchise management operations for catalog management.

Handles franchise creation, management, and organization.
"""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FranchiseService:
    """Handles franchise management operations."""

    def __init__(self, catalog_dir: Path, storage_manager: Any) -> None:
        self.catalog_dir = catalog_dir
        self.characters_dir = catalog_dir / "characters"
        self.metadata_dir = catalog_dir / "metadata"
        self.index_file = catalog_dir / "index.json"
        self.storage_manager = storage_manager

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
                from ....core.config.yaml_loader import YAMLConfigLoader

                char_data = YAMLConfigLoader.load_yaml(char_file)
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
            # Load current index
            index = self.storage_manager._load_or_create_index()

            index["franchises"][franchise_name] = {
                "name": franchise_name,
                "description": franchise_metadata.get("description", ""),
                "owner": franchise_metadata.get("owner", ""),
                "created_at": franchise_metadata.get("created_at", ""),
                "character_count": franchise_metadata.get("character_count", 0),
                "voice_availability": franchise_metadata.get("voice_availability", 0.0),
            }

            index["last_updated"] = datetime.now(timezone.utc).isoformat()

            # Save updated index
            with open(self.index_file, "w") as f:
                json.dump(index, f, indent=2)

        except Exception as e:
            logger.error(f"Error updating franchise index: {e}")

    async def _remove_franchise_from_index(self, franchise_name: str) -> None:
        """Remove franchise from main index."""
        try:
            # Load current index
            index = self.storage_manager._load_or_create_index()

            if franchise_name in index.get("franchises", {}):
                del index["franchises"][franchise_name]
                index["last_updated"] = datetime.now(timezone.utc).isoformat()

                # Save updated index
                with open(self.index_file, "w") as f:
                    json.dump(index, f, indent=2)

        except Exception as e:
            logger.error(f"Error removing franchise from index: {e}")
