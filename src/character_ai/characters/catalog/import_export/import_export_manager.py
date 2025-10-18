"""
Import/Export operations for catalog management.

Handles catalog import and export functionality.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class ImportExportService:
    """Handles catalog import and export operations."""

    def __init__(
        self, catalog_dir: Path, storage_manager: Any, search_manager: Any
    ) -> None:
        self.catalog_dir = catalog_dir
        self.characters_dir = catalog_dir / "characters"
        self.exports_dir = catalog_dir / "exports"
        self.storage_manager = storage_manager
        self.search_manager = search_manager

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
                        self.exports_dir
                        / f"catalog_export_{franchise}_{timestamp}.yaml"
                    )
                else:
                    output_file = self.exports_dir / f"catalog_export_{timestamp}.yaml"

            # Load search index to get all characters
            search_index = await self.search_manager._load_search_index()

            export_data: dict[str, Any] = {
                "export_info": {
                    "timestamp": datetime.now().isoformat(),
                    "franchise_filter": franchise,
                    "total_characters": 0,
                },
                "characters": [],
            }

            # Export characters
            for char_key, char_data in search_index.get("characters", {}).items():
                if franchise and char_data.get("franchise") != franchise:
                    continue

                # Load full character data
                character = await self.storage_manager.load_character(
                    char_data["name"], char_data["franchise"]
                )
                if character:
                    # Convert to dict for export
                    char_dict = self.storage_manager._character_to_dict(
                        character, char_data["franchise"]
                    )
                    export_data["characters"].append(char_dict)

            export_data["export_info"]["total_characters"] = len(
                export_data["characters"]
            )

            # Write export file
            with open(output_file, "w") as f:
                yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)

            logger.info(
                f"Exported {len(export_data['characters'])} characters to {output_file}"
            )
            return output_file

        except Exception as e:
            logger.error(f"Error exporting catalog: {e}")
            raise

    async def import_catalog(
        self,
        import_file: Path,
        franchise: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Import catalog from YAML file."""
        try:
            from ....core.config.yaml_loader import YAMLConfigLoader

            import_data = YAMLConfigLoader.load_yaml(import_file)

            if not import_data or "characters" not in import_data:
                raise ValueError("Invalid import file format")

            results: dict[str, Any] = {
                "imported_count": 0,
                "skipped": 0,
                "errors": 0,
                "errors_list": [],
                "voice_processed_count": 0,
                "total_characters": 0,
            }

            # Load current index
            index = self.storage_manager._load_or_create_index()

            for char_data in import_data["characters"]:
                try:
                    # Determine target franchise
                    target_franchise = franchise or char_data.get(
                        "franchise", "imported"
                    )

                    # Check if character already exists
                    existing_char = await self.storage_manager.load_character(
                        char_data["name"], target_franchise
                    )

                    if existing_char and not overwrite:
                        logger.info(f"Skipping existing character: {char_data['name']}")
                        results["skipped"] += 1
                        continue

                    # Convert dict back to Character object
                    character = self.storage_manager._dict_to_character(char_data)

                    # Store character
                    await self.storage_manager.store_character(
                        character, index, target_franchise
                    )

                    results["imported_count"] += 1
                    results["total_characters"] += 1
                    logger.info(f"Imported character: {char_data['name']}")

                except Exception as e:
                    error_msg = f"Error importing character {char_data.get('name', 'unknown')}: {e}"
                    logger.error(error_msg)
                    results["errors"] += 1
                    results["errors_list"].append(error_msg)

            # Update search index
            await self.search_manager._load_search_index()

            logger.info(
                f"Import completed: {results['imported_count']} imported, "
                f"{results['skipped']} skipped, {results['errors']} errors"
            )

            return results

        except Exception as e:
            logger.error(f"Error importing catalog: {e}")
            raise
