"""
Catalog voice management system for enterprise character voice integration.

Provides voice cloning and management integrated with catalog storage system.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...characters.catalog import CatalogStorage
from ...core.protocols import VoiceManagerProtocol
from .voice_manager import VoiceService
from .voice_metadata_manager import VoiceMetadataService

logger = logging.getLogger(__name__)


class CatalogVoiceService(VoiceManagerProtocol, VoiceMetadataService):
    """Voice management integrated with catalog storage system."""

    def __init__(self, catalog_storage: Optional[CatalogStorage] = None):
        """Initialize catalog voice manager."""
        self.catalog_storage = catalog_storage or CatalogStorage()
        self.voice_manager = VoiceService()
        self.voice_metadata_file = (
            self.catalog_storage.metadata_dir / "voice_metadata.json"
        )

        # Initialize VoiceMetadataService with the metadata file
        VoiceMetadataService.__init__(self, self.voice_metadata_file)

        # Add franchises to metadata (CatalogVoiceService specific)
        if "franchises" not in self.voice_metadata:
            self.voice_metadata["franchises"] = {}

    async def initialize(self) -> None:
        """Initialize the catalog voice manager."""
        # Ensure voice storage directory exists
        self.catalog_storage.metadata_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"CatalogVoiceService initialized with metadata dir: {self.catalog_storage.metadata_dir}"
        )

    async def clone_character_voice(
        self,
        character_name: str,
        voice_file_path: str,
        franchise: str = "original",
        tts_processor: Optional[Any] = None,
        quality_score: Optional[float] = None,
    ) -> bool:
        """Clone voice for a character and integrate with catalog."""
        try:
            # Validate character exists in catalog
            character = await self.catalog_storage.load_character(
                character_name, franchise
            )
            if not character:
                logger.error(
                    f"Character '{character_name}' not found in franchise '{franchise}'"
                )
                return False

            # Clone voice using existing voice manager
            success = await self.voice_manager.inject_character_voice(
                character_name, voice_file_path, tts_processor
            )

            if not success:
                return False

            # Update catalog with voice information
            await self._update_character_voice_info(
                character_name, franchise, voice_file_path, quality_score
            )

            logger.info(
                f"Voice cloned for character '{character_name}' in franchise "
                f"'{franchise}'"
            )
            return True

        except Exception as e:
            logger.error(f"Error cloning voice for {character_name}: {e}")
            return False

    async def _update_character_voice_info(
        self,
        character_name: str,
        franchise: str,
        voice_file_path: str,
        quality_score: Optional[float] = None,
    ) -> None:
        """Update character voice information in catalog."""
        try:
            # Load character from catalog
            character = await self.catalog_storage.load_character(
                character_name, franchise
            )
            if not character:
                return

            # Get voice file info
            voice_info = await self.voice_manager.get_voice_info(character_name)
            if "error" in voice_info:
                return

            # Update voice metadata
            voice_key = f"{franchise}_{character_name}"
            self.voice_metadata["characters"][voice_key] = {
                "character_name": character_name,
                "franchise": franchise,
                "voice_file_path": voice_file_path,
                "voice_storage_path": voice_info.get("voice_file"),
                "file_size_bytes": voice_info.get("file_size_bytes", 0),
                "file_size_mb": voice_info.get("file_size_mb", 0),
                "quality_score": quality_score or 0.8,  # Default quality score
                "cloned_at": datetime.now(timezone.utc).isoformat(),
                "available": True,
            }

            # Update franchise voice stats
            if franchise not in self.voice_metadata["franchises"]:
                self.voice_metadata["franchises"][franchise] = {
                    "total_characters": 0,
                    "characters_with_voice": 0,
                    "voice_availability": 0.0,
                }

            self.voice_metadata["franchises"][franchise]["characters_with_voice"] += 1

            # Update quality scores
            if quality_score:
                self.voice_metadata["voice_quality_scores"][voice_key] = quality_score

            # Save metadata
            self._save_voice_metadata()

        except Exception as e:
            logger.error(f"Error updating voice info for {character_name}: {e}")

    def get_character_voice_info(self, character_key: str) -> Dict[str, Any]:
        """Get voice information for a character."""
        return dict(self.voice_metadata["characters"].get(character_key, {}))

    def list_characters_with_voice(self) -> List[str]:
        """List all characters with voice information."""
        return list(self.voice_metadata["characters"].keys())

    async def get_franchise_voice_stats(self, franchise: str) -> Dict[str, Any]:
        """Get voice statistics for a franchise."""
        return dict(
            self.voice_metadata["franchises"].get(
                franchise,
                {
                    "total_characters": 0,
                    "characters_with_voice": 0,
                    "voice_availability": 0.0,
                },
            )
        )

    async def search_characters_by_voice_quality(
        self,
        min_quality: float = 0.0,
        max_quality: float = 1.0,
        franchise: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search characters by voice quality score."""
        results = []

        for voice_key, voice_info in self.voice_metadata["characters"].items():
            if franchise is not None and voice_info.get("franchise") != franchise:
                continue

            quality_score = voice_info.get("quality_score", 0.0)
            if min_quality <= quality_score <= max_quality:
                results.append(voice_info)

        return results

    async def remove_character_voice(
        self, character_name: str, franchise: str = "original"
    ) -> bool:
        """Remove character voice from catalog."""
        try:
            # Remove from voice manager
            success = await self.voice_manager.remove_character_voice(character_name)

            if success:
                # Update metadata
                voice_key = f"{franchise}_{character_name}"
                if voice_key in self.voice_metadata["characters"]:
                    del self.voice_metadata["characters"][voice_key]

                # Update franchise stats
                if franchise in self.voice_metadata["franchises"]:
                    self.voice_metadata["franchises"][franchise][
                        "characters_with_voice"
                    ] = max(
                        0,
                        self.voice_metadata["franchises"][franchise][
                            "characters_with_voice"
                        ]
                        - 1,
                    )

                # Save metadata
                self._save_voice_metadata()

                logger.info(
                    f"Removed voice for character '{character_name}' in franchise "
                    f"'{franchise}'"
                )

            return success

        except Exception as e:
            logger.error(f"Error removing voice for {character_name}: {e}")
            return False

    async def get_voice_analytics(self) -> Dict[str, Any]:
        """Get comprehensive voice analytics."""
        try:
            total_characters = len(self.voice_metadata["characters"])
            total_franchises = len(self.voice_metadata["franchises"])

            # Calculate overall voice availability
            total_with_voice = sum(
                franchise_info.get("characters_with_voice", 0)
                for franchise_info in self.voice_metadata["franchises"].values()
            )

            overall_availability = total_with_voice / max(total_characters, 1)

            # Quality score statistics
            quality_scores = [
                voice_info.get("quality_score", 0.0)
                for voice_info in self.voice_metadata["characters"].values()
            ]

            avg_quality = (
                sum(quality_scores) / max(len(quality_scores), 1)
                if quality_scores
                else 0.0
            )
            max_quality = max(quality_scores) if quality_scores else 0.0
            min_quality = min(quality_scores) if quality_scores else 0.0

            # Franchise breakdown
            franchise_breakdown = {}
            for franchise, franchise_info in self.voice_metadata["franchises"].items():
                franchise_breakdown[franchise] = {
                    "total_characters": franchise_info.get("total_characters", 0),
                    "characters_with_voice": franchise_info.get(
                        "characters_with_voice", 0
                    ),
                    "voice_availability": franchise_info.get("characters_with_voice", 0)
                    / max(franchise_info.get("total_characters", 1), 1),
                }

            return {
                "total_characters": total_characters,
                "total_franchises": total_franchises,
                "characters_with_voice": total_with_voice,
                "overall_voice_availability": overall_availability,
                "quality_statistics": {
                    "average_quality": avg_quality,
                    "max_quality": max_quality,
                    "min_quality": min_quality,
                    "total_quality_scores": len(quality_scores),
                },
                "franchise_breakdown": franchise_breakdown,
                "last_updated": self.voice_metadata.get("last_updated"),
            }

        except Exception as e:
            logger.error(f"Error getting voice analytics: {e}")
            return {}

    async def export_voice_catalog(
        self, franchise: Optional[str] = None, output_file: Optional[Path] = None
    ) -> Path:
        """Export voice catalog to JSON file."""
        try:
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if franchise:
                    output_file = (
                        self.catalog_storage.exports_dir
                        / f"voice_catalog_{franchise}_{timestamp}.json"
                    )
                else:
                    output_file = (
                        self.catalog_storage.exports_dir
                        / f"voice_catalog_all_{timestamp}.json"
                    )

            # Filter characters by franchise if specified
            characters = []
            for voice_key, voice_info in self.voice_metadata["characters"].items():
                if franchise is None or voice_info.get("franchise") == franchise:
                    characters.append(voice_info)

            # Create export data
            export_data = {
                "voice_catalog_export": {
                    "version": "1.0",
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "franchise": franchise or "all",
                    "total_characters": len(characters),
                },
                "characters": characters,
                "franchise_stats": self.voice_metadata["franchises"],
                "analytics": await self.get_voice_analytics(),
            }

            # Create directory if it doesn't exist
            if not output_file.parent.exists():
                output_file.parent.mkdir(parents=True, exist_ok=True)

            # Write export file
            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Exported voice catalog to {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error exporting voice catalog: {e}")
            raise

    async def import_voice_catalog(self, catalog_file: Path) -> Dict[str, Any]:
        """Import voice catalog from JSON file."""
        try:
            with open(catalog_file, "r") as f:
                data = json.load(f)

            if "voice_catalog_export" not in data:
                raise ValueError(
                    "Invalid voice catalog format: missing "
                    "'voice_catalog_export' section"
                )

            imported_count = 0
            errors = []

            for voice_info in data.get("characters", []):
                try:
                    character_name = voice_info.get("character_name")
                    franchise = voice_info.get("franchise", "imported")

                    # Update voice metadata
                    voice_key = f"{franchise}_{character_name}"
                    self.voice_metadata["characters"][voice_key] = voice_info
                    imported_count += 1

                except Exception as e:
                    errors.append(
                        f"Error importing voice info for "
                        f"{voice_info.get('character_name', 'unknown')}: {e}"
                    )

            # Update franchise stats
            if "franchise_stats" in data:
                self.voice_metadata["franchises"].update(data["franchise_stats"])

            # Save metadata
            self._save_voice_metadata()

            result = {
                "imported_count": imported_count,
                "total_voice_info": len(data.get("characters", [])),
                "errors": errors,
            }

            logger.info(f"Imported {imported_count} voice records from {catalog_file}")
            return result

        except Exception as e:
            logger.error(f"Error importing voice catalog: {e}")
            raise

    async def _save_voice_metadata_entry(
        self, character_name: str, franchise: str, voice_info: Dict[str, Any]
    ) -> None:
        """Save voice metadata entry for a character."""
        try:
            key = f"{franchise}_{character_name}"
            self.voice_metadata["characters"][key] = voice_info

            # Update franchise stats
            if franchise not in self.voice_metadata["franchises"]:
                self.voice_metadata["franchises"][franchise] = {
                    "total_characters": 0,
                    "with_voice": 0,
                    "voice_availability": 0.0,
                }

            self.voice_metadata["franchises"][franchise]["total_characters"] += 1
            if not voice_info.get("is_default", False):
                self.voice_metadata["franchises"][franchise]["with_voice"] += 1

            # Update voice availability
            total_chars = self.voice_metadata["franchises"][franchise][
                "total_characters"
            ]
            with_voice = self.voice_metadata["franchises"][franchise]["with_voice"]
            self.voice_metadata["franchises"][franchise]["voice_availability"] = (
                with_voice / total_chars if total_chars > 0 else 0.0
            )

            # Save metadata
            self._save_voice_metadata()

        except Exception as e:
            logger.error(f"Error saving voice metadata for {character_name}: {e}")
            raise
