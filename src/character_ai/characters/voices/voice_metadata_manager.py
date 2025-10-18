"""
Base class for voice metadata management.

Provides common functionality for loading and saving voice metadata across
different voice manager implementations.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from ...core.persistence.json_manager import JSONRepository

logger = logging.getLogger(__name__)


class VoiceMetadataService:
    """Base class for voice metadata management operations."""

    def __init__(self, metadata_file: Path):
        """Initialize voice metadata manager.

        Args:
            metadata_file: Path to the voice metadata JSON file
        """
        self.metadata_file = metadata_file
        self.voice_metadata: Dict[str, Any] = self._load_voice_metadata()

    def _load_voice_metadata(self) -> Dict[str, Any]:
        """Load voice metadata from storage.

        Returns:
            Dictionary containing voice metadata with default structure
        """
        default_metadata = {
            "characters": {},
            "voice_quality_scores": {},
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

        return JSONRepository.load_json(self.metadata_file, default_metadata)

    def _save_voice_metadata(self) -> None:
        """Save voice metadata to storage."""
        self.voice_metadata["last_updated"] = datetime.now(timezone.utc).isoformat()

        success = JSONRepository.save_json(self.metadata_file, self.voice_metadata)
        if not success:
            logger.error("Error saving voice metadata")

    def get_voice_metadata(self) -> Dict[str, Any]:
        """Get current voice metadata.

        Returns:
            Current voice metadata dictionary
        """
        return self.voice_metadata

    def update_voice_metadata(self, updates: Dict[str, Any]) -> None:
        """Update voice metadata with new values.

        Args:
            updates: Dictionary of updates to apply to metadata
        """
        self.voice_metadata.update(updates)
        self._save_voice_metadata()

    def get_character_voice_info(self, character_key: str) -> Dict[str, Any]:
        """Get voice information for a character.

        Args:
            character_key: Key identifying the character in metadata

        Returns:
            Character voice information dictionary
        """
        return dict(self.voice_metadata["characters"].get(character_key, {}))

    def set_character_voice_info(
        self, character_key: str, voice_info: Dict[str, Any]
    ) -> None:
        """Set voice information for a character.

        Args:
            character_key: Key identifying the character in metadata
            voice_info: Voice information dictionary to store
        """
        self.voice_metadata["characters"][character_key] = voice_info
        self._save_voice_metadata()

    def remove_character_voice_info(self, character_key: str) -> bool:
        """Remove voice information for a character.

        Args:
            character_key: Key identifying the character in metadata

        Returns:
            True if character was removed, False if not found
        """
        if character_key in self.voice_metadata["characters"]:
            del self.voice_metadata["characters"][character_key]
            self._save_voice_metadata()
            return True
        return False

    def list_characters_with_voice(self) -> list[str]:
        """List all character keys that have voice information.

        Returns:
            List of character keys with voice data
        """
        return list(self.voice_metadata["characters"].keys())

    def get_voice_quality_score(self, character_key: str) -> float:
        """Get voice quality score for a character.

        Args:
            character_key: Key identifying the character in metadata

        Returns:
            Voice quality score (0.0-1.0), default 0.0 if not found
        """
        return float(
            self.voice_metadata["voice_quality_scores"].get(character_key, 0.0)
        )

    def set_voice_quality_score(self, character_key: str, score: float) -> None:
        """Set voice quality score for a character.

        Args:
            character_key: Key identifying the character in metadata
            score: Quality score (0.0-1.0)
        """
        self.voice_metadata["voice_quality_scores"][character_key] = score
        self._save_voice_metadata()
