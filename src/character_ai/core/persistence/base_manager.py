"""
Base data manager for common manager functionality.

Provides common patterns for data managers including initialization,
storage management, and persistence operations.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from .json_manager import JSONRepository

logger = logging.getLogger(__name__)


class BaseDataManager(ABC):
    """Base class for data managers with common initialization and persistence patterns."""

    def __init__(self, storage_path: Path, manager_name: str):
        """
        Initialize base data manager.

        Args:
            storage_path: Path to storage directory
            manager_name: Name of the manager for logging
        """
        self.storage_path = Path(storage_path)
        self.manager_name = manager_name
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the data manager."""
        if not self._initialized:
            # Ensure storage directory exists
            self.storage_path.mkdir(parents=True, exist_ok=True)

            # Load initial data
            await self._load_data()

            self._initialized = True
            logger.info(
                f"{self.manager_name} initialized with storage: {self.storage_path}"
            )

    async def shutdown(self) -> None:
        """Shutdown the data manager."""
        if self._initialized:
            # Save any pending data
            await self._save_data()

            self._initialized = False
            logger.info(f"{self.manager_name} shutdown")

    async def health_check(self) -> bool:
        """Check if the data manager is healthy."""
        try:
            return self._initialized and self.storage_path.exists()
        except Exception as e:
            logger.error(f"Health check failed for {self.manager_name}: {e}")
            return False

    def get_storage_path(self) -> Path:
        """Get the storage path."""
        return self.storage_path

    def ensure_storage_exists(self) -> None:
        """Ensure storage directory exists."""
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def load_json_data(
        self, filename: str, default: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Load JSON data from storage.

        Args:
            filename: Name of the JSON file
            default: Default value if file doesn't exist

        Returns:
            Dictionary containing JSON data
        """
        file_path = self.storage_path / filename
        return JSONRepository.load_json(file_path, default or {})

    def save_json_data(self, filename: str, data: Dict[str, Any]) -> bool:
        """
        Save JSON data to storage.

        Args:
            filename: Name of the JSON file
            data: Dictionary to save

        Returns:
            True if successful, False otherwise
        """
        file_path = self.storage_path / filename
        return JSONRepository.save_json(file_path, data)

    def backup_data(
        self, filename: str, backup_suffix: str = ".backup"
    ) -> Optional[Path]:
        """
        Create backup of data file.

        Args:
            filename: Name of the file to backup
            backup_suffix: Suffix for backup file

        Returns:
            Path to backup file if successful, None otherwise
        """
        file_path = self.storage_path / filename
        return JSONRepository.backup_json(file_path, backup_suffix)

    def restore_from_backup(
        self, filename: str, backup_suffix: str = ".backup"
    ) -> bool:
        """
        Restore data from backup.

        Args:
            filename: Name of the file to restore
            backup_suffix: Suffix of backup file

        Returns:
            True if successful, False otherwise
        """
        file_path = self.storage_path / filename
        return JSONRepository.restore_from_backup(file_path, backup_suffix)

    @abstractmethod
    async def _load_data(self) -> None:
        """Load data during initialization. Must be implemented by subclasses."""
        pass

    @abstractmethod
    async def _save_data(self) -> None:
        """Save data during shutdown. Must be implemented by subclasses."""
        pass
