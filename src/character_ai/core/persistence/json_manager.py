"""
Centralized JSON persistence utilities.

Eliminates duplicate JSON load/save patterns across managers and provides
consistent error handling, logging, and atomic operations.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class JSONRepository:
    """Centralized JSON persistence with consistent error handling and atomic operations."""

    @staticmethod
    def load_json(
        path: Path, default: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Load JSON data from file with consistent error handling.

        Args:
            path: Path to JSON file
            default: Default value to return if file doesn't exist or fails to load

        Returns:
            Dictionary containing JSON data or default value
        """
        if default is None:
            default = {}

        if not path.exists():
            logger.debug(f"JSON file does not exist: {path}")
            return default

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data is None:
                logger.warning(f"JSON file is empty: {path}")
                return default

            logger.debug(f"Successfully loaded JSON from {path}")
            return data if isinstance(data, dict) else default

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file {path}: {e}")
            return default
        except Exception as e:
            logger.error(f"Unexpected error loading JSON file {path}: {e}")
            return default

    @staticmethod
    def save_json(path: Path, data: Dict[str, Any], *, atomic: bool = True) -> bool:
        """
        Save JSON data to file with atomic operation support.

        Args:
            path: Path to save JSON file
            data: Dictionary to save as JSON
            atomic: If True, write to temp file then rename (atomic operation)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            if atomic:
                # Write to temporary file first, then rename (atomic operation)
                temp_file = path.with_suffix(".tmp")
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                temp_file.rename(path)
            else:
                # Direct write
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Successfully saved JSON to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save JSON file {path}: {e}")
            return False

    @staticmethod
    def load_json_objects(
        path: Path,
        from_dict_fn: Callable[[Dict[str, Any]], T],
        default: Optional[Dict[str, T]] = None,
    ) -> Dict[str, T]:
        """
        Load JSON data and convert to objects using provided conversion function.

        Args:
            path: Path to JSON file
            from_dict_fn: Function to convert dict to object (e.g., Profile.from_dict)
            default: Default value to return if file doesn't exist or fails to load

        Returns:
            Dictionary of converted objects
        """
        if default is None:
            default = {}

        raw_data = JSONRepository.load_json(path, {})
        if not raw_data:
            return default

        try:
            objects = {}
            for key, obj_data in raw_data.items():
                if isinstance(obj_data, dict):
                    objects[key] = from_dict_fn(obj_data)
                else:
                    logger.warning(f"Skipping non-dict value for key '{key}' in {path}")

            logger.debug(f"Successfully loaded {len(objects)} objects from {path}")
            return objects

        except Exception as e:
            logger.error(f"Failed to convert JSON objects from {path}: {e}")
            return default

    @staticmethod
    def save_json_objects(
        path: Path,
        objects: Dict[str, Any],
        to_dict_fn: Callable[[Any], Dict[str, Any]],
        *,
        atomic: bool = True,
    ) -> bool:
        """
        Convert objects to dictionaries and save as JSON.

        Args:
            path: Path to save JSON file
            objects: Dictionary of objects to save
            to_dict_fn: Function to convert object to dict (e.g., profile.to_dict)
            atomic: If True, use atomic write operation

        Returns:
            True if successful, False otherwise
        """
        try:
            data = {}
            for key, obj in objects.items():
                if hasattr(obj, to_dict_fn.__name__):
                    data[key] = to_dict_fn(obj)
                else:
                    logger.warning(
                        f"Object {key} does not have {to_dict_fn.__name__} method"
                    )
                    data[key] = obj  # Fallback to raw object

            return JSONRepository.save_json(path, data, atomic=atomic)

        except Exception as e:
            logger.error(f"Failed to convert objects to JSON for {path}: {e}")
            return False

    @staticmethod
    def backup_json(path: Path, backup_suffix: str = ".backup") -> Optional[Path]:
        """
        Create a backup of existing JSON file.

        Args:
            path: Path to JSON file to backup
            backup_suffix: Suffix to add to backup filename

        Returns:
            Path to backup file if successful, None otherwise
        """
        if not path.exists():
            logger.debug(f"No file to backup: {path}")
            return None

        try:
            backup_path = path.with_suffix(path.suffix + backup_suffix)
            backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            logger.info(f"Created backup: {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Failed to create backup of {path}: {e}")
            return None

    @staticmethod
    def restore_from_backup(path: Path, backup_suffix: str = ".backup") -> bool:
        """
        Restore JSON file from backup.

        Args:
            path: Path to JSON file to restore
            backup_suffix: Suffix of backup file

        Returns:
            True if successful, False otherwise
        """
        backup_path = path.with_suffix(path.suffix + backup_suffix)
        if not backup_path.exists():
            logger.error(f"Backup file does not exist: {backup_path}")
            return False

        try:
            backup_path.rename(path)
            logger.info(f"Restored from backup: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore from backup {backup_path}: {e}")
            return False
