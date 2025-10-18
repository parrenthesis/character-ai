"""
Centralized YAML configuration loading utilities.

Provides consistent YAML loading with error handling, validation, and logging.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")


class YAMLConfigLoader:
    """Centralized YAML configuration loader with consistent error handling."""

    @staticmethod
    def load_yaml(path: Path) -> Dict[str, Any]:
        """
        Load YAML file with consistent error handling and logging.

        Args:
            path: Path to YAML file

        Returns:
            Dictionary containing YAML data, empty dict if file is empty/invalid

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        if not path.exists():
            raise FileNotFoundError(f"YAML file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if data is None:
                logger.warning(f"YAML file is empty or contains only comments: {path}")
                return {}

            logger.debug(f"Successfully loaded YAML from {path}")
            return data if isinstance(data, dict) else {}

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file {path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading YAML file {path}: {e}")
            raise

    @staticmethod
    def load_yaml_schema(path: Path, schema: Type[T]) -> T:
        """
        Load YAML file and validate against a schema.

        Args:
            path: Path to YAML file
            schema: Pydantic model or dataclass to validate against

        Returns:
            Validated object of type T

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValidationError: If data doesn't match schema
        """
        data = YAMLConfigLoader.load_yaml(path)

        try:
            if hasattr(schema, "from_dict"):
                # Pydantic model
                return schema.from_dict(data)  # type: ignore
            elif hasattr(schema, "__dataclass_fields__"):
                # Dataclass
                return schema(**data)
            else:
                # Assume it's a regular class with __init__
                return schema(**data)

        except Exception as e:
            logger.error(f"Schema validation failed for {path}: {e}")
            raise

    @staticmethod
    def load_yaml_safe(
        path: Path, default: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Load YAML file safely, returning default on any error.

        Args:
            path: Path to YAML file
            default: Default value to return on error (defaults to empty dict)

        Returns:
            Dictionary containing YAML data or default value
        """
        if default is None:
            default = {}

        try:
            return YAMLConfigLoader.load_yaml(path)
        except Exception as e:
            logger.warning(f"Failed to load YAML file {path}, using default: {e}")
            return default

    @staticmethod
    def save_yaml(data: Dict[str, Any], path: Path) -> None:
        """
        Save data to YAML file with consistent formatting.

        Args:
            data: Dictionary to save
            path: Path to save YAML file

        Raises:
            yaml.YAMLError: If YAML serialization fails
        """
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

            logger.debug(f"Successfully saved YAML to {path}")

        except Exception as e:
            logger.error(f"Failed to save YAML file {path}: {e}")
            raise


# Convenience functions for backward compatibility
def load_yaml(path: Path) -> Dict[str, Any]:
    """Convenience function for loading YAML files."""
    return YAMLConfigLoader.load_yaml(path)


def load_yaml_safe(
    path: Path, default: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience function for safe YAML loading."""
    return YAMLConfigLoader.load_yaml_safe(path, default)
