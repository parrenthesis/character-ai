"""Model registry for managing available models and their metadata."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages model metadata and availability queries."""

    def __init__(
        self, config: Config, hardware_config: Optional[Dict[str, Any]] = None
    ):
        self.config = config
        self.hardware_config = hardware_config or {}
        self._model_registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from config."""
        # Try to get model registry from runtime config first
        registry = getattr(self.config.runtime, "model_registry", {})

        # If not found, try to load directly from YAML file
        if not registry:
            try:
                # Find the runtime.yaml file
                current_dir = os.path.dirname(os.path.abspath(__file__))
                search_dir = current_dir
                runtime_yaml_path = None

                # Walk up the directory tree to find the project root
                for _ in range(5):  # Limit search depth
                    potential_path = os.path.join(search_dir, "configs", "runtime.yaml")
                    if os.path.exists(potential_path):
                        runtime_yaml_path = potential_path
                        break
                    search_dir = os.path.dirname(search_dir)

                if runtime_yaml_path:
                    from .config.yaml_loader import YAMLConfigLoader

                    yaml_data = YAMLConfigLoader.load_yaml(Path(runtime_yaml_path))
                    registry = yaml_data.get("model_registry", {})
                    logger.info(f"Loaded model registry from {runtime_yaml_path}")
            except Exception as e:
                logger.warning(f"Failed to load model registry from YAML: {e}")
                registry = {}

        return registry

    def get_model_info(self, model_type: str, model_name: str) -> Dict[str, Any]:
        """Get model metadata from registry."""
        result = self._model_registry.get(model_type, {}).get(model_name, {})
        return result if result is not None else {}

    def list_available_models(
        self, model_type: str, hardware_constraints: Optional[Dict] = None
    ) -> List[str]:
        """List available models, filtered by hardware if specified."""
        models = self._model_registry.get(model_type, {})
        if not hardware_constraints:
            return list(models.keys())

        # Filter by hardware constraints
        compatible_models = []
        for model_name, model_info in models.items():
            if self._check_hardware_compatibility(model_info, hardware_constraints):
                compatible_models.append(model_name)

        return compatible_models

    def _check_hardware_compatibility(
        self, requirements: Dict, constraints: Dict
    ) -> bool:
        """Check if model requirements are compatible with hardware constraints."""
        # Simple compatibility check - can be extended
        if "memory_gb" in requirements and "available_memory_gb" in constraints:
            if requirements["memory_gb"] > constraints["available_memory_gb"]:
                return False
        return True

    def get_hardware_model_config(self, model_type: str) -> Optional[Dict[str, Any]]:
        """Get hardware-specific model configuration, overriding registry defaults."""
        if not self.hardware_config:
            return None

        # Get model selection from hardware config
        models_config = self.hardware_config.get("models", {})
        selected_model = models_config.get(model_type)

        if not selected_model:
            return None

        # Get the model info from registry
        model_info = self.get_model_info(model_type, selected_model)
        if not model_info:
            logger.warning(
                f"Model {selected_model} not found in registry for type {model_type}"
            )
            return None

        # Merge with hardware-specific overrides
        hw_config = self.hardware_config.get("optimizations", {}).get(model_type, {})
        merged_config = {**model_info, **hw_config}

        logger.info(
            f"Hardware config for {model_type}: {selected_model} with overrides: {hw_config}"
        )
        return merged_config
