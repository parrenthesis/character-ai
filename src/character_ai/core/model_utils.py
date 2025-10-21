"""Shared utilities for model loading."""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_local_model_path(
    config: Any, model_type: str, model_name: str
) -> Optional[str]:
    """
    Get local model path from registry.

    Args:
        config: Configuration object with model_registry
        model_type: Type of model ('stt', 'tts', 'llm')
        model_name: HuggingFace model name to match

    Returns:
        Absolute path to local model directory, or None if not found
    """
    try:
        if not hasattr(config, "runtime") or not hasattr(
            config.runtime, "model_registry"
        ):
            return None

        registry = config.runtime.model_registry
        models = registry.get(model_type, {})

        # Find model by matching model_name
        for model_info in models.values():
            if model_info.get("model_name") == model_name:
                model_path = model_info.get("model_path")
                if model_path:
                    path = Path(model_path)
                    if not path.is_absolute():
                        path = Path.cwd() / path
                    if path.exists():
                        return str(path)
                    logger.warning(f"Model path does not exist: {path}")
        return None
    except Exception as e:
        logger.warning(f"Error getting local model path: {e}")
        return None
