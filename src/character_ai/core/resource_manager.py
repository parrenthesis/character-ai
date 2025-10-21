"""Runtime resource coordination and model lifecycle management."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

import psutil

from .config import Config
from .model_instantiator import ModelInstantiator
from .model_registry import ModelRegistry
from .model_warmup import ModelWarmup

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manages runtime resource allocation and model lifecycle coordination."""

    def __init__(
        self,
        config: Config,
        edge_optimizer: Optional[Any] = None,
        hardware_config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.edge_optimizer = edge_optimizer
        self.hardware_config = hardware_config or {}
        self._loaded_models: Dict[str, Union[Any, None]] = {}
        self._loaded_model_names: Dict[
            str, str
        ] = {}  # Track which specific model is loaded
        self._model_last_used: Dict[str, float] = {}
        self._idle_timeout = 300  # 5 minutes

        # Initialize component services
        self._model_registry = ModelRegistry(config, hardware_config)
        self._model_instantiator = ModelInstantiator(
            config, self._model_registry, edge_optimizer, hardware_config
        )
        self._model_warmup = ModelWarmup(self._loaded_models)

    def get_model_info(self, model_type: str, model_name: str) -> Dict[str, Any]:
        """Get model metadata from registry."""
        return self._model_registry.get_model_info(model_type, model_name)

    def get_loaded_model_name(self, model_type: str) -> Optional[str]:
        """Get the name of the currently loaded model for a given type."""
        return self._loaded_model_names.get(model_type)

    def list_available_models(
        self, model_type: str, hardware_constraints: Optional[Dict] = None
    ) -> List[str]:
        """List available models, filtered by hardware if specified."""
        return self._model_registry.list_available_models(
            model_type, hardware_constraints
        )

    async def preload_models(
        self,
        models: List[str],  # ["stt", "llm", "tts"]
    ) -> Dict[str, bool]:
        """Coordinate pre-loading of multiple models"""
        return await self._model_instantiator.preload_models(
            models, self._loaded_models, self._loaded_model_names
        )

    async def preload_models_with_config(
        self,
        model_configs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, bool]:
        """Preload models using specific configurations."""
        return await self._model_instantiator.preload_models_with_config(
            model_configs, self._loaded_models, self._loaded_model_names
        )

    async def warmup_all_models(
        self, character: Optional[Any] = None
    ) -> Dict[str, bool]:
        """Warm up all loaded models with dummy inference."""
        return await self._model_warmup.warmup_all_models(character)

    def pin_models(self, pin: bool = True) -> None:
        """Pin/unpin models to prevent idle unloading."""
        if pin:
            self._idle_timeout = 0  # Never unload
            logger.info("Models pinned - will not be unloaded due to inactivity")
        else:
            self._idle_timeout = 300  # 5 minutes
            logger.info(
                "Models unpinned - will be unloaded after 5 minutes of inactivity"
            )

    async def check_memory_pressure(self) -> bool:
        """Detect if system is under memory pressure"""
        try:
            memory = psutil.virtual_memory()
            # Consider high pressure if >=90% memory used
            return memory.percent >= 90
        except Exception as e:
            logger.warning(f"Failed to check memory pressure: {e}")
            return False

    async def unload_idle_models(self) -> None:
        """Unload models that haven't been used recently"""
        if self._idle_timeout == 0:
            return  # Models are pinned

        current_time = time.time()
        models_to_unload = []

        for model_name, last_used in self._model_last_used.items():
            if current_time - last_used > self._idle_timeout:
                models_to_unload.append(model_name)

        for model_name in models_to_unload:
            if model_name in self._loaded_models:
                logger.info(f"Unloading idle model: {model_name}")

                # Call shutdown on the processor if it has one
                processor = self._loaded_models[model_name]
                if processor and hasattr(processor, "shutdown"):
                    try:
                        if asyncio.iscoroutinefunction(processor.shutdown):
                            await processor.shutdown()
                        else:
                            processor.shutdown()
                    except Exception as e:
                        logger.warning(f"Error shutting down {model_name}: {e}")

                del self._loaded_models[model_name]
                if model_name in self._loaded_model_names:
                    del self._loaded_model_names[model_name]
                del self._model_last_used[model_name]

        if models_to_unload:
            # Clear CUDA cache after unloading
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache after model unloading")

    def mark_model_used(self, model_name: str) -> None:
        """Mark a model as recently used"""
        self._model_last_used[model_name] = time.time()

    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return list(self._loaded_models.keys())

    def get_processor(self, model_name: str) -> Optional[Any]:
        """Get a loaded processor by name"""
        return self._loaded_models.get(model_name)

    def get_stt_processor(self) -> Optional[Any]:
        """Get the STT processor"""
        return self._loaded_models.get("stt")

    def get_llm_processor(self) -> Optional[Any]:
        """Get the LLM processor"""
        return self._loaded_models.get("llm")

    def get_tts_processor(self) -> Optional[Any]:
        """Get the TTS processor"""
        return self._loaded_models.get("tts")
