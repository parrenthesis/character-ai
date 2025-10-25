"""Model instantiation and preloading coordination."""

import logging
import time
from typing import Any, Dict, List, Optional

from .config import Config
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelInstantiator:
    """Handles model instantiation and preloading."""

    def __init__(
        self,
        config: Config,
        model_registry: ModelRegistry,
        edge_optimizer: Optional[Any] = None,
        hardware_config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.model_registry = model_registry
        self.edge_optimizer = edge_optimizer
        self.hardware_config = hardware_config or {}

    async def preload_models(
        self,
        models: List[str],  # ["stt", "llm", "tts"]
        loaded_models: Dict[str, Any],
        loaded_model_names: Dict[str, str],
    ) -> Dict[str, bool]:
        """Coordinate pre-loading of multiple models"""
        results = {}

        for model_name in models:
            try:
                if model_name == "stt":
                    success = await self._preload_stt(loaded_models, loaded_model_names)
                    results["stt"] = success

                elif model_name == "llm":
                    success = await self._preload_llm(loaded_models, loaded_model_names)
                    results["llm"] = success

                elif model_name == "tts":
                    success = await self._preload_tts(loaded_models, loaded_model_names)
                    results["tts"] = success

                else:
                    logger.warning(f"Unknown model type for preloading: {model_name}")
                    results[model_name] = False

            except Exception as e:
                logger.error(f"Failed to preload {model_name}: {e}")
                results[model_name] = False

        return results

    async def _preload_stt(
        self, loaded_models: Dict[str, Any], loaded_model_names: Dict[str, str]
    ) -> bool:
        """Pre-initialize STT with hardware-specific configuration."""
        start_time = time.time()

        from ..algorithms.conversational_ai.processors.stt.wav2vec2_processor import (
            Wav2Vec2Processor,
        )

        # Get hardware-specific STT config
        hw_stt_config = self.model_registry.get_hardware_model_config("stt")
        if hw_stt_config:
            # Use hardware-specific model
            stt_model = hw_stt_config.get("model_name", "facebook/wav2vec2-base-960h")
            logger.info(f"Using hardware-specific STT model: {stt_model}")
        else:
            # Use default model
            stt_model = "facebook/wav2vec2-base-960h"
            logger.info(f"Using default STT model: {stt_model}")

        if self.edge_optimizer:
            stt_config = await self.edge_optimizer.optimize_wav2vec2_for_toy()
        else:
            stt_config = self.config

        # Check if hardware config forces CPU for STT
        force_cpu = False
        if self.hardware_config:
            stt_hw_config = self.hardware_config.get("optimizations", {}).get("stt", {})
            force_cpu = stt_hw_config.get("use_cpu", False)
            if force_cpu:
                logger.info(
                    "Hardware config forces CPU for STT to avoid CUDA conflicts"
                )

        stt_processor = Wav2Vec2Processor(
            stt_config, model_name=stt_model, force_cpu=force_cpu
        )
        await stt_processor.initialize()
        loaded_models["stt"] = stt_processor
        loaded_model_names["stt"] = stt_model  # Track which model was loaded
        load_time = time.time() - start_time
        logger.info(f"✅ STT model pre-loaded in {load_time:.2f}s")
        # Console echo for test visibility
        try:
            import os as _os

            if _os.getenv("CAI_ENVIRONMENT", "").lower() == "testing":
                import click as _click

                _click.echo(f"✅ STT model pre-loaded in {load_time:.2f}s")
        except Exception:
            pass

        # Clear CUDA cache after STT loads
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache after STT load")

        return True

    async def _preload_llm(
        self, loaded_models: Dict[str, Any], loaded_model_names: Dict[str, str]
    ) -> bool:
        """Pre-initialize LLM with hardware-specific configuration."""
        start_time = time.time()

        from ..algorithms.conversational_ai.processors.llm.llama_cpp_processor import (
            LlamaCppProcessor,
        )

        # Get hardware-specific LLM config
        hw_llm_config = self.model_registry.get_hardware_model_config("llm")
        if hw_llm_config:
            # Use hardware-specific model and optimizations
            # Get model name from hardware config, not from the registry config
            if self.hardware_config and "models" in self.hardware_config:
                model_name = self.hardware_config["models"].get(
                    "llm", "llama-3.2-3b-instruct"
                )
            else:
                model_name = hw_llm_config.get("model_name", "llama-3.2-3b-instruct")
            optimizations = hw_llm_config.get("optimizations", {})

            # Set the model path in the config so the LLM processor uses it
            if "model_path" in hw_llm_config:
                self.config.models.llama_gguf_path = hw_llm_config["model_path"]
                logger.info(f"Set model path in config: {hw_llm_config['model_path']}")

            logger.info(
                f"Using hardware-specific LLM: {model_name} with optimizations: {optimizations}"
            )
        else:
            # Fallback to edge optimizer or default config
            if self.edge_optimizer:
                llm_config = await self.edge_optimizer.optimize_llm_for_toy()
            else:
                llm_config = self.config

            # Check if hardware config forces CPU for LLM
            force_cpu = False
            if self.hardware_config:
                llm_hw_config = self.hardware_config.get("optimizations", {}).get(
                    "llm", {}
                )
                force_cpu = llm_hw_config.get("use_cpu", False)

            llm_processor = LlamaCppProcessor(
                llm_config, use_cpu=force_cpu, hardware_config=self.hardware_config
            )
            await llm_processor.initialize()
            loaded_models["llm"] = llm_processor
            loaded_model_names["llm"] = "llama-3.2-3b-instruct"  # Default
            logger.info("✅ LLM model pre-loaded")

            # Clear CUDA cache after LLM loads
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache after LLM load")

            return True

        # If we have hardware config, use it
        force_cpu = hw_llm_config.get("use_cpu", False)
        llm_processor = LlamaCppProcessor(
            self.config, use_cpu=force_cpu, hardware_config=self.hardware_config
        )
        await llm_processor.initialize()
        loaded_models["llm"] = llm_processor
        loaded_model_names["llm"] = model_name
        load_time = time.time() - start_time
        model_name = (
            self.hardware_config.get("models", {}).get("llm", "unknown")
            if self.hardware_config
            else "unknown"
        )
        hw_config = (
            self.hardware_config.get("optimizations", {}).get("llm", {})
            if self.hardware_config
            else {}
        )
        n_threads = hw_config.get("n_threads", "?")
        n_gpu_layers = hw_config.get("n_gpu_layers", 0)
        logger.info(
            f"✅ LLM model pre-loaded in {load_time:.2f}s ({model_name}, {n_threads} threads, {n_gpu_layers} GPU layers)"
        )
        # Console echo for test visibility
        try:
            import os as _os

            if _os.getenv("CAI_ENVIRONMENT", "").lower() == "testing":
                import click as _click

                _click.echo(
                    f"✅ LLM model pre-loaded in {load_time:.2f}s ({model_name}, {n_threads} threads, {n_gpu_layers} GPU layers)"
                )
        except Exception:
            pass

        # Clear CUDA cache after LLM loads
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache after LLM load")

        return True

    async def _preload_tts(
        self, loaded_models: Dict[str, Any], loaded_model_names: Dict[str, str]
    ) -> bool:
        """Pre-initialize TTS with hardware-specific configuration."""
        start_time = time.time()

        from ..algorithms.conversational_ai.processors.tts.coqui_processor import (
            CoquiProcessor,
        )

        # Get hardware-specific TTS config
        hw_tts_config = self.model_registry.get_hardware_model_config("tts")
        if hw_tts_config:
            # Get the registry key from hardware config, not the resolved model name
            if self.hardware_config and "models" in self.hardware_config:
                tts_model = self.hardware_config["models"].get("tts", "coqui-xtts-v2")
            else:
                tts_model = "coqui-xtts-v2"
            logger.info(f"Using hardware-specific TTS model: {tts_model}")
        else:
            # Use default model
            tts_model = "coqui-xtts-v2"
            logger.info(f"Using default TTS model: {tts_model}")

        if self.edge_optimizer:
            tts_config = await self.edge_optimizer.optimize_coqui_for_toy()
        else:
            tts_config = self.config

        # Get TTS hardware configuration
        use_half_precision = None
        if self.hardware_config:
            tts_hw_config = self.hardware_config.get("optimizations", {}).get("tts", {})
            use_half_precision = tts_hw_config.get("use_half_precision")

        # Get model info from registry to get the full model name
        model_info = self.model_registry.get_model_info("tts", tts_model)
        if not model_info:
            logger.warning(f"Model {tts_model} not found in registry for type tts")
            return False

        tts_processor = CoquiProcessor(
            tts_config,
            model_name=model_info["model_name"],  # Use full model name from registry
            gpu_device=None,  # Let it auto-detect
            use_half_precision=use_half_precision,
        )
        await tts_processor.initialize()
        loaded_models["tts"] = tts_processor
        loaded_model_names["tts"] = tts_model  # Track which model was loaded
        load_time = time.time() - start_time
        logger.info(f"✅ TTS model pre-loaded in {load_time:.2f}s")
        # Console echo for test visibility
        try:
            import os as _os

            if _os.getenv("CAI_ENVIRONMENT", "").lower() == "testing":
                import click as _click

                _click.echo(f"✅ TTS model pre-loaded in {load_time:.2f}s")
        except Exception:
            pass

        return True

    async def preload_models_with_config(
        self,
        model_configs: Dict[str, Dict[str, Any]],
        loaded_models: Dict[str, Any],
        loaded_model_names: Dict[str, str],
    ) -> Dict[str, bool]:
        """Preload models using specific configurations."""
        results = {}

        for model_type, config in model_configs.items():
            try:
                model_name = config.get("model_name")
                if not model_name:
                    logger.warning(f"No model_name specified for {model_type}")
                    results[model_type] = False
                    continue

                # Get model info from registry
                model_info = self.model_registry.get_model_info(model_type, model_name)
                if not model_info:
                    logger.warning(
                        f"Model {model_name} not found in registry for type {model_type}"
                    )
                    results[model_type] = False
                    continue

                # Instantiate processor
                processor = self._instantiate_processor(
                    model_type, model_name, model_info
                )
                await processor.initialize()

                # Store in loaded models
                old_processor = loaded_models.get(model_type)
                loaded_models[model_type] = processor
                # Store the original registry key, not the resolved model name
                loaded_model_names[model_type] = model_name
                results[model_type] = True

                logger.debug(
                    f"preload_models_with_config: {model_type} model {model_name} - old processor: {id(old_processor) if old_processor else None}, new processor: {id(processor)}"
                )
                logger.info(f"✅ {model_type.upper()} model {model_name} pre-loaded")

            except Exception as e:
                logger.error(f"Failed to preload {model_type}: {e}")
                results[model_type] = False

        return results

    def _instantiate_processor(
        self,
        model_type: str,
        model_name: str,
        model_info: Dict,
        gpu_device: Optional[str] = None,
    ) -> Any:
        """Dynamically instantiate processor based on registry info."""
        # Import based on class name
        if model_type == "stt":
            from ..algorithms.conversational_ai.processors.stt.wav2vec2_processor import (
                Wav2Vec2Processor,
            )

            return Wav2Vec2Processor(self.config, model_name=model_info["model_name"])
        elif model_type == "llm":
            from ..algorithms.conversational_ai.processors.llm.llama_cpp_processor import (
                LlamaCppProcessor,
            )

            # Check if hardware config forces CPU for LLM
            force_cpu = False
            if self.hardware_config:
                llm_hw_config = self.hardware_config.get("optimizations", {}).get(
                    "llm", {}
                )
                force_cpu = llm_hw_config.get("use_cpu", False)
            return LlamaCppProcessor(
                self.config, use_cpu=force_cpu, hardware_config=self.hardware_config
            )
        elif model_type == "tts":
            from ..algorithms.conversational_ai.processors.tts.coqui_processor import (
                CoquiProcessor,
            )

            # Get TTS hardware configuration (use_half_precision)
            # Note: gpu_device is already passed as parameter, don't override it here
            use_half_precision = None
            if self.hardware_config:
                tts_config = self.hardware_config.get("optimizations", {}).get(
                    "tts", {}
                )
                use_half_precision = tts_config.get("use_half_precision")

            logger.debug(
                f"TTS Processor Config: model_name={model_name}, model_info={model_info}"
            )
            logger.debug(f"Hardware Config TTS: {self.hardware_config.get('tts', {})}")
            logger.debug(
                f"TTS Hardware Optimizations: {self.hardware_config.get('optimizations', {}).get('tts', {})}"
            )
            logger.debug(
                f"TTS use_half_precision: {use_half_precision}, gpu_device: {gpu_device}"
            )

            return CoquiProcessor(
                self.config,
                model_name=model_info["model_name"],
                gpu_device=gpu_device,  # Use the parameter passed in, don't reset to None
                use_half_precision=use_half_precision,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
