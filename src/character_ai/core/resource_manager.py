"""Runtime resource coordination and model lifecycle management."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import psutil

from .config import Config
from .protocols import AudioData

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
        self._model_registry = self._load_registry()

    def _get_hardware_model_config(self, model_type: str) -> Optional[Dict[str, Any]]:
        """Get hardware-specific model configuration, overriding registry defaults."""
        if not self.hardware_config:
            return None

        # Get model selection from hardware config
        models_config = self.hardware_config.get("models", {})
        selected_model = models_config.get(model_type)

        if not selected_model:
            return None

        # Find the model in the registry
        model_registry = self._model_registry.get(model_type, {})
        if selected_model in model_registry:
            model_config_raw = model_registry[selected_model]
            if model_config_raw is not None:
                model_config = model_config_raw.copy()

                # Apply hardware-specific optimizations
                optimizations = self.hardware_config.get("optimizations", {})
                if model_type in optimizations:
                    model_config["optimizations"] = optimizations[model_type]

                logger.info(
                    f"Using hardware-specific {model_type} model: {selected_model}"
                )
                return model_config  # type: ignore[no-any-return]

        logger.warning(
            f"Hardware profile specifies {model_type} model '{selected_model}' but it's not in registry"
        )
        return None

    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from config."""
        # Try to get model registry from runtime config first
        registry = getattr(self.config.runtime, "model_registry", {})

        # If not found, try to load directly from YAML file
        if not registry:
            try:
                import os

                import yaml

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
                    with open(runtime_yaml_path, "r") as f:
                        yaml_data = yaml.safe_load(f)
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

        compatible = []
        for name, info in models.items():
            reqs = info.get("requirements", {})
            if self._check_hardware_compatibility(reqs, hardware_constraints):
                compatible.append(name)
        return compatible

    def _check_hardware_compatibility(
        self, requirements: Dict, constraints: Dict
    ) -> bool:
        """Check if model requirements match hardware constraints."""
        if requirements.get("min_memory_gb", 0) > constraints.get(
            "max_memory_gb", float("inf")
        ):
            return False
        if not requirements.get("cpu_only", True) and constraints.get(
            "cpu_only", False
        ):
            return False
        return True

    async def preload_models(
        self,
        models: List[str],  # ["stt", "llm", "tts"]
    ) -> Dict[str, bool]:
        """Coordinate pre-loading of multiple models"""
        results = {}

        for model_name in models:
            try:
                if model_name == "stt":
                    # Pre-initialize STT with hardware-specific configuration
                    from ..algorithms.conversational_ai.wav2vec2_processor import (
                        Wav2Vec2Processor,
                    )

                    # Get hardware-specific STT config
                    hw_stt_config = self._get_hardware_model_config("stt")
                    if hw_stt_config:
                        # Use hardware-specific model
                        stt_model = hw_stt_config.get(
                            "model_name", "facebook/wav2vec2-base-960h"
                        )
                        logger.info(f"Using hardware-specific STT model: {stt_model}")
                    else:
                        # Use default model
                        stt_model = "facebook/wav2vec2-base-960h"
                        logger.info(f"Using default STT model: {stt_model}")

                    if self.edge_optimizer:
                        stt_config = (
                            await self.edge_optimizer.optimize_wav2vec2_for_toy()
                        )
                    else:
                        stt_config = self.config

                    # Check if hardware config forces CPU for STT
                    force_cpu = False
                    if self.hardware_config:
                        stt_hw_config = self.hardware_config.get(
                            "optimizations", {}
                        ).get("stt", {})
                        force_cpu = stt_hw_config.get("use_cpu", False)
                        if force_cpu:
                            logger.info(
                                "Hardware config forces CPU for STT to avoid CUDA conflicts"
                            )

                    stt_processor = Wav2Vec2Processor(
                        stt_config, model_name=stt_model, force_cpu=force_cpu
                    )
                    await stt_processor.initialize()
                    self._loaded_models["stt"] = stt_processor
                    self._loaded_model_names[
                        "stt"
                    ] = stt_model  # Track which model was loaded
                    results["stt"] = True
                    logger.info("✅ STT model pre-loaded")

                    # Clear CUDA cache after STT loads
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("Cleared CUDA cache after STT load")

                elif model_name == "llm":
                    # Pre-initialize LLM with hardware-specific configuration
                    from ..algorithms.conversational_ai.llama_cpp_processor import (
                        LlamaCppProcessor,
                    )

                    # Get hardware-specific LLM config
                    hw_llm_config = self._get_hardware_model_config("llm")
                    if hw_llm_config:
                        # Use hardware-specific model and optimizations
                        model_name = hw_llm_config.get(
                            "model_name", "llama-3.2-3b-instruct"
                        )
                        optimizations = hw_llm_config.get("optimizations", {})
                        logger.info(
                            f"Using hardware-specific LLM: {model_name} with optimizations: {optimizations}"
                        )
                    else:
                        # Fallback to edge optimizer or default config
                        if self.edge_optimizer:
                            llm_config = (
                                await self.edge_optimizer.optimize_llm_for_toy()
                            )
                        else:
                            llm_config = self.config

                        # Check if hardware config forces CPU for LLM
                        force_cpu = False
                        if self.hardware_config:
                            llm_hw_config = self.hardware_config.get(
                                "optimizations", {}
                            ).get("llm", {})
                            force_cpu = llm_hw_config.get("use_cpu", False)
                            if force_cpu:
                                logger.info(
                                    "Hardware config forces CPU for LLM to avoid CUDA conflicts"
                                )

                        llm_processor = LlamaCppProcessor(
                            llm_config,
                            use_cpu=force_cpu,
                            hardware_config=self.hardware_config,
                        )
                        await llm_processor.initialize()
                        self._loaded_models["llm"] = llm_processor
                        self._loaded_model_names[
                            "llm"
                        ] = "llama-3.2-3b-instruct"  # Track default model
                        results["llm"] = True
                        logger.info("✅ LLM model pre-loaded (default config)")
                        continue

                    # Check if hardware config forces CPU for LLM
                    force_cpu = False
                    if self.hardware_config:
                        llm_hw_config = self.hardware_config.get(
                            "optimizations", {}
                        ).get("llm", {})
                        force_cpu = llm_hw_config.get("use_cpu", False)
                        if force_cpu:
                            logger.info(
                                "Hardware config forces CPU for LLM to avoid CUDA conflicts"
                            )

                    # Create processor with hardware config
                    llm_processor = LlamaCppProcessor(
                        self.config,
                        use_cpu=force_cpu,
                        hardware_config=self.hardware_config,
                    )
                    await llm_processor.initialize()
                    self._loaded_models["llm"] = llm_processor
                    self._loaded_model_names[
                        "llm"
                    ] = model_name  # Track hardware-specific model
                    results["llm"] = True
                    logger.info("✅ LLM model pre-loaded (hardware-specific)")

                    # Clear CUDA cache after LLM loads
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("Cleared CUDA cache after LLM load")

                elif model_name == "tts":
                    # Pre-initialize TTS with hardware-specific configuration
                    from .config import DEFAULT_COQUI_MODEL

                    # Get hardware-specific TTS config
                    hw_tts_config = self._get_hardware_model_config("tts")
                    if hw_tts_config:
                        # Use hardware-specific model
                        tts_model = hw_tts_config.get("model_name", DEFAULT_COQUI_MODEL)
                        logger.info(f"Using hardware-specific TTS model: {tts_model}")
                    else:
                        # Use default model from registry
                        tts_model = DEFAULT_COQUI_MODEL
                        logger.info(f"Using default TTS model: {tts_model}")

                    # Check if we have a valid edge_optimizer (not just hardware_config dict)
                    if self.edge_optimizer and hasattr(
                        self.edge_optimizer, "optimize_coqui_for_toy"
                    ):
                        await self.edge_optimizer.optimize_coqui_for_toy()
                    else:
                        pass

                    logger.info(f"Instantiating TTS processor with model: {tts_model}")

                    # Determine GPU device from hardware config
                    gpu_device = None
                    if self.hardware_config:
                        import torch

                        if torch.cuda.is_available():
                            # Check if TTS should use GPU (default: yes if CUDA available)
                            # use_half_precision controls FP16 vs FP32, not CPU vs GPU
                            gpu_device = "cuda"
                            logger.info(
                                "Hardware config enables GPU acceleration for TTS preload"
                            )
                        else:
                            logger.info("CUDA not available, using CPU for TTS")
                    else:
                        logger.info("No hardware config, using CPU for TTS")

                    # Use _instantiate_processor instead of direct CoquiProcessor creation
                    tts_processor = self._instantiate_processor(
                        "tts", tts_model, {"model_name": tts_model}, gpu_device
                    )
                    await tts_processor.initialize()
                    self._loaded_models["tts"] = tts_processor
                    # Find registry key for this model_name (e.g., "coqui-xtts-v2" for "tts_models/multilingual/multi-dataset/xtts_v2")
                    registry_key = None
                    tts_registry = self._model_registry.get("tts", {})
                    for key, info in tts_registry.items():
                        if info.get("model_name") == tts_model:
                            registry_key = key
                            break
                    self._loaded_model_names["tts"] = (
                        registry_key or tts_model
                    )  # Track registry key or fallback to model path
                    results["tts"] = True
                    logger.info(
                        f"✅ TTS model pre-loaded: {tts_model} (registry key: {registry_key})"
                    )
                else:
                    results[model_name] = False
                    logger.warning(f"Unknown model type: {model_name}")

                if results.get(model_name, False):
                    self._model_last_used[model_name] = time.time()

            except Exception as e:
                logger.error(f"Failed to pre-load {model_name}: {e}")
                results[model_name] = False

        return results

    async def preload_models_with_config(
        self,
        models_config: Dict[str, str],  # {"stt": "wav2vec2-base", ...}
    ) -> Dict[str, bool]:
        """Load specific models by name from registry."""
        results = {}
        for model_type, model_name in models_config.items():
            model_info = self.get_model_info(model_type, model_name)
            if not model_info:
                logger.error(
                    f"Model {model_name} not found in registry for type {model_type}"
                )
                results[model_type] = False
                continue

            # Check if this exact model is already loaded (preserve preload optimization)
            if (
                model_type in self._loaded_models
                and model_type in self._loaded_model_names
            ):
                if self._loaded_model_names.get(model_type) == model_name:
                    logger.info(
                        f"{model_type.upper()} model {model_name} already loaded, skipping reload"
                    )
                    results[model_type] = True
                    continue

            try:
                # Determine GPU device for TTS (same logic as preload_models)
                gpu_device = None
                if model_type == "tts" and self.hardware_config:
                    import torch

                    if torch.cuda.is_available():
                        gpu_device = "cuda"
                        logger.info(
                            f"Hardware config enables GPU for {model_type} model switch"
                        )

                processor = self._instantiate_processor(
                    model_type, model_name, model_info, gpu_device
                )
                await processor.initialize()
                self._loaded_models[model_type] = processor
                self._loaded_model_names[model_type] = model_name  # Track loaded model
                results[model_type] = True
                self._model_last_used[model_type] = time.time()
                logger.info(
                    f"✅ {model_type.upper()} model {model_name} loaded from registry"
                )
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
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
            from ..algorithms.conversational_ai.wav2vec2_processor import (
                Wav2Vec2Processor,
            )

            return Wav2Vec2Processor(self.config, model_name=model_info["model_name"])
        elif model_type == "llm":
            from ..algorithms.conversational_ai.llama_cpp_processor import (
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
            from ..algorithms.conversational_ai.coqui_processor import CoquiProcessor

            # Get TTS hardware configuration (use_half_precision)
            # Note: gpu_device is already passed as parameter, don't override it here
            use_half_precision = None
            if self.hardware_config:
                tts_config = self.hardware_config.get("optimizations", {}).get(
                    "tts", {}
                )
                use_half_precision = tts_config.get("use_half_precision")

            return CoquiProcessor(
                self.config,
                model_name=model_info["model_name"],
                gpu_device=gpu_device,  # Use the parameter passed in, don't reset to None
                use_half_precision=use_half_precision,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    async def warmup_all_models(self) -> Dict[str, bool]:
        """Warm up all loaded models with dummy inference."""
        results = {}

        # Parallel warmup for speed
        warmup_tasks = []
        if "stt" in self._loaded_models:
            warmup_tasks.append(self._warmup_stt())
        if "llm" in self._loaded_models:
            warmup_tasks.append(self._warmup_llm())
        if "tts" in self._loaded_models:
            warmup_tasks.append(self._warmup_tts())

        if warmup_tasks:
            warmup_results = await asyncio.gather(*warmup_tasks, return_exceptions=True)
            for i, model_type in enumerate(["stt", "llm", "tts"]):
                if model_type in self._loaded_models:
                    results[model_type] = not isinstance(warmup_results[i], Exception)

        return results

    async def _warmup_stt(self) -> None:
        """Run dummy audio through STT to warm up caches."""
        logger.info("Warming up STT model...")
        processor = self.get_stt_processor()
        if processor:
            # 1 second of silence
            dummy_audio = AudioData(
                data=np.zeros(16000, dtype=np.float32).tobytes(),
                sample_rate=16000,
                channels=1,
                duration=1.0,
            )
            await processor.process_audio(dummy_audio)
            logger.info("✅ STT warmed up")

    async def _warmup_llm(self) -> None:
        """Run dummy prompt through LLM to warm up."""
        logger.info("Warming up LLM model...")
        processor = self.get_llm_processor()
        if processor:
            # Short dummy prompt to trigger JIT/cache
            await processor.process_text("Hello")
            logger.info("✅ LLM warmed up")

    async def _warmup_tts(self) -> None:
        """Run dummy text through TTS to warm up."""
        logger.info("Warming up TTS model...")
        processor = self.get_tts_processor()
        if processor:
            try:
                # Check if this is XTTS v2 (voice cloning model)
                if (
                    hasattr(processor, "model_name")
                    and "xtts" in processor.model_name.lower()
                ):
                    # XTTS v2 requires voice reference for synthesis
                    # Use a default voice reference for warmup
                    dummy_text = "Hello, this is a warmup test."

                    # Try to find a default voice reference
                    voice_path = None
                    try:
                        # Look for any voice sample in the characters directory
                        import glob

                        # Search for voice samples in processed_samples directories
                        # Use absolute path from project root
                        import os

                        project_root = os.getcwd()
                        voice_pattern = os.path.join(
                            project_root,
                            "configs/characters/*/*/processed_samples/*.wav",
                        )
                        logger.info(
                            f"Searching for voice files with pattern: {voice_pattern}"
                        )
                        voice_files = glob.glob(voice_pattern)
                        logger.info(f"Found voice files: {voice_files}")

                        if voice_files:
                            voice_path = voice_files[0]  # Use first available voice
                            logger.info(f"Using voice sample for warmup: {voice_path}")
                        else:
                            logger.warning("No voice samples found for XTTS v2 warmup")
                    except Exception as e:
                        logger.warning(
                            f"Could not find voice reference for warmup: {e}"
                        )

                    if voice_path:
                        # Warmup with explicit synchronization for CUDA stability
                        logger.info("Warming up XTTS v2 with voice reference")
                        try:
                            await processor.synthesize_speech(
                                dummy_text, voice_path=voice_path
                            )
                            logger.info("✅ XTTS v2 model warmed up successfully")
                        except Exception as e:
                            logger.warning(f"Warmup failed: {e}")
                            logger.info(
                                "✅ TTS warmup skipped (will warm up during first synthesis)"
                            )
                    else:
                        logger.warning(
                            "Skipping XTTS v2 warmup - no voice reference available"
                        )
                        logger.info(
                            "✅ TTS warmup skipped (will warm up during first synthesis)"
                        )
                else:
                    # For non-voice-cloning models, warm up with dummy text
                    dummy_text = "Hello, this is a warmup test."
                    await processor.synthesize_speech(dummy_text)
                    logger.info("✅ TTS model warmed up successfully")
            except Exception as e:
                logger.warning(
                    f"TTS warmup failed, will warm up during first synthesis: {e}"
                )
                logger.info(
                    "✅ TTS warmup skipped (will warm up during first synthesis)"
                )

    def pin_models(self, pin: bool = True) -> None:
        """Pin/unpin models to prevent idle unloading."""
        if pin:
            self._idle_timeout = 999999999  # Very large number = never unload
            logger.info("🔒 Models pinned - idle unloading disabled")
        else:
            self._idle_timeout = 300  # Reset to default
            logger.info("🔓 Models unpinned - idle unloading enabled")

    async def check_memory_pressure(self) -> bool:
        """Detect if system is under memory pressure"""
        try:
            memory = psutil.virtual_memory()
            # Consider memory pressure if usage is above 85%
            return memory.percent > 85.0
        except Exception:
            # If we can't check memory, assume no pressure
            return False

    async def unload_idle_models(self) -> None:
        """Unload models that haven't been used recently"""
        current_time = time.time()
        models_to_unload = []

        for model_name, last_used in self._model_last_used.items():
            if current_time - last_used > self._idle_timeout:
                models_to_unload.append(model_name)

        for model_name in models_to_unload:
            try:
                if model_name in self._loaded_models:
                    processor = self._loaded_models[model_name]
                    # Call shutdown method if available
                    if processor is not None and hasattr(processor, "shutdown"):
                        try:
                            await processor.shutdown()
                        except Exception as shutdown_error:
                            logger.error(
                                f"Error shutting down model {model_name}: {shutdown_error}"
                            )
                            # Continue with unloading even if shutdown fails
                    logger.info(f"Unloaded idle model: {model_name}")
                    del self._loaded_models[model_name]
                del self._model_last_used[model_name]
            except Exception as e:
                logger.error(f"Error unloading model {model_name}: {e}")
                # Continue with other models

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
