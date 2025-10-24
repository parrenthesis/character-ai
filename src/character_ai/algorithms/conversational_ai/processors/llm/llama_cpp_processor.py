"""
Llama.cpp backend processor for CPU-only, low-RAM deployments using GGUF.
"""

from __future__ import annotations

# CRITICAL: Import torch_init FIRST to set environment variables before any torch imports
# isort: off
from .....core import torch_init  # noqa: F401

# isort: on

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .....core.config import Config
from .....core.exceptions import AudioProcessingError, ModelError
from .....core.protocols import (
    BaseTextProcessor,
    EmbeddingResult,
    ModelInfo,
    TextResult,
)

logger = logging.getLogger(__name__)


class LlamaCppProcessor(BaseTextProcessor):
    def __init__(
        self,
        config: Config,
        use_cpu: bool = False,
        hardware_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__("llama_cpp")
        self.config: Config = config
        self.use_cpu = use_cpu
        self.hardware_config = hardware_config
        self.model: Optional[Any] = None

    async def initialize(self) -> None:
        try:
            from llama_cpp import Llama

            gguf_path = getattr(self.config.models, "llama_gguf_path", "")
            if not Path(gguf_path).exists():
                raise FileNotFoundError(f"GGUF model not found: {gguf_path}")

            # Get context size from config
            n_ctx = getattr(self.config.models, "llama_n_ctx", 2048)

            # Get optimization parameters from config
            n_threads = getattr(self.config, "max_cpu_threads", 2)
            n_batch = 128

            # Check for GPU availability and enable offloading if possible
            n_gpu_layers = 0
            if self.use_cpu:
                logger.info(
                    "Using CPU-only mode (forced by hardware config to avoid CUDA conflicts)"
                )
            else:
                try:
                    import torch

                    if torch.cuda.is_available():
                        # Get GPU layers from hardware config, fallback to default
                        if self.hardware_config:
                            llm_hw_config = self.hardware_config.get(
                                "optimizations", {}
                            ).get("llm", {})
                            n_gpu_layers = llm_hw_config.get("n_gpu_layers", 20)
                        else:
                            n_gpu_layers = 20  # Default fallback

                        if n_gpu_layers > 0:
                            logger.info(
                                f"Enabling GPU offloading with {n_gpu_layers} layers"
                            )
                        else:
                            logger.info("GPU offloading disabled by hardware config")
                    else:
                        logger.info("CUDA not available, using CPU-only mode")
                except ImportError:
                    logger.info("PyTorch not available, using CPU-only mode")

            self.model = Llama(
                model_path=gguf_path,
                n_threads=n_threads,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                logits_all=False,
                n_batch=n_batch,
                verbose=False,
            )

            self.model_info = ModelInfo(
                name="llama_cpp",
                type="language_model",
                size="gguf",
                memory_usage="~1-2GB (q4)",
                precision="int4",
                quantization="q4",
                loaded_at=time.time(),
                status="loaded",
            )
            self._initialized = True
            logger.info("llama.cpp model loaded")
        except Exception as e:
            logger.error(f"Failed to load llama.cpp model: {e}")
            raise ModelError(
                f"Failed to load llama.cpp model: {e}", component="LlamaCppProcessor"
            )

    async def shutdown(self) -> None:
        self.model = None
        self._initialized = False

    async def process_text(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> TextResult:
        if not self._initialized or self.model is None:
            raise AudioProcessingError(
                "LlamaCpp processor not initialized", component="LlamaCppProcessor"
            )
        try:
            start = time.time()
            # Check if prompt is already formatted (contains character name or system markers)
            # If so, use it directly; otherwise apply _prepare_prompt wrapper
            if self._is_preformatted_prompt(text):
                prompt = text  # Use as-is for character prompts from LLMPromptBuilder
                logger.debug("Using preformatted character prompt")
            else:
                prompt = await self._prepare_prompt(text, context)
                logger.debug("Applied generic prompt wrapper")

            # Use configured max_tokens for complete responses
            max_tokens = (
                self.config.interaction.max_new_tokens
            )  # Use config value (controlled via runtime.yaml)

            out = self.model(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=self.config.interaction.temperature,
                stop=[
                    "</s>",
                    "\n\n",  # Stop at double newline (end of thought)
                    "User:",
                    "Human:",
                    " - ",  # Stop at list markers
                    "Assistant:",
                ],  # Stop tokens for shorter responses (removed bare "\n" to allow multi-line responses)
            )
            resp = out["choices"][0]["text"].strip()
            dt = time.time() - start

            # Log if response is empty to help debug
            if not resp:
                logger.warning(
                    f"Empty response from llama.cpp for prompt: {prompt[:100]}..."
                )

            return TextResult(
                text=resp,
                metadata={"model": "llama_cpp", "processing_time": dt},
                processing_time=dt,
            )
        except Exception as e:
            logger.error(f"Error generating with llama.cpp: {e}")
            return await self._create_error_result(str(e))

    def _is_preformatted_prompt(self, text: str) -> bool:
        """Check if prompt is already formatted (e.g., from LLMPromptBuilder)."""
        # Look for markers that indicate a preformatted character prompt
        preformat_markers = [
            "You are ",  # Character introduction
            "Key characteristics:",  # Character description
            "Output rules:",  # Formatting rules
            "<conversation_history>",  # Conversation context
            "\nUser:",  # Dialogue format
            "CRITICAL:",  # Instruction emphasis
        ]
        return any(marker in text for marker in preformat_markers)

    async def _prepare_prompt(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Wrap generic text input with system prompt (only for non-character prompts)."""
        system_prompt = "You are a helpful AI assistant for children."
        if context and context.get("conversation_history"):
            system_prompt += (
                f"\n\nPrevious conversation:\n{context['conversation_history']}"
            )
        return f"[SYSTEM]\n{system_prompt}\n[USER]\n{text}\n[ASSISTANT]"

    async def _create_error_result(self, error_message: str) -> TextResult:
        return TextResult(
            text="",
            error=error_message,
            metadata={"component": "LlamaCppProcessor", "error": True},
        )

    async def generate_text(self, prompt: str, **kwargs: Any) -> TextResult:
        """Generate text from prompt."""
        return await self.process_text(prompt, {"generate": True, **kwargs})

    async def get_embeddings(self, text: str) -> EmbeddingResult:
        """Extract embeddings from text."""
        if self.model is None:
            return EmbeddingResult(
                embeddings=[],
                error="Model not initialized",
                metadata={"component": "LlamaCppProcessor", "error": True},
            )

        try:
            # Use the model's embedding functionality
            # Llama.cpp models can generate embeddings using the model's internal representations
            embeddings = self.model.create_embedding(text)
            return EmbeddingResult(
                embeddings=embeddings,
                metadata={
                    "component": "LlamaCppProcessor",
                    "method": "llama_cpp_embedding",
                },
            )
        except Exception as e:
            return EmbeddingResult(
                embeddings=[],
                error=f"Failed to generate embeddings: {str(e)}",
                metadata={"component": "LlamaCppProcessor", "error": True},
            )
