"""
Llama.cpp backend processor for CPU-only, low-RAM deployments using GGUF.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ...core.config import Config
from ...core.exceptions import AudioProcessingError, ModelError
from ...core.protocols import BaseTextProcessor, ModelInfo, TextResult

logger = logging.getLogger(__name__)


class LlamaCppProcessor(BaseTextProcessor):
    def __init__(self, config: Config):
        super().__init__("llama_cpp")
        self.config: Config = config
        self.model: Optional[Any] = None

    async def initialize(self) -> None:
        try:
            from llama_cpp import Llama

            gguf_path = getattr(self.config.models, "llama_gguf_path", "")
            if not Path(gguf_path).exists():
                raise FileNotFoundError(f"GGUF model not found: {gguf_path}")

            self.model = Llama(
                model_path=gguf_path,
                n_threads=getattr(self.config, "max_cpu_threads", 2),
                n_ctx=2048,
                logits_all=False,
                n_batch=128,
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
            prompt = await self._prepare_prompt(text, context)
            out = self.model(
                prompt=prompt,
                max_tokens=self.config.interaction.max_new_tokens,
                temperature=self.config.interaction.temperature,
                stop=["</s>"],
            )
            resp = out["choices"][0]["text"].strip()
            dt = time.time() - start
            return TextResult(
                text=resp,
                metadata={"model": "llama_cpp", "processing_time": dt},
                processing_time=dt,
            )
        except Exception as e:
            logger.error(f"Error generating with llama.cpp: {e}")
            return await self._create_error_result(str(e))

    async def _prepare_prompt(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
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
