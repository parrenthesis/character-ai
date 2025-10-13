"""
Edge model optimization utilities for toy hardware constraints.

Produces lightweight Config instances tuned for edge (2-4GB RAM, <500ms latency).
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Dict, Optional

from .config import Config, Environment


class EdgeModelOptimizer:
    """Create per-model configs optimized for toy constraints.

    The returned Config objects are safe defaults for edge devices:
    - Prefer tiny/compact models
    - Enable quantization-friendly settings
    - Avoid gated/auth-required models by default
    """

    def __init__(self, constraints: Any, base_config: Optional[Config] = None) -> None:
        self.constraints = constraints
        self.base_config = base_config or Config(environment=Environment.TESTING)
        self._summary: Dict[str, Any] = {
            "created_at": time.time(),
            "constraints": {
                "max_memory_gb": getattr(constraints, "max_memory_gb", 4.0),
                "max_cpu_cores": getattr(constraints, "max_cpu_cores", 4),
                "battery_life_hours": getattr(constraints, "battery_life_hours", 8.0),
                "target_latency_ms": getattr(constraints, "target_latency_ms", 500),
            },
            "optimizations": {},
        }

    async def optimize_wav2vec2_for_toy(self) -> Config:
        """Return a Config tuned for Wav2Vec2 on edge."""
        cfg = deepcopy(self.base_config)
        cfg.models.wav2vec2_model = (
            cfg.models.wav2vec2_model or "facebook/wav2vec2-base"
        )
        cfg.interaction.sample_rate = cfg.interaction.sample_rate or 16000
        cfg.use_free_models_only = True
        cfg.enable_cpu_limiting = True
        cfg.max_cpu_threads = getattr(self.constraints, "max_cpu_cores", 2)
        self._summary["optimizations"]["wav2vec2"] = {
            "model": cfg.models.wav2vec2_model,
            "sample_rate": cfg.models.sample_rate,
        }
        return cfg

    async def optimize_llm_for_toy(self) -> Config:
        """Return a Config tuned for a small, ungated, token-free LLM.

        Defaults to TinyLlama which is compatible with AutoModel and does not
        require gated access or tokens.
        """
        cfg = deepcopy(self.base_config)
        cfg.models.llama_backend = "llama_cpp"  # Use llama.cpp for edge deployment
        cfg.models.llama_model = (
            cfg.models.llama_model or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        cfg.models.llama_quantization = cfg.models.llama_quantization or "4bit"
        cfg.use_free_models_only = True
        cfg.enable_cpu_limiting = True
        cfg.max_cpu_threads = getattr(self.constraints, "max_cpu_cores", 2)
        self._summary["optimizations"]["llm"] = {
            "model": cfg.models.llama_model,
            "quantization": cfg.models.llama_quantization,
        }
        return cfg

    async def optimize_coqui_for_toy(self) -> Config:
        """Return a Config tuned for Coqui TTS on edge."""
        cfg = deepcopy(self.base_config)
        cfg.models.coqui_model = (
            cfg.models.coqui_model or "tts_models/en/ljspeech/tacotron2-DDC"
        )
        cfg.use_free_models_only = True
        cfg.enable_cpu_limiting = True
        cfg.max_cpu_threads = getattr(self.constraints, "max_cpu_cores", 2)
        self._summary["optimizations"]["coqui"] = {"model": cfg.models.coqui_model}
        return cfg

    async def get_edge_optimization_summary(self) -> Dict[str, Any]:
        """Return a snapshot of the applied edge optimizations."""
        return dict(self._summary)

    # Compatibility with older tests expecting a single call
    async def optimize_all_models_for_toy(self) -> Dict[str, Any]:
        """Run per-model optimizations and return a combined summary."""
        await self.optimize_wav2vec2_for_toy()
        await self.optimize_llm_for_toy()
        await self.optimize_coqui_for_toy()
        return await self.get_edge_optimization_summary()
