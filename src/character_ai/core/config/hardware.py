"""
Hardware configuration for Character AI Platform.

Contains GPU and hardware-related configuration classes.
"""

from dataclasses import dataclass


@dataclass
class GPUConfig:
    """GPU configuration settings."""

    device: str = "cuda:0"
    memory_limit: str = "2GB"  # Edge default; adjusted per environment
    precision: str = "mixed"  # mixed, fp16, int8, int4
    batch_size: int = 8
    quantization: str = "fp16"  # 8bit, 4bit, fp16
    gradient_checkpointing: bool = True
    torch_compile: bool = True
    cuda_graphs: bool = True
    max_gpu_memory_gb: float = 20.0  # Leave 4GB for system
    cache_size: int = 1000  # Embedding cache size

    def __post_init__(self) -> None:
        """Validate GPU configuration."""
        if self.precision not in ["mixed", "fp16", "int8", "int4"]:
            raise ValueError(f"Invalid precision: {self.precision}")
        if self.quantization not in ["8bit", "4bit", "fp16"]:
            raise ValueError(f"Invalid quantization: {self.quantization}")
