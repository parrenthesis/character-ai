"""
Centralized PyTorch environment variable initialization.

This module MUST be imported before any other modules that import torch
to ensure proper environment variable setup for PyTorch 2.8 compatibility.
"""

import os

# Set environment variable to fix XTTS v2 compatibility with PyTorch 2.8
# This MUST be set before torch is imported anywhere in the codebase
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# Add early CPU thread limiting for all threading libraries
# This MUST happen before any imports of numpy, scipy, librosa, soundfile, torch, etc.


# Try to detect hardware config early for thread count
def _get_early_thread_limit() -> int:
    """Get thread limit from hardware config or environment."""

    # Check for explicit env var override (testing)
    if os.environ.get("CAI_ENABLE_CPU_LIMITING", "false").lower() == "true":
        return int(os.environ.get("CAI_MAX_CPU_THREADS", "2"))

    # Try to load hardware config
    try:
        from pathlib import Path

        import yaml

        # Detect hardware profile
        config_dir = Path("configs/hardware")
        if config_dir.exists():
            # Simple detection: check for GPU
            has_gpu = os.path.exists("/dev/nvidia0") or os.path.exists("/dev/dri")

            if has_gpu:
                config_file = config_dir / "desktop.yaml"
            else:
                # Check CPU count for Pi detection
                import multiprocessing

                cpu_count = multiprocessing.cpu_count()
                if cpu_count <= 4:
                    config_file = config_dir / "raspberry_pi.yaml"
                else:
                    config_file = config_dir / "orange_pi.yaml"

            if config_file.exists():
                with open(config_file) as f:
                    hw_config = yaml.safe_load(f)
                    n_threads = (
                        hw_config.get("optimizations", {})
                        .get("llm", {})
                        .get("n_threads")
                    )
                    if n_threads:
                        return int(n_threads)
    except Exception:
        pass  # Fallback to default

    # Default: use half of available cores (conservative)
    try:
        import multiprocessing

        return max(2, multiprocessing.cpu_count() // 2)
    except Exception:
        return 4  # Safe fallback


# Apply thread limits before any library imports
_thread_limit = _get_early_thread_limit()
os.environ["OPENBLAS_NUM_THREADS"] = str(_thread_limit)
os.environ["MKL_NUM_THREADS"] = str(_thread_limit)
os.environ["OMP_NUM_THREADS"] = str(_thread_limit)
os.environ["NUMEXPR_NUM_THREADS"] = str(_thread_limit)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(_thread_limit)

# Additional environment variables for Coqui TTS and other libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Reduce transformers verbosity
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # Disable progress bars
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"  # Disable telemetry
os.environ[
    "PYTORCH_ALLOC_CONF"
] = "max_split_size_mb:128"  # Limit CUDA memory allocation

# Additional aggressive thread limiting
os.environ["MKL_DYNAMIC"] = "false"  # Disable MKL dynamic threading
os.environ["OMP_THREAD_LIMIT"] = str(_thread_limit)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # Limit OpenMP nesting
os.environ["OMP_WAIT_POLICY"] = "passive"  # Use passive waiting
os.environ["OMP_DYNAMIC"] = "false"  # Disable dynamic thread adjustment

# CRITICAL: Set PyTorch thread limits immediately after env vars, before any torch usage
# This must happen before torch is imported anywhere else in the codebase
try:
    import torch

    torch.set_num_threads(_thread_limit)
    try:
        torch.set_num_interop_threads(_thread_limit)
    except RuntimeError:
        # Interop threads already set - this is expected if torch was imported elsewhere first
        # The environment variables we set above should still help limit threading
        pass
except Exception:
    # If torch import fails, that's fine - it will be imported later
    pass

# Note: CPU affinity was removed as it was causing process kills
# The environment variables and PyTorch thread limits should be sufficient

# Additional PyTorch environment variables for stability
os.environ.setdefault(
    "PYTHONWARNINGS",
    "ignore::RuntimeWarning:sys:1,ignore::RuntimeWarning:unittest.mock,ignore::RuntimeWarning:tracemalloc",
)

# Log that torch environment has been initialized
import logging  # noqa: E402 - Must import after setting env vars

logger = logging.getLogger(__name__)
logger.debug("PyTorch environment variables initialized")
