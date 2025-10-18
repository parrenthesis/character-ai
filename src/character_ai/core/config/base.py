"""
Base configuration infrastructure for Character AI Platform.

Contains core imports, warnings setup, constants, and the Environment enum.
"""

# CRITICAL: Import torch_init FIRST to set environment variables before any torch imports
# isort: off
from .. import torch_init  # noqa: F401

# isort: on

import logging
import os
import warnings
from enum import Enum

# Suppress safe-to-ignore warnings from ML libraries early in import chain
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*clean_up_tokenization_spaces.*"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*pkg_resources is deprecated.*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*GPT2InferenceModel has generative capabilities.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*this function's implementation will be changed.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Some weights of Wav2Vec2ForCTC were not initialized.*",
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*The attention mask is not set.*"
)

# Set specific environment variables to suppress only the identified safe warnings
os.environ.setdefault(
    "TOKENIZERS_PARALLELISM", "false"
)  # Suppress tokenizer parallelism warning

logger = logging.getLogger(__name__)

# CPU limiting for development
if os.environ.get("CAI_ENABLE_CPU_LIMITING", "false").lower() == "true":
    max_threads = int(os.environ.get("CAI_MAX_CPU_THREADS", "2"))
    os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
    os.environ["MKL_NUM_THREADS"] = str(max_threads)
    os.environ["OMP_NUM_THREADS"] = str(max_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)

    # Also limit PyTorch threads
    try:
        import torch

        torch.set_num_threads(max_threads)
    except ImportError:
        pass

    logger.info(f"CPU limiting enabled: {max_threads} threads")

# Default model constants
DEFAULT_COQUI_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


class Environment(Enum):
    """Environment types for configuration."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    DEMO = "demo"
    TESTING = "testing"
