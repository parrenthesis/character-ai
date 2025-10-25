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

# Note: CPU thread limiting is now handled in torch_init.py (before library imports)
# and engine_lifecycle.py (for PyTorch-specific limits after torch is imported)

# Default model constants
DEFAULT_COQUI_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


class Environment(Enum):
    """Environment types for configuration."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    DEMO = "demo"
    TESTING = "testing"
