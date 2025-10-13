"""
Centralized PyTorch environment variable initialization.

This module MUST be imported before any other modules that import torch
to ensure proper environment variable setup for PyTorch 2.8 compatibility.
"""

import os

# Set environment variable to fix XTTS v2 compatibility with PyTorch 2.8
# This MUST be set before torch is imported anywhere in the codebase
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# Additional PyTorch environment variables for stability
os.environ.setdefault(
    "TOKENIZERS_PARALLELISM", "false"
)  # Suppress tokenizer parallelism warning
os.environ.setdefault(
    "PYTHONWARNINGS",
    "ignore::RuntimeWarning:sys:1,ignore::RuntimeWarning:unittest.mock,ignore::RuntimeWarning:tracemalloc",
)

# Log that torch environment has been initialized
import logging  # noqa: E402 - Must import after setting env vars

logger = logging.getLogger(__name__)
logger.debug("PyTorch environment variables initialized")
