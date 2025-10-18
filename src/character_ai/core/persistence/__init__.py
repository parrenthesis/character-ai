"""
Persistence utilities for Character AI Platform.

Provides centralized JSON persistence, data management, and storage utilities.
"""

from .base_manager import BaseDataManager
from .json_manager import JSONRepository

__all__ = [
    "BaseDataManager",
    "JSONRepository",
]
