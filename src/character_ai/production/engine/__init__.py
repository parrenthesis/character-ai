"""
Real-time interaction engine modules.

Components for the real-time character interaction system.
"""

from .audio_processor import AudioProcessor
from .character_manager import CharacterInteractionController
from .core_engine import CoreRealTimeEngine
from .performance_monitor import PerformanceMonitor

__all__ = [
    "AudioProcessor",
    "CharacterInteractionController",
    "CoreRealTimeEngine",
    "PerformanceMonitor",
]
