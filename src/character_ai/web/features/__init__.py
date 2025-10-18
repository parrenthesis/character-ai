"""
Feature web API endpoints.

Contains API endpoints for optional features like personalization, parental controls,
language support, and multilingual audio.
"""

from .language_api import router as language_router
from .multilingual_audio_api import router as multilingual_audio_router
from .parental_controls_api import router as parental_controls_router

__all__ = [
    "language_router",
    "multilingual_audio_router",
    "parental_controls_router",
]
