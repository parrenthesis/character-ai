"""
Core web API endpoints.

Contains the main API endpoints for health, streaming, and toy interactions.
"""

from .character_api import app as character_api_app
from .health_api import health_router
from .streaming_api import streaming_router

__all__ = [
    "health_router",
    "streaming_router",
    "character_api_app",
]
