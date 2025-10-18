"""
Character API package.

Provides modular character interaction endpoints organized by functionality.
"""

from .auth import auth_router
from .config import config_router
from .health import health_router
from .interaction import interaction_router
from .session import session_router

__all__ = [
    "auth_router",
    "config_router",
    "health_router",
    "interaction_router",
    "session_router",
]
