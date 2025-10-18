"""
Security middleware for FastAPI applications.

Provides middleware for authentication, authorization, and security checks.
"""

from typing import Optional

from .manager import DeviceIdentityService
from .types import DeviceIdentity, DeviceRole


class SecurityMiddleware:
    """Security middleware for FastAPI applications."""

    def __init__(self, device_manager: DeviceIdentityService):
        self.device_manager = device_manager

    def get_current_device(
        self, token: Optional[str] = None
    ) -> Optional[DeviceIdentity]:
        """Get current device from JWT token."""
        if not token:
            return None

        payload = self.device_manager.verify_jwt_token(token)
        if not payload:
            return None

        # Update last seen
        self.device_manager.update_last_seen()

        return self.device_manager.get_device_identity()

    def check_permission(
        self, device: DeviceIdentity, required_capability: str
    ) -> bool:
        """Check if device has required capability."""
        return required_capability in device.capabilities

    def check_role(self, device: DeviceIdentity, required_role: DeviceRole) -> bool:
        """Check if device has required role or higher."""
        role_hierarchy = {
            DeviceRole.GUEST: 0,
            DeviceRole.USER: 1,
            DeviceRole.ADMIN: 2,
            DeviceRole.SERVICE: 3,
        }

        device_level = role_hierarchy.get(device.role, 0)
        required_level = role_hierarchy.get(required_role, 0)

        return device_level >= required_level
