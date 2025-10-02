"""
FastAPI security dependencies for authentication and authorization.

Provides decorators and dependency injection for JWT authentication,
RBAC, and rate limiting.
"""

import logging
from typing import Any, Callable, Dict, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..core.config import Config
from ..core.security import (
    DeviceIdentity,
    DeviceIdentityManager,
    DeviceRole,
    SecurityMiddleware,
)

logger = logging.getLogger(__name__)

# Global security manager instance
_security_manager: Optional[DeviceIdentityManager] = None
_security_middleware: Optional[SecurityMiddleware] = None


def get_security_manager() -> DeviceIdentityManager:
    """Get the global security manager instance."""
    global _security_manager
    if _security_manager is None:
        config = Config()
        _security_manager = DeviceIdentityManager(config.security)  # type: ignore
    return _security_manager


def get_security_middleware() -> SecurityMiddleware:
    """Get the global security middleware instance."""
    global _security_middleware
    if _security_middleware is None:
        _security_middleware = SecurityMiddleware(get_security_manager())
    return _security_middleware


# HTTP Bearer token scheme
security_scheme = HTTPBearer(auto_error=False)


async def get_current_device(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> Optional[DeviceIdentity]:
    """Get the current authenticated device from JWT token."""
    if not credentials:
        return None

    security_middleware = get_security_middleware()
    device = security_middleware.get_current_device(credentials.credentials)

    if device:
        # Check rate limiting
        client_id = request.client.host if request.client else "unknown"
        if not security_middleware.device_manager.check_rate_limit(client_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )

    return device


async def require_authentication(
    device: Optional[DeviceIdentity] = Depends(get_current_device),
) -> DeviceIdentity:
    """Require authentication - raises 401 if not authenticated."""
    if not device:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return device


async def require_role(
    required_role: DeviceRole, device: DeviceIdentity = Depends(require_authentication)
) -> DeviceIdentity:
    """Require specific role or higher."""
    security_middleware = get_security_middleware()
    if not security_middleware.check_role(device, required_role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role {required_role.value} or higher required",
        )
    return device


async def require_capability(
    required_capability: str, device: DeviceIdentity = Depends(require_authentication)
) -> DeviceIdentity:
    """Require specific capability."""
    security_middleware = get_security_middleware()
    if not security_middleware.check_permission(device, required_capability):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Capability '{required_capability}' required",
        )
    return device


async def require_admin(
    device: DeviceIdentity = Depends(require_authentication),
) -> DeviceIdentity:
    """Require admin role."""
    return await require_role(DeviceRole.ADMIN, device)


async def require_user_or_admin(
    device: DeviceIdentity = Depends(require_authentication),
) -> DeviceIdentity:
    """Require user role or higher."""
    return await require_role(DeviceRole.USER, device)


# Rate limiting decorator
def rate_limit(requests_per_minute: int = 60) -> Callable:
    """Rate limiting decorator for endpoints."""

    def decorator(func: Any) -> Any:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract request from kwargs if available
            request = kwargs.get("request")
            if request:
                client_id = request.client.host
                security_manager = get_security_manager()
                if not security_manager.check_rate_limit(client_id):
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded",
                    )
            return await func(*args, **kwargs)

        return wrapper

    return decorator


# Security headers middleware
async def add_security_headers(request: Request, call_next: Any) -> Any:
    """Add security headers to responses."""
    response = await call_next(request)

    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Add CORS headers if needed
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"

    response.headers[
        "Access-Control-Allow-Headers"
    ] = "Content-Type, Authorization, x-admin-token"

    return response


# Device registration endpoint dependency
async def get_device_registration_info() -> Dict[str, Any]:
    """Get device registration information."""
    security_manager = get_security_manager()
    device = security_manager.get_device_identity()

    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Device not registered"
        )

    return {
        "device_id": device.device_id,
        "device_name": device.device_name,
        "role": device.role.value,
        "capabilities": device.capabilities,
        "public_key": security_manager.get_public_key_pem(),
        "created_at": device.created_at,
        "last_seen": device.last_seen,
    }


# JWT token generation endpoint dependency
async def generate_device_token(
    device: DeviceIdentity = Depends(require_authentication),
    additional_claims: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a JWT token for the authenticated device."""
    security_manager = get_security_manager()
    token = security_manager.generate_jwt_token(additional_claims)

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": security_manager.config.jwt_expiry_seconds,
        "device_id": device.device_id,
    }
