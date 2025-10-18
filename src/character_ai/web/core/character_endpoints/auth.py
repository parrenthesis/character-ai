"""
Authentication endpoints.

Handles device registration, token generation, and authentication.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException

from ....features.security import DeviceIdentity
from ....observability import get_logger
from ...security_deps import require_authentication

logger = get_logger(__name__)

# Create router
auth_router = APIRouter(prefix="/api/v1/character", tags=["character-auth"])


def get_security_manager() -> Any:
    """Get the security manager instance."""
    from ...security_deps import get_security_manager as _get_security_manager

    return _get_security_manager()


@auth_router.post("/auth/register")
async def register_device(device_name: Optional[str] = None) -> Dict[str, Any]:
    """Register a new device and get authentication token."""
    try:
        security_manager = get_security_manager()
        await security_manager.initialize()

        device = security_manager.get_device_identity()
        if not device:
            raise HTTPException(
                status_code=500, detail="Failed to create device identity"
            )

        # Generate JWT token
        token = security_manager.generate_jwt_token()

        return {
            "device_id": device.device_id,
            "device_name": device.device_name,
            "access_token": token,
            "token_type": "bearer",
            "expires_in": security_manager.config.jwt_expiry_seconds,
            "role": device.role.value,
            "capabilities": device.capabilities,
        }
    except Exception as e:
        logger.error(f"Failed to register device: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@auth_router.get("/auth/me")
async def get_current_device_info(
    device: DeviceIdentity = Depends(require_authentication),
) -> Dict[str, Any]:
    """Get current device information."""
    return {
        "device_id": device.device_id,
        "device_name": device.device_name,
        "role": device.role.value,
        "capabilities": device.capabilities,
        "created_at": device.created_at,
        "last_seen": device.last_seen,
        "metadata": device.metadata,
    }


@auth_router.post("/auth/token")
async def generate_token(
    device: DeviceIdentity = Depends(require_authentication),
    additional_claims: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a new JWT token for the authenticated device."""
    try:
        security_manager = get_security_manager()
        token = security_manager.generate_jwt_token(additional_claims)

        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": security_manager.config.jwt_expiry_seconds,
            "device_id": device.device_id,
        }
    except Exception as e:
        logger.error(f"Failed to generate token: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@auth_router.get("/auth/public-key")
async def get_public_key() -> Dict[str, Any]:
    """Get the device's public key for signature verification."""
    try:
        security_manager = get_security_manager()
        public_key = security_manager.get_public_key_pem()

        if not public_key:
            raise HTTPException(status_code=404, detail="Public key not available")

        return {"public_key": public_key, "algorithm": "RSA", "key_size": 2048}
    except Exception as e:
        logger.error(f"Failed to get public key: {e}")
        raise HTTPException(status_code=500, detail=str(e))
