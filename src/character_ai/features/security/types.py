"""
Security types and data structures.

Contains enums, dataclasses, and type definitions for security and authentication.
"""

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List


class DeviceRole(Enum):
    """Device roles for RBAC."""

    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    SERVICE = "service"


@dataclass
class DeviceIdentity:
    """Device identity information."""

    device_id: str
    device_name: str
    role: DeviceRole
    capabilities: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "role": self.role.value,
            "capabilities": self.capabilities,
            "created_at": self.created_at,
            "last_seen": self.last_seen,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceIdentity":
        """Create from dictionary."""
        return cls(
            device_id=data["device_id"],
            device_name=data["device_name"],
            role=DeviceRole(data["role"]),
            capabilities=data.get("capabilities", []),
            created_at=data.get("created_at", time.time()),
            last_seen=data.get("last_seen", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SecurityConfig:
    """Security configuration."""

    jwt_secret: str = field(
        default_factory=lambda: os.environ.get("CAI_JWT_SECRET", "")
    )
    jwt_algorithm: str = "HS256"
    jwt_expiry_seconds: int = 3600  # 1 hour
    device_id_file: Path = field(default_factory=lambda: Path("configs/device_id.json"))

    private_key_file: Path = field(
        default_factory=lambda: Path(
            os.environ.get("CAI_PRIVATE_KEY_FILE", "configs/device_private.pem")
        )
    )
    public_key_file: Path = field(
        default_factory=lambda: Path(
            os.environ.get("CAI_PUBLIC_KEY_FILE", "configs/device_public.pem")
        )
    )

    # Key generation settings
    key_size: int = 2048
    key_algorithm: str = "RSA"

    # Security settings
    require_authentication: bool = True
    allow_guest_access: bool = False
    session_timeout_seconds: int = 3600

    # Rate limiting
    max_requests_per_minute: int = 100
    max_requests_per_hour: int = 1000

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.jwt_secret and self.require_authentication:
            raise ValueError("JWT secret is required when authentication is enabled")

        if self.jwt_expiry_seconds <= 0:
            raise ValueError("JWT expiry must be positive")

        if self.key_size < 1024:
            raise ValueError("Key size must be at least 1024 bits")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "jwt_algorithm": self.jwt_algorithm,
            "jwt_expiry_seconds": self.jwt_expiry_seconds,
            "device_id_file": str(self.device_id_file),
            "private_key_file": str(self.private_key_file),
            "public_key_file": str(self.public_key_file),
            "key_size": self.key_size,
            "key_algorithm": self.key_algorithm,
            "require_authentication": self.require_authentication,
            "allow_guest_access": self.allow_guest_access,
            "session_timeout_seconds": self.session_timeout_seconds,
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_requests_per_hour": self.max_requests_per_hour,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecurityConfig":
        """Create from dictionary."""
        return cls(
            jwt_secret=data.get("jwt_secret", ""),
            jwt_algorithm=data.get("jwt_algorithm", "HS256"),
            jwt_expiry_seconds=data.get("jwt_expiry_seconds", 3600),
            device_id_file=Path(data.get("device_id_file", "configs/device_id.json")),
            private_key_file=Path(
                data.get("private_key_file", "configs/device_private.pem")
            ),
            public_key_file=Path(
                data.get("public_key_file", "configs/device_public.pem")
            ),
            key_size=data.get("key_size", 2048),
            key_algorithm=data.get("key_algorithm", "RSA"),
            require_authentication=data.get("require_authentication", True),
            allow_guest_access=data.get("allow_guest_access", False),
            session_timeout_seconds=data.get("session_timeout_seconds", 3600),
            max_requests_per_minute=data.get("max_requests_per_minute", 100),
            max_requests_per_hour=data.get("max_requests_per_hour", 1000),
        )
