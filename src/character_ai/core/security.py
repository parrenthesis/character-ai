"""
Security and authentication module for the Character AI.

Provides device identity, JWT authentication, and security utilities.
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import jwt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.dsa import DSAPrivateKey, DSAPublicKey
from cryptography.hazmat.primitives.asymmetric.ec import (
    ECDSA,
    EllipticCurvePrivateKey,
    EllipticCurvePublicKey,
)
from cryptography.hazmat.primitives.asymmetric.ed448 import (
    Ed448PrivateKey,
    Ed448PublicKey,
)
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

logger = logging.getLogger(__name__)


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
        default_factory=lambda: Path("configs/device_public.pem")
    )
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst: int = 10
    enable_device_registration: bool = True
    require_https: bool = False  # Set to True in production

    def __post_init__(self) -> None:
        """Validate security configuration."""
        if not self.jwt_secret:
            raise ValueError("CAI_JWT_SECRET environment variable must be set")
        if not self.private_key_file.exists():
            logger.warning(f"Private key file not found: {self.private_key_file}")
            # Generate a new key if it doesn't exist
            self._generate_private_key()

    def _generate_private_key(self) -> None:
        """Generate a new private key if it doesn't exist."""
        try:
            # Create configs directory if it doesn't exist
            self.private_key_file.parent.mkdir(parents=True, exist_ok=True)

            # Generate private key
            private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

            # Save private key
            with open(self.private_key_file, "wb") as f:
                f.write(
                    private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption(),
                    )
                )

            # Generate and save public key
            public_key = private_key.public_key()
            with open(self.public_key_file, "wb") as f:
                f.write(
                    public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo,
                    )
                )

            logger.info(f"Generated new private key: {self.private_key_file}")
            logger.info(f"Generated new public key: {self.public_key_file}")

        except Exception as e:
            logger.error(f"Failed to generate private key: {e}")
            raise


class DeviceIdentityManager:
    """Manages device identity and authentication."""

    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self._device_identity: Optional[DeviceIdentity] = None
        self._private_key: Optional[bytes] = None
        self._public_key: Optional[bytes] = None
        self._rate_limit_tokens: Dict[str, float] = {}
        self._rate_limit_requests: Dict[str, List[float]] = {}

    async def initialize(self) -> None:
        """Initialize the device identity manager."""
        try:
            # Load or create device identity
            await self._load_or_create_device_identity()

            # Load or generate cryptographic keys
            await self._load_or_generate_keys()

            if self._device_identity is not None:
                logger.info(
                    f"Device identity manager initialized for device: {self._device_identity.device_id}"
                )
        except Exception as e:
            logger.error(f"Failed to initialize device identity manager: {e}")
            raise

    async def _load_or_create_device_identity(self) -> None:
        """Load existing device identity or create a new one."""
        if self.config.device_id_file.exists():
            try:
                with open(self.config.device_id_file, "r") as f:
                    data = json.load(f)
                self._device_identity = DeviceIdentity.from_dict(data)
                logger.info(
                    f"Loaded existing device identity: {self._device_identity.device_id}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load device identity: {e}. Creating new one."
                )
                await self._create_new_device_identity()
        else:
            await self._create_new_device_identity()

    async def _create_new_device_identity(self) -> None:
        """Create a new device identity."""
        device_id = str(uuid.uuid4())
        device_name = f"toy-{device_id[:8]}"

        self._device_identity = DeviceIdentity(
            device_id=device_id,
            device_name=device_name,
            role=DeviceRole.USER,
            capabilities=["read", "write", "interact"],
        )

        # Save to file
        self.config.device_id_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.device_id_file, "w") as f:
            json.dump(self._device_identity.to_dict(), f, indent=2)

        logger.info(f"Created new device identity: {device_id}")

    async def _load_or_generate_keys(self) -> None:
        """Load existing cryptographic keys or generate new ones."""
        if (
            self.config.private_key_file.exists()
            and self.config.public_key_file.exists()
        ):
            try:
                with open(self.config.private_key_file, "rb") as f:
                    self._private_key = f.read()
                with open(self.config.public_key_file, "rb") as f:
                    self._public_key = f.read()
                logger.info("Loaded existing cryptographic keys")
            except Exception as e:
                logger.warning(f"Failed to load keys: {e}. Generating new ones.")
                await self._generate_new_keys()
        else:
            await self._generate_new_keys()

    async def _generate_new_keys(self) -> None:
        """Generate new RSA key pair."""
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        # Serialize private key
        self._private_key = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        # Serialize public key
        public_key = private_key.public_key()
        self._public_key = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        # Save keys to files
        self.config.private_key_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.private_key_file, "wb") as f:
            f.write(self._private_key)
        with open(self.config.public_key_file, "wb") as f:
            f.write(self._public_key)

        logger.info("Generated new cryptographic keys")

    def generate_jwt_token(
        self, additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a JWT token for the device."""
        if not self._device_identity:
            raise ValueError("Device identity not initialized")

        now = time.time()
        payload = {
            "device_id": self._device_identity.device_id,
            "device_name": self._device_identity.device_name,
            "role": self._device_identity.role.value,
            "capabilities": self._device_identity.capabilities,
            "iat": now,
            "exp": now + self.config.jwt_expiry_seconds,
            "iss": "character.ai",
            "sub": self._device_identity.device_id,
        }

        if additional_claims:
            payload.update(additional_claims)

        token = jwt.encode(
            payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm
        )
        return token

    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(
                token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm]
            )
            return dict(payload)
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None

    def check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        now = time.time()

        # Clean old requests
        if client_id in self._rate_limit_requests:
            self._rate_limit_requests[client_id] = [
                req_time
                for req_time in self._rate_limit_requests[client_id]
                if now - req_time < 60  # Keep only last minute
            ]
        else:
            self._rate_limit_requests[client_id] = []

        # Check rate limit
        if (
            len(self._rate_limit_requests[client_id])
            >= self.config.rate_limit_requests_per_minute
        ):
            return False

        # Add current request
        self._rate_limit_requests[client_id].append(now)
        return True

    def get_device_identity(self) -> Optional[DeviceIdentity]:
        """Get the current device identity."""
        return self._device_identity

    def update_last_seen(self) -> None:
        """Update the last seen timestamp."""
        if self._device_identity:
            self._device_identity.last_seen = time.time()
            # Save updated identity
            with open(self.config.device_id_file, "w") as f:
                json.dump(self._device_identity.to_dict(), f, indent=2)

    def get_public_key_pem(self) -> Optional[str]:
        """Get the public key in PEM format."""
        if self._public_key:
            return self._public_key.decode("utf-8")
        return None

    def sign_data(self, data: bytes) -> Optional[bytes]:
        """Sign data with the device's private key."""
        if not self._private_key:
            return None

        try:
            private_key = serialization.load_pem_private_key(
                self._private_key, password=None
            )

            # Handle different key types
            if isinstance(private_key, (Ed25519PrivateKey, Ed448PrivateKey)):
                # Ed25519/Ed448 don't use padding or algorithm parameters
                signature = private_key.sign(data)
            elif isinstance(private_key, DSAPrivateKey):
                # DSA uses algorithm but not padding
                signature = private_key.sign(data, algorithm=hashes.SHA256())
            elif isinstance(private_key, EllipticCurvePrivateKey):
                # EC uses algorithm but not padding
                signature = private_key.sign(
                    data, signature_algorithm=ECDSA(hashes.SHA256())
                )
            elif hasattr(private_key, "sign"):
                # RSA uses both padding and algorithm
                signature = private_key.sign(
                    data,
                    padding=padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH,
                    ),
                    algorithm=hashes.SHA256(),
                )
            else:
                logger.error("Private key type not supported for signing")
                return None
            return signature
        except Exception as e:
            logger.error(f"Failed to sign data: {e}")
            return None

    def verify_signature(self, data: bytes, signature: bytes) -> bool:
        """Verify a signature using the device's public key."""
        if not self._public_key:
            return False

        try:
            public_key = serialization.load_pem_public_key(self._public_key)

            # Handle different key types
            if isinstance(public_key, (Ed25519PublicKey, Ed448PublicKey)):
                # Ed25519/Ed448 don't use padding or algorithm parameters
                public_key.verify(signature, data)
            elif isinstance(public_key, DSAPublicKey):
                # DSA uses algorithm but not padding
                public_key.verify(signature, data, algorithm=hashes.SHA256())
            elif isinstance(public_key, EllipticCurvePublicKey):
                # EC uses algorithm but not padding
                public_key.verify(
                    signature, data, signature_algorithm=ECDSA(hashes.SHA256())
                )
            elif hasattr(public_key, "verify"):
                # RSA uses both padding and algorithm
                public_key.verify(
                    signature,
                    data,
                    padding=padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH,
                    ),
                    algorithm=hashes.SHA256(),
                )
            else:
                logger.error("Public key type not supported for verification")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to verify signature: {e}")
            return False


class SecurityMiddleware:
    """Security middleware for FastAPI applications."""

    def __init__(self, device_manager: DeviceIdentityManager):
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
