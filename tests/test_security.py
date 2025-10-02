"""
Comprehensive tests for the security system.

Tests device identity, JWT authentication, RBAC, and rate limiting.
"""

import tempfile
import time
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from character_ai.core.security import (
    DeviceIdentity,
    DeviceIdentityManager,
    DeviceRole,
    SecurityConfig,
    SecurityMiddleware,
)
from character_ai.web.security_deps import require_authentication
from character_ai.web.toy_api import app


class TestDeviceIdentity:
    """Test DeviceIdentity class."""

    def test_device_identity_creation(self) -> None:
        """Test creating a device identity."""
        device = DeviceIdentity(
            device_id="test-device-123",
            device_name="Test Device",
            role=DeviceRole.USER,
            capabilities=["read", "write"],
        )

        assert device.device_id == "test-device-123"
        assert device.device_name == "Test Device"
        assert device.role == DeviceRole.USER
        assert device.capabilities == ["read", "write"]
        assert device.created_at > 0
        assert device.last_seen > 0

    def test_device_identity_serialization(self) -> None:
        """Test device identity serialization."""
        device = DeviceIdentity(
            device_id="test-device-123",
            device_name="Test Device",
            role=DeviceRole.ADMIN,
            capabilities=["read", "write", "admin"],
        )

        # Test to_dict
        data = device.to_dict()
        assert data["device_id"] == "test-device-123"
        assert data["role"] == "admin"
        assert data["capabilities"] == ["read", "write", "admin"]

        # Test from_dict
        new_device = DeviceIdentity.from_dict(data)
        assert new_device.device_id == device.device_id
        assert new_device.role == device.role
        assert new_device.capabilities == device.capabilities


class TestDeviceIdentityManager:
    """Test DeviceIdentityManager class."""

    @pytest.fixture
    def temp_config(self) -> Generator[SecurityConfig, None, None]:
        """Create a temporary config for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SecurityConfig(
                device_id_file=Path(temp_dir) / "device_id.json",
                private_key_file=Path(temp_dir) / "private.pem",
                public_key_file=Path(temp_dir) / "public.pem",
                jwt_secret="test-secret-key",
            )
            yield config

    @pytest.fixture
    def device_manager(self, temp_config: SecurityConfig) -> DeviceIdentityManager:
        """Create a device manager for testing."""
        return DeviceIdentityManager(temp_config)

    @pytest.mark.asyncio
    async def test_initialize_new_device(
        self, device_manager: DeviceIdentityManager
    ) -> None:
        """Test initializing a new device."""
        await device_manager.initialize()

        device = device_manager.get_device_identity()
        assert device is not None
        assert device.device_id is not None
        assert device.device_name.startswith("toy-")
        assert device.role == DeviceRole.USER
        assert "read" in device.capabilities
        assert "write" in device.capabilities

    @pytest.mark.asyncio
    async def test_jwt_token_generation(
        self, device_manager: DeviceIdentityManager
    ) -> None:
        """Test JWT token generation and verification."""
        await device_manager.initialize()

        # Generate token
        token = device_manager.generate_jwt_token()
        assert token is not None
        assert isinstance(token, str)

        # Verify token
        payload = device_manager.verify_jwt_token(token)
        assert payload is not None
        device_identity = device_manager.get_device_identity()
        assert device_identity is not None
        assert payload["device_id"] == device_identity.device_id
        assert payload["role"] == "user"
        assert "iat" in payload
        assert "exp" in payload

    @pytest.mark.asyncio
    async def test_jwt_token_expiry(
        self, device_manager: DeviceIdentityManager
    ) -> None:
        """Test JWT token expiry."""
        await device_manager.initialize()

        # Generate token with short expiry
        device_manager.config.jwt_expiry_seconds = 1
        token = device_manager.generate_jwt_token()

        # Should be valid immediately
        payload = device_manager.verify_jwt_token(token)
        assert payload is not None

        # Wait for expiry
        time.sleep(2)

        # Should be expired
        payload = device_manager.verify_jwt_token(token)
        assert payload is None

    @pytest.mark.asyncio
    async def test_rate_limiting(self, device_manager: DeviceIdentityManager) -> None:
        """Test rate limiting functionality."""
        await device_manager.initialize()

        client_id = "test-client"

        # Should allow requests within limit
        for i in range(10):
            assert device_manager.check_rate_limit(client_id) is True

        # Should block when over limit
        device_manager.config.rate_limit_requests_per_minute = 5
        device_manager._rate_limit_requests[client_id] = []

        for i in range(5):
            assert device_manager.check_rate_limit(client_id) is True

        # This should be blocked
        assert device_manager.check_rate_limit(client_id) is False

    @pytest.mark.asyncio
    async def test_cryptographic_keys(
        self, device_manager: DeviceIdentityManager
    ) -> None:
        """Test cryptographic key generation and usage."""
        await device_manager.initialize()

        # Test public key retrieval
        public_key = device_manager.get_public_key_pem()
        assert public_key is not None
        assert "BEGIN PUBLIC KEY" in public_key

        # Test data signing
        test_data = b"test data to sign"
        signature = device_manager.sign_data(test_data)
        assert signature is not None
        assert isinstance(signature, bytes)

        # Test signature verification
        assert device_manager.verify_signature(test_data, signature) is True

        # Test with wrong data
        wrong_data = b"wrong data"
        assert device_manager.verify_signature(wrong_data, signature) is False


class TestSecurityMiddleware:
    """Test SecurityMiddleware class."""

    @pytest.fixture
    def mock_device_manager(self) -> Mock:
        """Create a mock device manager."""
        manager = Mock(spec=DeviceIdentityManager)
        manager.verify_jwt_token.return_value = {
            "device_id": "test-device",
            "role": "user",
            "capabilities": ["read", "write"],
        }
        manager.get_device_identity.return_value = DeviceIdentity(
            device_id="test-device",
            device_name="Test Device",
            role=DeviceRole.USER,
            capabilities=["read", "write"],
        )
        return manager

    @pytest.fixture
    def security_middleware(self, mock_device_manager: Mock) -> SecurityMiddleware:
        """Create security middleware with mock device manager."""
        return SecurityMiddleware(mock_device_manager)

    def test_get_current_device_valid_token(
        self, security_middleware: SecurityMiddleware, mock_device_manager: Mock
    ) -> None:
        """Test getting current device with valid token."""
        device = security_middleware.get_current_device("valid-token")

        assert device is not None
        assert device.device_id == "test-device"
        assert device.role == DeviceRole.USER
        mock_device_manager.verify_jwt_token.assert_called_once_with("valid-token")

    def test_get_current_device_invalid_token(
        self, security_middleware: SecurityMiddleware, mock_device_manager: Mock
    ) -> None:
        """Test getting current device with invalid token."""
        mock_device_manager.verify_jwt_token.return_value = None

        device = security_middleware.get_current_device("invalid-token")

        assert device is None
        mock_device_manager.verify_jwt_token.assert_called_once_with("invalid-token")

    def test_check_permission(self, security_middleware: SecurityMiddleware) -> None:
        """Test permission checking."""
        device = DeviceIdentity(
            device_id="test-device",
            device_name="Test Device",
            role=DeviceRole.USER,
            capabilities=["read", "write"],
        )

        # Should have required capability
        assert security_middleware.check_permission(device, "read") is True
        assert security_middleware.check_permission(device, "write") is True

        # Should not have required capability
        assert security_middleware.check_permission(device, "admin") is False

    def test_check_role(self, security_middleware: SecurityMiddleware) -> None:
        """Test role checking."""
        user_device = DeviceIdentity(
            device_id="user-device",
            device_name="User Device",
            role=DeviceRole.USER,
            capabilities=["read", "write"],
        )

        admin_device = DeviceIdentity(
            device_id="admin-device",
            device_name="Admin Device",
            role=DeviceRole.ADMIN,
            capabilities=["read", "write", "admin"],
        )

        # User should have user role
        assert security_middleware.check_role(user_device, DeviceRole.USER) is True
        assert security_middleware.check_role(user_device, DeviceRole.GUEST) is True

        # User should not have admin role
        assert security_middleware.check_role(user_device, DeviceRole.ADMIN) is False

        # Admin should have all roles
        assert security_middleware.check_role(admin_device, DeviceRole.USER) is True
        assert security_middleware.check_role(admin_device, DeviceRole.ADMIN) is True


class TestSecurityAPI:
    """Test security-related API endpoints."""

    @pytest.fixture
    def mock_security_manager(self) -> AsyncMock:
        """Create a mock security manager."""
        manager = AsyncMock(spec=DeviceIdentityManager)
        manager.initialize.return_value = None
        manager.get_device_identity.return_value = DeviceIdentity(
            device_id="test-device-123",
            device_name="Test Device",
            role=DeviceRole.USER,
            capabilities=["read", "write"],
        )
        manager.generate_jwt_token.return_value = "test-jwt-token"
        # Create a mock config object
        manager.config = Mock()
        manager.config.jwt_expiry_seconds = 3600
        manager.get_public_key_pem.return_value = (
            "-----BEGIN PUBLIC KEY-----\ntest-key\n-----END PUBLIC KEY-----"
        )
        return manager

    def test_register_device(self, mock_security_manager: AsyncMock) -> None:
        """Test device registration endpoint."""
        with patch(
            "character_ai.web.toy_api.get_security_manager",
            return_value=mock_security_manager,
        ):
            with TestClient(app) as client:
                response = client.post("/api/v1/toy/auth/register")

                assert response.status_code == 200
                data = response.json()
                assert "device_id" in data
                assert "access_token" in data
                assert "token_type" in data
                assert data["token_type"] == "bearer"
                assert data["role"] == "user"

    def test_get_current_device_info_authenticated(
        self, mock_security_manager: AsyncMock
    ) -> None:
        """Test getting current device info when authenticated."""
        # Mock the authentication dependency
        mock_device = DeviceIdentity(
            device_id="test-device-123",
            device_name="Test Device",
            role=DeviceRole.USER,
            capabilities=["read", "write"],
        )

        # Override the dependency
        async def mock_require_authentication() -> DeviceIdentity:
            return mock_device

        app.dependency_overrides[require_authentication] = mock_require_authentication

        try:
            with TestClient(app) as client:
                response = client.get("/api/v1/toy/auth/me")

                assert response.status_code == 200
                data = response.json()
                assert data["device_id"] == "test-device-123"
                assert data["role"] == "user"
                assert "read" in data["capabilities"]
        finally:
            app.dependency_overrides.clear()

    def test_get_current_device_info_unauthenticated(self) -> None:
        """Test getting current device info when not authenticated."""

        # Override the dependency to raise an authentication error
        async def mock_require_authentication() -> None:
            from fastapi import HTTPException, status

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        app.dependency_overrides[require_authentication] = mock_require_authentication

        try:
            with TestClient(app) as client:
                response = client.get("/api/v1/toy/auth/me")

                assert response.status_code == 401  # Unauthorized
        finally:
            app.dependency_overrides.clear()

    def test_generate_token(self, mock_security_manager: AsyncMock) -> None:
        """Test token generation endpoint."""
        mock_device = DeviceIdentity(
            device_id="test-device-123",
            device_name="Test Device",
            role=DeviceRole.USER,
            capabilities=["read", "write"],
        )

        # Override the dependency
        async def mock_require_authentication() -> DeviceIdentity:
            return mock_device

        app.dependency_overrides[require_authentication] = mock_require_authentication

        try:
            with patch(
                "character_ai.web.toy_api.get_security_manager",
                return_value=mock_security_manager,
            ):
                with TestClient(app) as client:
                    response = client.post("/api/v1/toy/auth/token")

                    assert response.status_code == 200
                    data = response.json()
                    assert "access_token" in data
                    assert data["token_type"] == "bearer"
                    assert data["device_id"] == "test-device-123"
        finally:
            app.dependency_overrides.clear()

    def test_get_public_key(self, mock_security_manager: AsyncMock) -> None:
        """Test getting public key endpoint."""
        with patch(
            "character_ai.web.toy_api.get_security_manager",
            return_value=mock_security_manager,
        ):
            with TestClient(app) as client:
                response = client.get("/api/v1/toy/auth/public-key")

                assert response.status_code == 200
                data = response.json()
                assert "public_key" in data
                assert data["algorithm"] == "RSA"
                assert data["key_size"] == 2048


class TestSecurityIntegration:
    """Integration tests for security system."""

    @pytest.mark.asyncio
    async def test_full_authentication_flow(self) -> None:
        """Test the complete authentication flow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SecurityConfig(
                device_id_file=Path(temp_dir) / "device_id.json",
                private_key_file=Path(temp_dir) / "private.pem",
                public_key_file=Path(temp_dir) / "public.pem",
                jwt_secret="test-secret-key",
            )

            # Initialize device manager
            device_manager = DeviceIdentityManager(config)
            await device_manager.initialize()

            # Generate token
            token = device_manager.generate_jwt_token()
            assert token is not None

            # Verify token
            payload = device_manager.verify_jwt_token(token)
            assert payload is not None
            device_identity = device_manager.get_device_identity()
            assert device_identity is not None
            assert payload["device_id"] == device_identity.device_id

            # Test rate limiting
            assert device_manager.check_rate_limit("test-client") is True

            # Test cryptographic operations
            test_data = b"test data"
            signature = device_manager.sign_data(test_data)
            assert signature is not None
            assert device_manager.verify_signature(test_data, signature) is True
