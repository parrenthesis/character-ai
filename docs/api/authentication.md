# Authentication Guide

## Overview

The Character AI API supports multiple authentication methods for different use cases.

## Authentication Methods

### 1. Device Registration

Register a new device to receive authentication credentials:

```bash
POST /api/v1/character/auth/register
```

**Response:**
```json
{
  "device_id": "device-uuid-123",
  "device_name": "toy-device-456",
  "access_token": "jwt-token-here",
  "token_type": "bearer",
  "expires_in": 3600,
  "role": "user",
  "capabilities": ["read", "write", "interact"]
}
```

### 2. JWT Token Authentication

Use the Bearer token in the Authorization header:

```bash
curl -H "Authorization: Bearer your-jwt-token" \
     https://api.example.com/api/v1/character/characters
```

### 3. Admin Token Authentication

For administrative operations, use the x-admin-token header:

```bash
curl -H "x-admin-token: your-admin-token" \
     https://api.example.com/api/v1/character/memory/clear
```

## Device Roles

- **GUEST**: Read-only access to public information
- **USER**: Full interaction capabilities
- **ADMIN**: Administrative operations
- **SERVICE**: System service operations

## Security Features

- **Rate Limiting**: 60 requests/minute per device
- **JWT Expiry**: Tokens expire after 1 hour by default
- **Device Identity**: Each device has a unique identity
- **Cryptographic Signing**: Data integrity verification
- **HTTPS Required**: Production deployments require HTTPS

## Error Responses

### 401 Unauthorized
```json
{
  "detail": "Authentication required"
}
```

### 403 Forbidden
```json
{
  "detail": "Insufficient permissions"
}
```

### 429 Too Many Requests
```json
{
  "detail": "Rate limit exceeded"
}
```

## Best Practices

1. **Store tokens securely** - Use secure storage for JWT tokens
2. **Handle token expiry** - Implement token refresh logic
3. **Respect rate limits** - Implement exponential backoff
4. **Use HTTPS** - Always use HTTPS in production
5. **Validate responses** - Check response status codes
