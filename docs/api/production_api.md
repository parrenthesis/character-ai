# Production API Documentation

## Overview

The Character AI provides a comprehensive REST API for production deployment with enterprise-grade security, monitoring, and scalability.

## Base URL

```
https://your-domain.com/api/v1
```

## Authentication

All API endpoints require JWT authentication. Include the token in the Authorization header:

```bash
Authorization: Bearer <your-jwt-token>
```

### Getting a Token

```bash
# Register device and get token
curl -X POST https://your-domain.com/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "device_name": "production-server",
    "capabilities": ["character_management", "voice_processing"]
  }'
```

## Core Endpoints

### Health and Status

#### GET /health
Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

#### GET /health/detailed
Detailed health check with component status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "database": "healthy",
    "redis": "healthy",
    "models": "healthy",
    "storage": "healthy"
  },
  "metrics": {
    "memory_usage_mb": 2048,
    "cpu_usage_percent": 45.2,
    "active_connections": 150
  }
}
```

### Character Management

#### GET /characters
List all characters with pagination and filtering.

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `limit` (int): Items per page (default: 20)
- `species` (string): Filter by species
- `archetype` (string): Filter by archetype
- `search` (string): Search by name or description

**Response:**
```json
{
  "characters": [
    {
      "id": "char_123",
      "name": "FriendlyBot",
      "species": "robot",
      "archetype": "helper",
      "personality_traits": ["friendly", "helpful"],
      "abilities": ["flight", "translation"],
      "topics": ["science", "technology"],
      "backstory": "A helpful robot assistant",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:30:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 150,
    "pages": 8
  }
}
```

#### POST /characters
Create a new character.

**Request Body:**
```json
{
  "name": "NewCharacter",
  "species": "robot",
  "archetype": "helper",
  "personality_traits": ["friendly"],
  "abilities": ["flight"],
  "topics": ["science"],
  "backstory": "A new character",
  "goals": ["help humans"],
  "fears": ["rust"],
  "likes": ["oil"],
  "dislikes": ["water"]
}
```

**Response:**
```json
{
  "id": "char_456",
  "name": "NewCharacter",
  "species": "robot",
  "archetype": "helper",
  "personality_traits": ["friendly"],
  "abilities": ["flight"],
  "topics": ["science"],
  "backstory": "A new character",
  "goals": ["help humans"],
  "fears": ["rust"],
  "likes": ["oil"],
  "dislikes": ["water"],
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

#### GET /characters/{character_id}
Get a specific character by ID.

**Response:**
```json
{
  "id": "char_123",
  "name": "FriendlyBot",
  "species": "robot",
  "archetype": "helper",
  "personality_traits": ["friendly", "helpful"],
  "abilities": ["flight", "translation"],
  "topics": ["science", "technology"],
  "backstory": "A helpful robot assistant",
  "goals": ["help humans"],
  "fears": ["rust"],
  "likes": ["oil"],
  "dislikes": ["water"],
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

#### PUT /characters/{character_id}
Update a character.

**Request Body:**
```json
{
  "name": "UpdatedCharacter",
  "backstory": "Updated backstory"
}
```

**Response:**
```json
{
  "id": "char_123",
  "name": "UpdatedCharacter",
  "species": "robot",
  "archetype": "helper",
  "personality_traits": ["friendly", "helpful"],
  "abilities": ["flight", "translation"],
  "topics": ["science", "technology"],
  "backstory": "Updated backstory",
  "goals": ["help humans"],
  "fears": ["rust"],
  "likes": ["oil"],
  "dislikes": ["water"],
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:35:00Z"
}
```

#### DELETE /characters/{character_id}
Delete a character.

**Response:**
```json
{
  "message": "Character deleted successfully",
  "id": "char_123"
}
```

### Voice Processing

#### POST /voice/process
Process audio input and generate response.

**Request Body:**
```json
{
  "audio_data": "base64_encoded_audio",
  "character_id": "char_123",
  "language": "en",
  "format": "wav"
}
```

**Response:**
```json
{
  "text": "Hello! How can I help you today?",
  "audio_response": "base64_encoded_audio",
  "character_id": "char_123",
  "processing_time_ms": 1250,
  "language": "en",
  "confidence": 0.95
}
```

#### POST /voice/synthesize
Synthesize text to speech.

**Request Body:**
```json
{
  "text": "Hello, world!",
  "character_id": "char_123",
  "voice_style": "friendly",
  "language": "en"
}
```

**Response:**
```json
{
  "audio_data": "base64_encoded_audio",
  "character_id": "char_123",
  "voice_style": "friendly",
  "language": "en",
  "duration_ms": 2000
}
```

### LLM Management

#### GET /llm/models
List available LLM models.

**Response:**
```json
{
  "models": [
    {
      "id": "llama-3.2-3b-instruct",
      "name": "Llama 3.2 3B Instruct",
      "provider": "local",
      "size_gb": 2.1,
      "status": "installed",
      "capabilities": ["text_generation", "conversation"]
    }
  ]
}
```

#### POST /llm/models/install
Install a new LLM model.

**Request Body:**
```json
{
  "model_id": "llama-3.2-7b-instruct",
  "provider": "local"
}
```

**Response:**
```json
{
  "message": "Model installation started",
  "model_id": "llama-3.2-7b-instruct",
  "status": "installing"
}
```

#### GET /llm/models/{model_id}/status
Get installation status of a model.

**Response:**
```json
{
  "model_id": "llama-3.2-7b-instruct",
  "status": "installing",
  "progress_percent": 75,
  "estimated_time_remaining": "5 minutes"
}
```

### Monitoring and Metrics

#### GET /metrics
Get system metrics.

**Response:**
```json
{
  "system": {
    "memory_usage_mb": 2048,
    "cpu_usage_percent": 45.2,
    "disk_usage_percent": 65.8,
    "network_io_mb": 1024
  },
  "application": {
    "active_connections": 150,
    "requests_per_minute": 1200,
    "average_response_time_ms": 250,
    "error_rate_percent": 0.5
  },
  "models": {
    "whisper_processing_time_ms": 800,
    "llm_processing_time_ms": 1200,
    "tts_processing_time_ms": 600
  }
}
```

#### GET /logs
Get application logs with filtering.

**Query Parameters:**
- `level` (string): Log level (debug, info, warning, error)
- `component` (string): Component name
- `start_time` (string): Start time (ISO format)
- `end_time` (string): End time (ISO format)
- `limit` (int): Maximum number of logs

**Response:**
```json
{
  "logs": [
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "level": "info",
      "component": "voice_processor",
      "message": "Audio processing completed",
      "metadata": {
        "duration_ms": 800,
        "character_id": "char_123"
      }
    }
  ],
  "total": 1000,
  "page": 1,
  "limit": 100
}
```

### Configuration

#### GET /config
Get current configuration.

**Response:**
```json
{
  "environment": "production",
  "version": "1.0.0",
  "features": {
    "voice_processing": true,
    "character_management": true,
    "llm_integration": true,
    "parental_controls": true
  },
  "limits": {
    "max_characters": 1000,
    "max_audio_duration_seconds": 30,
    "rate_limit_per_minute": 1000
  }
}
```

#### PUT /config
Update configuration.

**Request Body:**
```json
{
  "limits": {
    "max_characters": 2000,
    "max_audio_duration_seconds": 60
  }
}
```

**Response:**
```json
{
  "message": "Configuration updated successfully",
  "updated_fields": ["limits.max_characters", "limits.max_audio_duration_seconds"]
}
```

## Error Handling

All API endpoints return appropriate HTTP status codes and error messages.

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid character data",
    "details": {
      "field": "name",
      "reason": "Name is required"
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456"
  }
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Default**: 1000 requests per minute per device
- **Burst**: 100 requests per minute
- **Headers**: Rate limit information is included in response headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248600
```

## WebSocket Support

### Real-time Communication

```javascript
// Connect to WebSocket
const ws = new WebSocket('wss://your-domain.com/ws');

// Send audio data
ws.send(JSON.stringify({
  type: 'audio',
  data: audioBlob,
  character_id: 'char_123'
}));

// Receive response
ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  if (response.type === 'audio_response') {
    // Handle audio response
  }
};
```

## SDK Examples

### Python SDK

```python
from cai_sdk import CAIClient

# Initialize client
client = CAIClient(
    base_url="https://your-domain.com/api/v1",
    token="your-jwt-token"
)

# Create character
character = client.characters.create({
    "name": "MyCharacter",
    "species": "robot",
    "archetype": "helper"
})

# Process audio
response = client.voice.process(
    audio_data=audio_bytes,
    character_id=character.id
)
```

### JavaScript SDK

```javascript
import { CAIClient } from '@cai/sdk';

// Initialize client
const client = new CAIClient({
  baseUrl: 'https://your-domain.com/api/v1',
  token: 'your-jwt-token'
});

// Create character
const character = await client.characters.create({
  name: 'MyCharacter',
  species: 'robot',
  archetype: 'helper'
});

// Process audio
const response = await client.voice.process({
  audioData: audioBlob,
  characterId: character.id
});
```

## Production Considerations

### Security
- All endpoints require authentication
- HTTPS is enforced in production
- Rate limiting prevents abuse
- Input validation and sanitization
- CORS configuration for web clients

### Performance
- Response time monitoring
- Connection pooling
- Caching for frequently accessed data
- Load balancing support

### Monitoring
- Health check endpoints
- Metrics collection
- Log aggregation
- Alert configuration

### Scalability
- Horizontal scaling support
- Database connection pooling
- Redis for session management
- CDN for static assets
