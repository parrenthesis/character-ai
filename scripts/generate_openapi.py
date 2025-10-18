"""
Generate OpenAPI documentation for the Character AI API.

This script generates comprehensive API documentation including all endpoints,
authentication, request/response schemas, and examples.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import toml
import yaml
from fastapi.openapi.utils import get_openapi

# Import the main app
sys.path.append(str(Path(__file__).parent.parent / "src"))

from character_ai.web.core.character_api import app  # noqa: E402


def get_project_version() -> str:
    """Extract version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "r") as f:
        pyproject_data = toml.load(f)
    return pyproject_data["tool"]["poetry"]["version"]


def generate_openapi_spec() -> Dict[str, Any]:
    """Generate the OpenAPI specification for the API."""

    # Get version from pyproject.toml
    version = get_project_version()

    # Custom OpenAPI configuration
    openapi_schema = get_openapi(
        title="Character AI API",
        version=version,
        description="""
# Character AI API

A comprehensive API for managing interactive AI characters in toys and devices.

## Features

- **Character Management**: Create, configure, and manage AI characters
- **Real-time Interaction**: Process audio input and generate character responses
- **Session Memory**: Maintain conversation context with configurable limits
- **Safety & Security**: Content filtering and safety analysis
- **Voice Management**: Handle voice artifacts and embeddings
- **Authentication**: Device identity and JWT-based authentication

## Authentication

The API supports multiple authentication methods:

1. **Device Registration**: Register a new device and receive authentication tokens
2. **JWT Tokens**: Use Bearer tokens for authenticated requests
3. **Admin Tokens**: Special tokens for administrative operations

## Rate Limiting

- Default: 60 requests per minute per device
- Burst allowance: 10 requests
- Rate limit headers are included in responses

## Safety & Privacy

- All content is analyzed for toxicity and PII
- Child-safe content filtering
- Configurable safety thresholds
- Privacy-first design with on-device processing

## Character Profiles

Characters are defined using YAML profiles with:
- Personality traits and conversation topics
- Voice characteristics and TTS settings
- Safety rules and content guidelines
- Custom prompt templates

## Session Memory

- Rolling conversation windows
- Configurable limits (turns, tokens, age)
- Character-specific memory policies
- Privacy-focused (in-memory only)
        """,
        routes=app.routes,
        servers=[
            {"url": "http://localhost:8000", "description": "Development server"},
            {"url": "https://api.character.ai.com", "description": "Production server"},
        ],
    )

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token obtained from device registration",
        },
        "AdminToken": {
            "type": "apiKey",
            "in": "header",
            "name": "x-admin-token",
            "description": "Admin token for administrative operations",
        },
    }

    # Add global security requirements
    openapi_schema["security"] = [{"BearerAuth": []}, {"AdminToken": []}]

    # Add custom tags for better organization
    openapi_schema["tags"] = [
        {
            "name": "Authentication",
            "description": "Device registration and authentication endpoints",
        },
        {"name": "Characters", "description": "Character management and configuration"},
        {
            "name": "Interaction",
            "description": "Real-time interaction and audio processing",
        },
        {"name": "Memory", "description": "Session memory management"},
        {"name": "Safety", "description": "Content safety and analysis"},
        {"name": "Voice", "description": "Voice artifact management"},
        {"name": "System", "description": "System status and performance metrics"},
    ]

    # Add examples for common request/response patterns
    openapi_schema["components"]["examples"] = {
        "CharacterCreate": {
            "summary": "Create a new character",
            "value": {
                "name": "Sparkle the Unicorn",
                "character_type": "pony",
                "custom_topics": ["friendship", "magic", "adventure"],
            },
        },
        "AudioProcessing": {
            "summary": "Process audio input",
            "value": {
                "audio_data": "base64-encoded-audio-data",
                "character_name": "sparkle",
                "format": "wav",
            },
        },
        "SafetyAnalysis": {
            "summary": "Analyze text for safety",
            "value": {"text": "Hello, how are you today?"},
        },
    }

    return openapi_schema


def generate_api_docs(output_dir: Path) -> None:
    """Generate comprehensive API documentation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate OpenAPI spec
    openapi_spec = generate_openapi_spec()

    # Save as JSON
    json_path = output_dir / "openapi.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(openapi_spec, f, indent=2)
    print(f"Generated OpenAPI JSON: {json_path}")

    # Save as YAML
    yaml_path = output_dir / "openapi.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            openapi_spec,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    print(f"Generated OpenAPI YAML: {yaml_path}")

    # Generate endpoint summary
    try:
        generate_endpoint_summary(openapi_spec, output_dir)
    except Exception as e:
        print(f"Warning: Failed to generate endpoint summary: {e}")

    # Generate authentication guide
    try:
        generate_auth_guide(output_dir)
    except Exception as e:
        print(f"Warning: Failed to generate authentication guide: {e}")

    # Generate character profile examples
    try:
        generate_character_examples(output_dir)
    except Exception as e:
        print(f"Warning: Failed to generate character examples: {e}")


def generate_endpoint_summary(spec: Dict[str, Any], output_dir: Path) -> None:
    """Generate a summary of all API endpoints."""
    summary_path = output_dir / "endpoints.md"

    endpoints = []
    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                endpoints.append(
                    {
                        "method": method.upper(),
                        "path": path,
                        "summary": details.get("summary", ""),
                        "description": details.get("description", ""),
                        "tags": details.get("tags", []),
                        "security": details.get("security", []),
                    }
                )

    # Sort by tag, then by method, then by path
    endpoints.sort(
        key=lambda x: (
            x["tags"][0] if x["tags"] and len(x["tags"]) > 0 else "",
            x["method"],
            x["path"],
        )
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# API Endpoints Summary\n\n")
        f.write(f"Total endpoints: {len(endpoints)}\n\n")

        current_tag = None
        for endpoint in endpoints:
            if endpoint["tags"] and endpoint["tags"][0] != current_tag:
                current_tag = endpoint["tags"][0]
                f.write(f"\n## {current_tag}\n\n")

            f.write(f"### {endpoint['method']} {endpoint['path']}\n")
            f.write(f"**Summary**: {endpoint['summary']}\n\n")
            if endpoint["description"]:
                f.write(f"**Description**: {endpoint['description']}\n\n")
            if endpoint["security"]:
                f.write(f"**Security**: {', '.join(endpoint['security'])}\n\n")
            f.write("---\n\n")

    print(f"Generated endpoint summary: {summary_path}")


def generate_auth_guide(output_dir: Path) -> None:
    """Generate authentication guide."""
    auth_path = output_dir / "authentication.md"

    with open(auth_path, "w", encoding="utf-8") as f:
        f.write(
            """# Authentication Guide

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
curl -H "Authorization: Bearer your-jwt-token" \\
     https://api.example.com/api/v1/character/characters
```

### 3. Admin Token Authentication

For administrative operations, use the x-admin-token header:

```bash
curl -H "x-admin-token: your-admin-token" \\
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
"""
        )

    print(f"Generated authentication guide: {auth_path}")


def generate_character_examples(output_dir: Path) -> None:
    """Generate character profile examples."""
    examples_path = output_dir / "character_examples.md"

    with open(examples_path, "w", encoding="utf-8") as f:
        f.write(
            """# Character Profile Examples

## Basic Character Profile

```yaml
# profile.yaml
schema_version: 1
id: sparkle
display_name: "Sparkle the Unicorn"
character_type: pony
language: en
traits:
  personality: "friendly and magical"
  favorite_color: "rainbow"
  special_power: "spreading joy"
voice_style: "cheerful and energetic"
topics:
  - friendship
  - magic
  - adventure
  - helping others
safety:
  content_filter: true
  age_appropriate: true
  banned_topics: ["violence", "scary"]
llm:
  model: "llama-2-7b-chat"
  temperature: 0.7
  max_tokens: 150
stt:
  model: "wav2vec2-base"
  language: "en"
tts:
  model: "coqui"
  voice_id: "sparkle_voice"
  speed: 1.0
consent:
  subject: "adult_guardian"
  date: "2024-01-15"
  purpose: "educational_entertainment"
  retention: "1_year"
```

## Advanced Character Profile

```yaml
# profile.yaml
schema_version: 1
id: robot_buddy
display_name: "Robo-Buddy"
character_type: robot
language: en
traits:
  personality: "helpful and curious"
  intelligence_level: "high"
  learning_ability: "adaptive"
  special_features: ["problem_solving", "teaching"]
voice_style: "friendly and robotic"
topics:
  - science
  - technology
  - learning
  - problem_solving
  - space
safety:
  content_filter: true
  age_appropriate: true
  educational_focus: true
  banned_topics: ["violence", "inappropriate"]
llm:
  model: "llama-2-13b-chat"
  temperature: 0.5
  max_tokens: 200
  system_prompt: "You are a helpful robot assistant focused on education and learning."
stt:
  model: "wav2vec2-large"
  language: "en"
  noise_reduction: true
tts:
  model: "coqui"
  voice_id: "robot_voice"
  speed: 0.9
  pitch: 1.1
consent:
  subject: "parent_guardian"
  date: "2024-01-15"
  purpose: "educational_assistance"
  retention: "2_years"
  data_sharing: false
```

## Custom Prompt Template

```markdown
# prompt.md
You are {{character_name}}, a {{character_type}} with a {{voice_style}} personality.

Your key traits:
{% for key, value in traits.items() %}
- {{key}}: {{value}}
{% endfor %}

You enjoy talking about: {{topics|join(', ')}}

Remember to:
- Be {{voice_style}} in your responses
- Stay focused on {{topics|join(', ')}}
- Keep things appropriate for children
- Be helpful and encouraging

User says: {{user_input}}

Respond as {{character_name}}:
```

## Character Index

```yaml
# index.yaml
schema_version: 1
characters:
  - id: sparkle
    display_name: "Sparkle the Unicorn"
    character_type: pony
    status: active
    last_updated: "2024-01-15T10:00:00Z"
  - id: robot_buddy
    display_name: "Robo-Buddy"
    character_type: robot
    status: active
    last_updated: "2024-01-15T10:00:00Z"
```

## API Usage Examples

### Create Character
```bash
curl -X POST "https://api.example.com/api/v1/character/character/create" \\
  -H "Authorization: Bearer your-token" \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "Sparkle",
    "character_type": "pony",
    "custom_topics": ["friendship", "magic"]
  }'
```

### Upload Character Profile
```bash
curl -X POST "https://api.example.com/api/v1/character/profiles/upload" \\
  -H "Authorization: Bearer your-token" \\
  -F "file=@character_profile.zip"
```

### Get Character Information
```bash
curl -H "Authorization: Bearer your-token" \\
     "https://api.example.com/api/v1/character/characters/sparkle"
```
"""
        )

    print(f"Generated character examples: {examples_path}")


def main():
    """Main function to generate all API documentation."""
    output_dir = Path("docs/api")

    print("Generating Character AI API Documentation...")
    print(f"Output directory: {output_dir}")

    try:
        generate_api_docs(output_dir)
        print("\n✅ API documentation generated successfully!")
        print(f"📁 Files created in: {output_dir}")
        print("\nGenerated files:")
        for file_path in output_dir.glob("*"):
            print(f"  - {file_path.name}")

    except Exception as e:
        print(f"❌ Error generating API documentation: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
