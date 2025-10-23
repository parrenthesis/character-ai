# Hybrid Memory System

## Overview

The Hybrid Memory System provides intelligent, persistent memory capabilities for Character AI, enabling characters to remember users, preferences, and conversation history across sessions. The system combines three approaches:

1. **Structured Preference Extraction**: Pattern-based extraction of user facts (name, interests, color, dislikes)
2. **Persistent Storage**: SQLite-based storage for cross-session memory
3. **LLM-based Summarization**: Intelligent conversation compression for infinite context

## Quick Start

### 1. Enable Memory System

```yaml
# configs/runtime.yaml
memory_system:
  enabled: true
  data_directory: "data"
```

### 2. Initialize in Code

```python
from src.character_ai.algorithms.conversational_ai.hybrid_memory import HybridMemorySystem, MemoryConfig

# Create config
memory_config = MemoryConfig(
    storage_db_path="data/conversations.db",
    preferences_storage_path="data/preferences.json"
)

# Initialize system
hybrid_memory = HybridMemorySystem(
    llm_provider=llm_provider,  # Optional
    config=memory_config,
    device_id="user_123"
)
```

### 3. Use in Conversation

```python
# Start session
session_id = hybrid_memory.start_session("character_name")

# Process turn
hybrid_memory.process_turn(
    character_name="character_name",
    user_input="Hello, my name is John",
    character_response="Hello John! Nice to meet you."
)

# Build context for LLM
context = hybrid_memory.build_context_for_llm(
    character_name="character_name",
    current_user_input="What do you know about me?"
)
```

## Architecture

### Core Components

```
HybridMemorySystem
├── PreferenceExtractor (Pattern-based extraction)
├── PreferenceStorage (JSON persistence)
├── OptimizedConversationStorage (SQLite with connection pooling)
├── ConversationSummarizer (LLM-based compression)
└── MemoryMetrics (Prometheus monitoring)
```

### Data Flow

```
User Input → Preference Extraction → Storage
           ↓
    Conversation Turn → SQLite Storage → Summarization
           ↓
    Context Building → LLM Prompt Enhancement
```

## Configuration

### Global Configuration (`configs/runtime.yaml`)

```yaml
memory_system:
  enabled: true  # Global feature flag
  data_directory: "data"  # Configurable path

  preferences:
    enabled: true
    use_llm_extraction: false  # Pattern matching only
    storage_path: "{data_directory}/user_preferences.json"

  storage:
    enabled: true
    db_path: "{data_directory}/conversations.db"
    max_turns_per_session: 50
    summarization_threshold: 20

  summarization:
    enabled: true
    max_summary_tokens: 150
    use_llm: true  # Fallback to pattern-based if false
```

### Hardware-Specific Configuration

The system automatically adapts to hardware capabilities:

- **Desktop**: Full hybrid system with LLM summarization
- **Raspberry Pi**: Lightweight mode with pattern-based extraction only
- **Orange Pi**: Balanced mode with selective LLM features

## Deployment

### Docker Deployment

The Docker setup uses a named volume `character_data` for persistent storage:

```yaml
# docker-compose.yml
volumes:
  character_data:/app/data

services:
  character-ai:
    volumes:
      - character_data:/app/data
```

### Data Persistence

- **Conversation History**: Stored in SQLite database at `/app/data/conversations.db`
- **User Preferences**: Stored in JSON file at `/app/data/user_preferences.json`
- **Memory Cache**: Stored in memory (temporary, not persistent)

### Backup and Restore

```bash
# Backup memory data
docker run --rm -v character_data:/data -v $(pwd):/backup alpine tar czf /backup/memory_backup.tar.gz -C /data .

# Restore memory data
docker run --rm -v character_data:/data -v $(pwd):/backup alpine tar xzf /backup/memory_backup.tar.gz -C /data
```

### Tar.gz Deployment

For tar.gz deployments, configure the data directory path in `configs/runtime.yaml`:

```yaml
memory_system:
  data_directory: "/opt/character-ai/data"  # Configurable path
```

## Features

### Preference Extraction

The system automatically extracts user preferences from conversations:

- **Name**: "My name is John" → `name: "John"`
- **Interests**: "I love hiking" → `interests: ["hiking"]`
- **Color**: "My favorite color is blue" → `favorite_color: "blue"`
- **Dislikes**: "I hate spiders" → `dislikes: ["spiders"]`

### Conversation Summarization

Long conversations are automatically summarized to maintain context:

- **Threshold**: Summarize when conversation exceeds 20 turns
- **Compression**: Maintains key information while reducing token count
- **Fallback**: Pattern-based summarization if LLM unavailable

### Cross-Session Memory

User preferences and conversation history persist across sessions:

- **Session Tracking**: Each conversation session is tracked
- **User Association**: Preferences linked to device ID
- **Context Building**: Previous conversations inform current responses

## Monitoring

### Metrics

The system provides comprehensive monitoring through Prometheus:

- **Memory Usage**: Cache hit rates, storage utilization
- **Performance**: Summarization latency, extraction accuracy
- **Storage**: Database size, conversation counts

### Grafana Dashboards

Pre-configured dashboards available in `monitoring/grafana/dashboards/`:

- **Memory System Dashboard**: System performance and usage
- **Character AI Dashboard**: Overall platform metrics

## CLI Commands

### Memory Management

```bash
# Backup memory data
poetry run cai memory backup --output memory_backup.tar.gz

# Restore memory data
poetry run cai memory restore --input memory_backup.tar.gz

# Clear user data (GDPR compliance)
poetry run cai memory clear --user-id user_123

# Export user data (GDPR compliance)
poetry run cai memory export --user-id user_123 --output user_data.json
```

## Security and Privacy

### Data Protection

- **Local Storage**: All data stored locally, no cloud transmission
- **Encryption**: Optional encryption for sensitive data
- **GDPR Compliance**: Full data export and deletion capabilities

### Access Control

- **Device Isolation**: Each device has separate memory space
- **Session Security**: Secure session management
- **Data Retention**: Configurable retention policies

## Troubleshooting

### Common Issues

1. **Memory Not Persisting**: Check data directory permissions
2. **Performance Issues**: Adjust summarization thresholds
3. **Storage Full**: Implement data retention policies

### Debug Mode

Enable debug logging for memory system:

```yaml
# configs/runtime.yaml
logging:
  level: DEBUG
  memory_system: true
```

## API Reference

### HybridMemorySystem

```python
class HybridMemorySystem:
    def start_session(self, character_name: str) -> str
    def process_turn(self, character_name: str, user_input: str, character_response: str) -> None
    def build_context_for_llm(self, character_name: str, current_user_input: str) -> str
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]
    def clear_user_data(self, user_id: str) -> None
```

### Configuration Classes

```python
class MemoryConfig:
    storage_db_path: str
    preferences_storage_path: str
    max_turns_per_session: int = 50
    summarization_threshold: int = 20
    max_summary_tokens: int = 150
    use_llm_summarization: bool = True
```
