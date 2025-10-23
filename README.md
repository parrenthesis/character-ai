# Character AI Platform

[![Tests](https://img.shields.io/badge/tests-98%20passing-green)](https://github.com/parrenthesis/character-ai/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)

> **Voice-driven AI character system**

A proof-of-concept platform for creating interactive AI characters that bring digital personalities to life, with real-time voice interaction using modern ML frameworks.

## What It Does

- **Voice interaction**: Speech-to-text → LLM → Text-to-speech
- **Character personalities**: Configurable AI personalities with persistent memory
- **Voice cloning**: Custom character voices using Coqui TTS
- **Hybrid memory system**: Intelligent conversation memory and user preference tracking
- **Edge deployment**: Designed for local hardware (Raspberry Pi, etc.)

**Pipeline:**
```
Audio Input → Wav2Vec2 (STT) → LLM → Coqui TTS → Audio Output
```

## Documentation

- **API Documentation**: `docs/api/`
- **Character Examples**: `docs/api/character_examples.md`
- **Memory System**: `docs/memory_system.md`
- **Deployment**: `docs/production_deployment.md`
- **Toy Deployment**: `docs/toy_deployment.md`

## Quick Start

### Prerequisites
- Python 3.10+
- Poetry
- Docker (optional)

### Setup

```bash
git clone https://github.com/parrenthesis/character-ai.git
cd character-ai
make setup
```

### Development

```bash
make setup-dev
make test
make lint
```

## Architecture

**Modular Components:**
- **`algorithms/`**: AI algorithms (STT, TTS, voice cloning, safety, memory)
- **`characters/`**: Character management (catalog, voices, safety, management)
- **`core/`**: Core platform (config, audio I/O, LLM, persistence, caching, database)
- **`features/`**: Feature modules (localization, security, parental controls, cost monitoring)
- **`observability/`**: Monitoring (logging, metrics, crash reporting, Grafana)
- **`production/`**: Production deployment (real-time engine, processing pipeline)
- **`services/`**: Service layer (STT, TTS, LLM, pipeline orchestration)
- **`web/`**: Web API (core endpoints, features, middleware, monitoring, streaming)
- **`hardware/`**: Hardware interfaces (power management, toy hardware)

**Edge Deployment:**
- Pre-downloaded ML models for offline operation
- Optimized for local hardware (Raspberry Pi, etc.)
- Modular architecture enables selective feature deployment

## Testing

```bash
make test
make test-coverage
make security
```

## Configuration

### Environment Variables (Optional)

```bash
export CAI_DEBUG="true"
export CAI_API__HOST="0.0.0.0"
export OPENAI_API_KEY="your-key-here"

# PyTorch 2.8+ compatibility (required for XTTS v2)
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

### Hardware Profiles

Optimize performance for different hardware configurations:

```bash
# Auto-detect hardware (default)
poetry run cai test voice-pipeline --character <name> --franchise <franchise> --realtime

# Use specific hardware profile
poetry run cai test voice-pipeline --character <name> --franchise <franchise> --realtime --hardware desktop
poetry run cai test voice-pipeline --character <name> --franchise <franchise> --realtime --hardware raspberry_pi
```

Available profiles in `configs/hardware/`:
- **desktop.yaml**: High-performance systems with GPU support
- **raspberry_pi.yaml**: Optimized for Raspberry Pi (CPU-only, reduced models)
- **orange_pi.yaml**: Optimized for Orange Pi devices

### Voice Activity Detection (VAD)

Configure speech detection thresholds in hardware profiles:

```yaml
# configs/hardware/desktop.yaml
vad:
  speech_start_threshold: 0.020  # Higher = ignore background noise
  speech_continue_threshold: 0.005  # Lower = sensitive during speech
  max_silence_duration_s: 0.4  # End speech after 400ms silence
  min_speech_duration_s: 0.05  # Minimum 50ms to count as speech
```

### Wake Word Detection

Enable wake word detection for hands-free operation:

```yaml
# In character profile.yaml
wake_words:
  - "hey computer"
  - "hello assistant"
```

Supports two detection methods:
- **energy**: Fast, lightweight energy-based detection
- **openwakeword**: ML-based detection (more accurate, higher latency)

### Character Configuration

```yaml
# configs/characters/data/profile.yaml
name: "Data" # From Star Trek: The Next Generation
personality: "Logical, curious, helpful"
voice: "calm, measured"
language: "en"
safety_level: "child_safe"
```

## Monitoring

- **Metrics**: Prometheus for system monitoring
- **Logging**: Structured JSON logs
- **Dashboards**: Grafana for visualization

## Security

- **Authentication**: JWT-based auth
- **Content Safety**: Child-safe filtering
- **Security Scanning**: Automated vulnerability checks

## Deployment

### Local Development
```bash
make docker-build
make docker-run
```

### Production
```bash
make bundle
docker-compose up -d
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Status

**Completed:**
- Voice interaction pipeline (STT → LLM → TTS)
- Character personality system
- Edge deployment support
- Security scanning and testing

**In Development:**
- Performance optimization for edge devices
- Enhanced voice cloning capabilities

## Support

- **Issues**: [GitHub Issues](https://github.com/parrenthesis/character-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/parrenthesis/character-ai/discussions)
