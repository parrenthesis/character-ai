# Character AI Platform

[![Tests](https://img.shields.io/badge/tests-117%20passing-green)](https://github.com/parrenthesis/character-ai/actions)
[![Security](https://img.shields.io/badge/security-0%20vulnerabilities-green)](https://github.com/parrenthesis/character-ai/security)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)

> **Voice-driven AI character system**

A proof-of-concept platform for creating interactive AI characters that bring digital personalities to life, with real-time voice interaction using modern ML frameworks.

## What It Does

- **Voice interaction**: Speech-to-text → LLM → Text-to-speech
- **Character personalities**: Configurable AI personalities with memory
- **Voice cloning**: Custom character voices using Coqui TTS
- **Edge deployment**: Designed for local hardware (Raspberry Pi, etc.)

**Pipeline:**
```
Audio Input → Wav2Vec2 (STT) → LLM → Coqui TTS → Audio Output
```

## Documentation

- **API Documentation**: `docs/api/`
- **Character Examples**: `docs/api/character_examples.md`
- **Deployment**: `docs/production_deployment.md`

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

**Components:**
- **Web**: FastAPI, WebSocket, REST API
- **Core**: Character Management, Voice Engine, LLM Integration
- **ML**: Wav2Vec2 (STT), Coqui TTS (TTS + Voice Cloning)
- **Monitoring**: Prometheus, Grafana, Security Scanning

**Edge Deployment:**
- Pre-downloaded ML models for offline operation
- Optimized for local hardware (Raspberry Pi, etc.)

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
```

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
