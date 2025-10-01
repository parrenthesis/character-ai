# 🧠 Character AI

[![Tests](https://img.shields.io/badge/tests-117%2F117%20passing-brightgreen)](https://github.com/your-org/character-ai)
[![Coverage](https://img.shields.io/badge/coverage-targeting%2095%25-blue)](https://github.com/your-org/character-ai)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-optimized-blue)](https://hub.docker.com)
[![CI/CD](https://img.shields.io/badge/ci%2Fcd-github%20actions-green)](https://github.com/your-org/character-ai/actions)

> **AI character platform for interactive toys and applications**

Character AI is a production-ready system for creating, managing, and deploying AI-powered interactive characters that bring digital personalities to life. Built for edge devices and cloud deployment with enterprise security, monitoring, and scalability.

## ✨ Features

### 🎭 **Character Management**
- **Multi-dimensional characters** with species, archetypes, personality traits, abilities, and topics
- **AI-powered character generation** using advanced LLMs
- **Character templates** with 20+ pre-defined character types
- **Character search and filtering** with advanced query capabilities
- **Character relationships** and franchise management

### 🎤 **Voice Processing**
- **Real-time speech-to-text** with Whisper integration
- **Text-to-speech synthesis** with XTTS voice cloning
- **Multi-language support** (8 languages)
- **Voice activity detection** and noise reduction
- **Character voice injection** and customization

### 🤖 **LLM Integration**
- **Multiple LLM providers** (OpenAI, Anthropic, Local models)
- **Open model management** with Ollama integration
- **Model optimization** for edge deployment
- **Cost monitoring** and usage tracking
- **Fallback mechanisms** for reliability

### 🔒 **Enterprise Security**
- **JWT authentication** with device identity management
- **Role-based access control** (RBAC)
- **Rate limiting** and security middleware
- **Cryptographic key management**
- **Parental controls** with content filtering

### 📊 **Production Ready**
- **Comprehensive monitoring** with Prometheus/Grafana
- **Structured logging** with ELK stack integration
- **Performance optimization** with latency budgets
- **Health checks** and alerting
- **Docker and Kubernetes** deployment
- **CI/CD pipeline** with GitHub Actions
- **Automated security scanning** with Bandit, Safety, Trivy
- **Dependency management** with Dependabot

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/character.ai.git
cd character.ai

# Set up environment (handles Python, dependencies, conflicts)
make setup

# Set up development environment (includes dev dependencies)
make setup-dev

# Set up environment variables (optional - only needed for custom configuration)
cp .env.example .env
# Edit .env with your configuration (uses CAI_* prefix)
# Only CAI_JWT_SECRET is required; everything else has sensible defaults

# Run tests
make test

# Start the platform
poetry run cai --help
```

### Basic Usage

```bash
# Create a character (interactive mode)
cai character create --name "FriendlyBot" --interactive

# Create from template
cai character create --template "friendly_robot" --name "MyRobot"

# Create with AI generation
cai character create --ai-generate "A wise dragon who loves to help children learn"

# List characters
cai character list

# List available LLM models
cai llm list-models

# Test character interaction
cai character test FriendlyBot --interactive
```

### Character Directory Structure

Characters are now stored in the new schema format:

```
configs/characters/
├── friendly_bot/
│   ├── profile.yaml      # Character profile and configuration
│   ├── prompts.yaml      # System prompts and conversation templates
│   └── voice_samples/    # Voice sample files for cloning
└── wise_dragon/
    ├── profile.yaml
    ├── prompts.yaml
    └── voice_samples/
```

## 📖 Documentation

- **[Production Deployment Guide](docs/production_deployment.md)** - Complete deployment guide
- **[API Documentation](docs/api/production_api.md)** - REST API reference
- **[Toy Deployment Guide](docs/toy_deployment.md)** - Edge device deployment
- **[Character Examples](docs/api/character_examples.md)** - Character creation examples
- **[Security Guide](docs/current_security_vulnerabilities.md)** - Security best practices

## 🏗️ Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web API       │    │   CLI Tools     │    │   Production    │
│   (FastAPI)     │    │   (Click)       │    │   Engine        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                    Core Platform                                │
├─────────────────┬─────────────────┬─────────────────┬───────────┤
│  Character      │  Voice          │  LLM            │  Security │
│  Management     │  Processing     │  Integration    │  & Auth   │
├─────────────────┼─────────────────┼─────────────────┼───────────┤
│  • Enhanced     │  • Whisper      │  • OpenAI       │  • JWT    │
│  • Templates    │  • XTTS         │  • Anthropic    │  • RBAC   │
│  • AI Gen       │  • Multi-lang   │  • Local        │  • Rate   │
│  • Search       │  • VAD          │  • Ollama       │  • Keys   │
└─────────────────┴─────────────────┴─────────────────┴───────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                    Additional Modules                          │
├─────────────────┬─────────────────┬─────────────────┬───────────┤
│  Hardware       │  Monitoring     │  Algorithms    │  Safety   │
│  Management     │  & Metrics      │  & AI          │  & Parental│
├─────────────────┼─────────────────┼─────────────────┼───────────┤
│  • Power Mgmt   │  • Prometheus    │  • Conversational│  • Content│
│  • Sensors      │  • Grafana      │  • Safety      │  • Filters │
│  • Toy Hardware │  • ELK Stack    │  • Classifiers │  • Controls│
└─────────────────┴─────────────────┴─────────────────┴───────────┘
```

### Deployment Options

#### 🐳 **Docker Deployment**
```bash
# Build models bundle
make bundle

# Build Docker images
make models-image
make runtime-image

# Run with Docker Compose (includes monitoring)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f character-ai

# Access monitoring
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

#### ☸️ **Kubernetes Deployment**
```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment
kubectl get pods -n cai-production
```

#### 🖥️ **Traditional Server**
```bash
# Install system dependencies
sudo apt install python3.10 nginx redis-server

# Deploy application
poetry install --no-dev
sudo systemctl start cai
```

## 🧪 Testing

The platform includes comprehensive testing:

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/character_ai --cov-report=html

# Run integration tests
CAI_RUN_INTEGRATION=1 poetry run pytest -m integration

# Run performance tests
poetry run pytest --benchmark-only
```

**Test Coverage:**
- ✅ **117 unit tests** passing
- ✅ **Integration tests** available
- ✅ **Targeting 95% coverage** (configured in pyproject.toml)
- ✅ **All security tests** passing

## 🔧 Configuration

### Environment Variables

**Everything works out of the box** Only set these when you need custom behavior:

```bash
# OPTIONAL: Override defaults only when needed
export CAI_DEBUG="true"                    # Enable debug mode
export CAI_MODELS__WHISPER_MODEL="tiny"   # Use smaller model
export CAI_API__HOST="0.0.0.0"           # Allow external connections
export CAI_GPU__DEVICE="cpu"              # Force CPU usage

# OPTIONAL: JWT secret for authentication (uses empty string by default)
export CAI_JWT_SECRET="your-secret-here"  # Only needed for auth endpoints

# OPTIONAL: Cloud LLM providers (falls back to local if not set)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

### Configuration Files

```yaml
# config.yaml
environment: production
models:
  whisper_model: "tiny"
  llama_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  xtts_model: "tts_models/multilingual/multi-dataset/xtts_v2"
security:
  require_https: true
  rate_limit_requests_per_minute: 1000
```

## 📊 Monitoring & CI/CD

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed

# Metrics endpoint
curl http://localhost:8000/metrics
```

### CI/CD Pipeline

The platform includes a comprehensive CI/CD pipeline:

- **Automated Testing**: 117 tests run on every commit
- **Code Quality**: Linting, type checking, and formatting
- **Security Scanning**: Bandit, Safety, and Trivy scans
- **Docker Builds**: Multi-platform image building
- **Dependency Updates**: Automated via Dependabot
- **Release Management**: Automated versioning and changelog generation

### GitHub Actions Workflows

- **`ci.yml`**: Main CI pipeline with testing and linting
- **`docker-publish.yml`**: Multi-platform Docker image publishing
- **`security-scan.yml`**: Weekly security vulnerability scans
- **`dependabot.yml`**: Automated dependency updates

### Monitoring Stack

- **Prometheus** for metrics collection (http://localhost:9090)
- **Grafana** for visualization (http://localhost:3000)
- **ELK Stack** for log aggregation
- **AlertManager** for notifications
- **Health checks** at `/health` and `/metrics` endpoints
- **Real-time dashboards** for system performance

## 🔒 Security

### Security Features

- **JWT Authentication** with device identity
- **Role-based Access Control** (RBAC)
- **Rate Limiting** and DDoS protection
- **Input Validation** and sanitization
- **Parental Controls** with content filtering
- **Audit Logging** and monitoring

### Security Deployment Checklist

- [ ] SSL/TLS certificates installed
- [ ] HTTPS enforced  
- [ ] Firewall rules configured
- [ ] Environment variables secured

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Set up development environment (handles all dependencies and conflicts)
make setup-dev

# Install pre-commit hooks
pre-commit install

# Run tests
make test

# Run linting and auto-fix
make lint

# Run security checks
make security

# Run all checks
make check-all
```

## 📄 License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/character.ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/character.ai/discussions)

## 🗺️ Roadmap

### ✅ **Completed**
- Core platform with 117 tests passing
- Multi-language support (8 languages)
- Character management system
- Voice processing pipeline
- LLM integration and management
- Security and authentication
- Production deployment guides
- **Docker optimization** with multi-stage builds
- **Monitoring stack** with Prometheus/Grafana
- **CI/CD pipeline** with GitHub Actions
- **Security scanning** with automated tools
- **Dependency management** with Dependabot

### 🚧 **In Progress**
- Enhanced character testing and voice integration
- Performance optimization and test coverage (targeting 95%+)
- Documentation updates and API generation

### 📋 **Planned**
- Advanced character relationships
- Multi-language support expansion
- Mobile SDK development
- Advanced analytics
- Multi-tenant support
- Enterprise integrations

---

**Built with ❤️ for the future of interactive AI**