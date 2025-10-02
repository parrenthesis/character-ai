# Character AI Platform

[![Tests](https://img.shields.io/badge/tests-117%20passing-green)](https://github.com/parrenthesis/character-ai/actions)
[![Security](https://img.shields.io/badge/security-0%20vulnerabilities-green)](https://github.com/parrenthesis/character-ai/security)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![Architecture](https://img.shields.io/badge/architecture-secure%20PyTorch%202.8.0-blue)](https://pytorch.org)

> **AI character platform for interactive applications**

A secure, production-ready system for creating AI-powered interactive characters that bring digital personalities to life, with real-time voice interaction, built with modern ML frameworks and enterprise-grade security.

## ✨ What It Does

**Core Functionality:**
- **Real-time voice interaction** with speech-to-text and text-to-speech
- **AI character personalities** with conversational memory
- **Voice cloning** for custom character voices
- **Multi-language support** with cultural adaptation
- **Child-safe content filtering** and parental controls

**System Design:**
```
Audio Input → Wav2Vec2 (STT) → LLM → Coqui TTS (TTS + Voice Cloning) → Audio Output
```

**Architecture:**
- **Secure PyTorch 2.8.0+** with all security vulnerabilities patched
- **Wav2Vec2** for speech recognition
- **Coqui TTS** for speech synthesis and voice cloning
- **Edge-optimized** for low-latency real-time interaction
- **Production-ready** with monitoring, logging, and security scanning

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Poetry (for dependency management)
- Docker (for containerized deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/parrenthesis/character-ai.git
cd character-ai

# Set up development environment
make setup-dev

# Run tests to verify installation
make test
```

### Development Workflow

```bash
# Run tests (117 tests)
make test

# Run linting and formatting
make lint

# Run security checks
make security

# Build and test Docker images locally
make docker-build    # Builds development image
make docker-test     # Runs tests in Docker environment
```

## 🏗️ Architecture

**Secure ML Pipeline:**
```
Audio Input → Wav2Vec2 (STT) → LLM → Coqui TTS (TTS + Voice Cloning) → Audio Output
```

**System Components:**
- **Web Layer**: FastAPI, WebSocket, REST API, Authentication
- **Core Layer**: Character Management, Voice Engine, LLM Integration, Safety
- **ML Layer**: Wav2Vec2 (STT), Coqui TTS (TTS + Voice Cloning), Secure PyTorch 2.8.0+
- **Monitoring**: Prometheus, Grafana, ELK Stack, Security Scanning

**Why This Architecture:**
- **Separate ML models image**: Pre-downloads large ML models (Wav2Vec2, Coqui TTS) to avoid download delays at runtime
- **Secure PyTorch 2.8.0+**: Patches all critical security vulnerabilities
- **Wav2Vec2 + Coqui TTS**: Secure, unified voice pipeline
- **Edge-optimized**: Designed for low-latency real-time interaction on edge devices

### Deployment Options

#### 🐳 **Docker Development**
```bash
# Build and run locally for development
make docker-build
make docker-run

# Or use docker-compose for full stack
make docker-compose-up

# Run tests in Docker
make docker-test

# Clean up Docker resources
make docker-clean
```

#### 🏭 **Production Deployment**
```bash
# Build models bundle (downloads ML models)
make bundle

# Build production Docker images
make models-image    # Contains pre-downloaded ML models
make runtime-image   # Contains the application runtime

# Deploy with Docker Compose (includes monitoring)
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
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=character-ai

# Access services
kubectl port-forward svc/character-ai 8000:8000
```

#### 🖥️ **Bare Metal Deployment**
```bash
# Install system dependencies
sudo apt update
sudo apt install python3.10 nginx redis-server

# Deploy application
poetry install --no-dev
sudo systemctl start cai
```

## 🧪 Testing

The platform includes comprehensive testing:

```bash
# Run all tests (117 tests)
make test

# Run with coverage
make test-coverage

# Run tests in Docker
make docker-test

# Run security checks
make security
```

**Test Status:**
- ✅ **117 tests passing** (100% pass rate)
- ✅ **0 security vulnerabilities** (Bandit, Safety, Detect-secrets)
- ✅ **All linting clean** (Ruff, Black, Isort, MyPy)
- ✅ **Secure architecture** (PyTorch 2.8.0+, Wav2Vec2, Coqui TTS)

## 🔧 Configuration

### Environment Variables

**Works out of the box** - only set these for custom behavior:

```bash
# OPTIONAL: Override defaults
export CAI_DEBUG="true"                    # Enable debug mode
export CAI_API__HOST="0.0.0.0"           # Allow external connections
export CAI_GPU__DEVICE="cpu"              # Force CPU usage

# OPTIONAL: Authentication (uses empty string by default)
export CAI_JWT_SECRET="your-secret-here"  # Only needed for auth endpoints

# OPTIONAL: Cloud LLM providers (falls back to local if not set)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

### Character Configuration

Characters are configured via YAML files in `configs/characters/`:

```yaml
# configs/characters/data/profile.yaml
name: "Data" # From Star Trek: The Next Generation
personality: "Logical, curious, helpful"
voice: "calm, measured"
language: "en"
safety_level: "child_safe"
```

## 📊 Monitoring & Observability

**Metrics (Prometheus):**
- Request latency and throughput
- Character interaction metrics
- System resource usage
- Error rates and success rates

**Logging (ELK Stack):**
- Structured JSON logs
- Request tracing
- Error aggregation
- Performance profiling

**Dashboards (Grafana):**
- Real-time system health
- Character performance metrics
- User interaction analytics
- Infrastructure monitoring

## 🔒 Security

**Authentication & Authorization:**
- JWT-based authentication
- Role-based access control (RBAC)
- Device identity management
- Secure key generation

**Content Safety:**
- Child-safe content filtering
- Parental control integration
- Content moderation
- Safety level enforcement

**Security Scanning:**
- Automated vulnerability scanning (Bandit, Safety, Detect-secrets)
- Dependency security checks
- Code security analysis
- Container security scanning

## 🚀 CI/CD Pipeline

**Automated Testing:**
- **117 tests** run on every commit
- **Code Quality**: Linting, type checking, and formatting
- **Security Scanning**: Bandit, Safety, and Detect-secrets scans
- **Docker Builds**: Production images built on releases only
- **Dependency Updates**: Automated via Dependabot

**GitHub Actions Workflows:**
- **`ci.yml`**: Main CI pipeline with testing and linting
- **`security-scan.yml`**: Weekly security vulnerability scans
- **`pr-checks.yml`**: Pull request validation

**Development Workflow:**
```bash
# Local development
make setup-dev
make test
make lint
make docker-build
```

## 📚 Documentation

Documentation is available in the `docs/` directory:

- **API Documentation**: Complete API reference with examples
- **Character Examples**: Sample character configurations
- **Production Deployment**: Step-by-step deployment guides
- **Security Guide**: Authentication and security best practices

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Set up development environment
make setup-dev

# Run tests (117 tests)
make test

# Run linting
make lint

# Build Docker images locally
make docker-build
```

### Code Quality

- **Linting**: Black, isort, ruff, mypy
- **Testing**: pytest with coverage
- **Security**: Bandit, Safety, detect-secrets
- **Documentation**: Sphinx with autodoc

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🎯 Current Status

**✅ Completed:**
- Secure architecture with PyTorch 2.8.0+ (all security vulnerabilities patched)
- Wav2Vec2 + Coqui TTS pipeline
- 117 tests passing with 0 security vulnerabilities
- Production-ready CI/CD pipeline
- Comprehensive monitoring and security scanning
- OpenAPI documentation generated

**🚧 In Development:**
- Performance optimization for edge devices
- Advanced voice cloning capabilities
- Enhanced monitoring dashboards

**📋 Future Work:**
- Advanced analytics
- Multi-tenant support
- Cloud deployment automation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/parrenthesis/character-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/parrenthesis/character-ai/discussions)
- **Security**: [GitHub Security Advisories](https://github.com/parrenthesis/character-ai/security/advisories)

---

**Character AI** - Secure, production-ready AI character platform with real-time voice interaction.
