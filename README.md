# Character AI Platform

[![Tests](https://img.shields.io/badge/tests-117%20passing-green)](https://github.com/parrenthesis/character-ai/actions)
[![Coverage](https://img.shields.io/badge/coverage-targeting%2095%25-blue)](https://github.com/your-org/character-ai)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-optimized-blue)](https://hub.docker.com)
[![CI/CD](https://img.shields.io/badge/ci%2Fcd-github%20actions-green)](https://github.com/your-org/character-ai/actions)

> **AI character platform for interactive toys and applications**

Character AI is a production-ready system for creating, managing, and deploying AI-powered interactive characters that bring digital personalities to life. Built for edge devices and cloud deployment with enterprise security, monitoring, and scalability.

## ✨ Features

### Core Platform
- **Character templates** with 20+ pre-defined character types
- **Multi-language support** (8 languages)
- **Voice synthesis** with emotion and personality
- **Conversational AI** with context awareness
- **Real-time interaction** with low latency
- **Child-safe content filtering** with parental controls

### Production Ready
- **Enterprise security** with JWT authentication and RBAC
- **Comprehensive monitoring** with Prometheus/Grafana
- **Structured logging** with ELK stack integration
- **Performance optimization** with latency budgets
- **Health checks** and alerting
- **Docker and Kubernetes** deployment
- **CI/CD pipeline** with GitHub Actions
- **Automated security scanning** with Bandit, Safety, Trivy
- **Dependency management** with Dependabot

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
# Run tests
make test

# Run linting and formatting
make lint
make format

# Run security checks
make security

# Build and test Docker images locally
make docker-build
make docker-test
```

## 🏗️ Architecture

```
┌─────────────────┬─────────────────┬─────────────────┬───────────┐
│   Web Layer    │   Core Layer   │  Hardware Layer │ Monitoring│
├─────────────────┼─────────────────┼─────────────────┼───────────┤
│ • FastAPI      │ • Character     │ • Toy Hardware │ • Prometheus│
│ • WebSocket    │   Management    │ • Sensors       │ • Grafana  │
│ • REST API     │ • Voice Engine  │ • Power Mgmt   │ • ELK Stack│
│ • Authentication│ • LLM Integration│ • Audio I/O    │ • Alerts   │
├─────────────────┼─────────────────┼─────────────────┼───────────┤
│ • Click CLI    │ • Safety        │ • Monitoring   │ • Logging  │
│ • Web UI       │ • Algorithms    │ • Performance  │ • Metrics  │
└─────────────────┴─────────────────┴─────────────────┴───────────┘
```

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
# Build models bundle
make bundle

# Build production Docker images
make models-image
make runtime-image

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
# Run all tests
make test

# Run with coverage
make test-coverage

# Run development tests
make test-dev

# Run tests in Docker
make docker-test

# Run security checks
make security
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

### Character Configuration

Characters are configured via YAML files in `configs/characters/`:

```yaml
# configs/characters/data/profile.yaml
name: "Data"
personality: "Logical, curious, helpful"
voice: "calm, measured"
language: "en"
safety_level: "child_safe"
```

## 📊 Monitoring & Observability

### Metrics (Prometheus)
- Request latency and throughput
- Character interaction metrics
- System resource usage
- Error rates and success rates

### Logging (ELK Stack)
- Structured JSON logs
- Request tracing
- Error aggregation
- Performance profiling

### Dashboards (Grafana)
- Real-time system health
- Character performance metrics
- User interaction analytics
- Infrastructure monitoring

## 🔒 Security

### Authentication & Authorization
- JWT-based authentication
- Role-based access control (RBAC)
- Device identity management
- Secure key generation

### Content Safety
- Child-safe content filtering
- Parental control integration
- Content moderation
- Safety level enforcement

### Security Scanning
- Automated vulnerability scanning
- Dependency security checks
- Code security analysis
- Container security scanning

## 🚀 CI/CD Pipeline

### Automated Testing
- **Automated Testing**: 117 tests run on every commit
- **Code Quality**: Linting, type checking, and formatting
- **Security Scanning**: Bandit, Safety, and Trivy scans
- **Docker Builds**: Production images built on releases only
- **Dependency Updates**: Automated via Dependabot
- **Release Management**: Automated versioning and changelog generation

### GitHub Actions Workflows

- **`ci.yml`**: Main CI pipeline with testing and linting
- **`docker-publish.yml`**: Production Docker image publishing (releases only)
- **`security-scan.yml`**: Weekly security vulnerability scans
- **`dependabot.yml`**: Automated dependency updates

### Development Workflow

```bash
# Local development
make setup-dev
make test
make lint
make docker-build

# CI/CD focuses on code quality
# Docker builds only happen on releases
```

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **API Documentation**: Complete API reference with examples
- **Character Examples**: Sample character configurations
- **Production Deployment**: Step-by-step deployment guides
- **Toy Deployment**: Hardware-specific deployment instructions
- **Security Guide**: Authentication and security best practices

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Set up development environment
make setup-dev

# Run tests
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

## 🎯 Roadmap

### Completed ✅
- Core platform with 117 tests passing
- Multi-language support (8 languages)
- 20+ pre-defined character templates
- Production-ready CI/CD pipeline
- Docker optimization and local development workflow
- Comprehensive security scanning
- Monitoring and observability stack

### In Progress 🚧
- Performance optimization
- Advanced character customization
- Enhanced monitoring dashboards

### Planned 📋
- Mobile app integration
- Advanced analytics
- Multi-tenant support
- Cloud deployment automation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/parrenthesis/character-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/parrenthesis/character-ai/discussions)
- **Security**: [GitHub Security Advisories](https://github.com/parrenthesis/character-ai/security/advisories)

---

**Character AI** - Bringing digital personalities to life through AI-powered interactive experiences.