# Contributing to Character AI

Thank you for your interest in contributing to the Character AI! This guide will help you get started with contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Poetry (for dependency management)
- Git
- Docker (optional, for local development)

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/your-username/character-ai.git
   cd character-ai
   ```

2. **Install dependencies**
   ```bash
   # Install Poetry if you haven't already
   curl -sSL https://install.python-poetry.org | python3 -

   # Set up development environment (handles all dependencies and conflicts)
   make setup-dev
   ```

3. **Set up pre-commit hooks**
   ```bash
   poetry run pre-commit install
   ```

4. **Create a development branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run development tests
make test-dev

# Run tests in Docker (optional)
make docker-test
```

### Test Structure

- **`tests/`**: Main test suite (runs in CI)
- **`tests_dev/`**: Development tests (local only)
- **Integration tests**: Marked with `@pytest.mark.integration`

### Writing Tests

```python
# tests/test_voice_manager.py
import pytest
from character_ai.characters.voices.voice_manager import VoiceManager

def test_voice_manager_initialization():
    """Test voice manager can be initialized."""
    manager = VoiceManager()
    assert manager is not None

@pytest.mark.integration
def test_voice_synthesis_integration():
    """Test voice synthesis with real audio."""
    # Integration test code here
    pass
```

## 🔍 Code Quality

### Linting and Formatting

```bash
# Run linting checks
make lint

# Auto-fix issues
make format

# Run development linting
make lint-dev
make format-dev
```

### Code Standards

- **Formatting**: Black, isort
- **Linting**: Ruff, mypy
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for all public functions
- **Testing**: pytest with comprehensive coverage

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality:

```bash
# Install pre-commit hooks
poetry run pre-commit install

# Run hooks manually
poetry run pre-commit run --all-files
```

## 🐳 Docker Development

### Local Docker Workflow

```bash
# Build Docker image locally
make docker-build

# Run container locally
make docker-run

# Run tests in Docker
make docker-test

# Run linting in Docker
make docker-lint

# Clean up Docker resources
make docker-clean
```

### Docker Compose

```bash
# Start full stack (includes monitoring)
make docker-compose-up

# Stop services
make docker-compose-down

# Check status
docker-compose ps
```

### Docker Best Practices

- **Local development**: Use `make docker-build` for testing
- **CI/CD**: Docker builds only happen on releases
- **Production**: Use `docker-publish.yml` workflow for releases
- **Development**: Focus on code quality, not Docker builds

## 🔒 Security

### Security Checks

```bash
# Run security scans
make security

# Check for secrets
poetry run detect-secrets scan --baseline .secrets.baseline

# Update secrets baseline
poetry run detect-secrets scan --update .secrets.baseline
```

### Security Guidelines

- **No secrets in code**: Use environment variables
- **Dependency scanning**: Regular security updates
- **Code review**: All changes require review
- **Security testing**: Automated in CI/CD

## 📝 Documentation

### Code Documentation

```python
def process_character_request(request: CharacterRequest) -> CharacterResponse:
    """Process a character interaction request.

    Args:
        request: The character request containing user input and context

    Returns:
        CharacterResponse with generated response and metadata

    Raises:
        ValidationError: If request format is invalid
        ProcessingError: If character processing fails
    """
    # Implementation here
    pass
```

### API Documentation

```bash
# Generate OpenAPI documentation
make generate-api-docs

# Generate JSON schemas
make generate-schemas
```

## 🚀 Development Workflow

### 1. Development Process

```bash
# Start development
git checkout -b feature/your-feature

# Make changes and test
make test
make lint

# Build and test Docker locally
make docker-build
make docker-test

# Commit changes
git add .
git commit -m "feat: add new feature"
```

### 2. Before Submitting

```bash
# Run full test suite
make test

# Run all quality checks
make check-all

# Build Docker image to ensure it works
make docker-build
```

### 3. Pull Request Process

1. **Create PR** with descriptive title and description
2. **Link issues** if applicable
3. **Add tests** for new functionality
4. **Update documentation** if needed
5. **Ensure CI passes** (tests, linting, security)

## 🏗️ Architecture Guidelines

### Project Structure

```
src/character_ai/
├── algorithms/          # AI algorithms (STT, TTS, voice cloning, safety)
│   ├── conversational_ai/  # Conversational AI processors
│   └── safety/            # Content safety and filtering
├── characters/          # Character management system
│   ├── catalog/          # Character catalog and storage
│   ├── management/       # Character lifecycle management
│   ├── safety/           # Character-specific safety filters
│   └── voices/           # Voice management and cloning
├── core/               # Core platform functionality
│   ├── audio_io/        # Audio input/output handling
│   ├── caching/         # Response caching
│   ├── config/          # Configuration management
│   ├── llm/             # Large Language Model integration
│   └── persistence/      # Data persistence layer
├── features/           # Feature modules
│   ├── cost_monitoring/ # Cost tracking and optimization
│   ├── localization/    # Multi-language support
│   ├── parental_controls/ # Parental control features
│   └── security/        # Security and authentication
├── hardware/           # Hardware interfaces
├── observability/      # Monitoring and observability
├── production/         # Production deployment
│   └── engine/         # Real-time processing engine
├── services/           # Service layer
│   ├── stt_service.py  # Speech-to-Text service
│   ├── tts_service.py  # Text-to-Speech service
│   ├── llm_service.py  # LLM service
│   └── pipeline_orchestrator.py # Pipeline coordination
└── web/               # Web API and interfaces
    ├── core/          # Core API endpoints
    ├── features/      # Feature-specific endpoints
    ├── middleware/    # Web middleware
    ├── monitoring/    # Monitoring endpoints
    └── streaming/     # Real-time streaming
```

### Adding New Features

1. **Create feature branch**
2. **Add tests first** (TDD approach)
3. **Implement feature**
4. **Update documentation**
5. **Test with Docker**
6. **Submit PR**

### Code Organization

- **Single responsibility**: Each module has one purpose
- **Dependency injection**: Use for testability
- **Interface segregation**: Small, focused interfaces
- **Error handling**: Comprehensive error management

## 🐛 Bug Reports

### Reporting Bugs

1. **Check existing issues** first
2. **Use issue template** with:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details

### Bug Fix Process

```bash
# Create bug fix branch
git checkout -b fix/bug-description

# Add test that reproduces bug
# Fix the bug
# Ensure test passes
# Submit PR
```

## 🎯 Performance Guidelines

### Performance Testing

```bash
# Run performance tests
poetry run pytest --benchmark-only

# Profile memory usage
poetry run pytest --profile

# Test with Docker
make docker-test
```

### Optimization Guidelines

- **Measure first**: Profile before optimizing
- **Test performance**: Include performance tests
- **Monitor resources**: Use monitoring tools
- **Document changes**: Explain performance improvements

## 📋 Release Process

### Version Management

- **Semantic versioning**: MAJOR.MINOR.PATCH
- **Changelog**: Update CHANGELOG.md
- **Tagging**: Use `v1.0.0` format
- **Docker images**: Built automatically on releases

### Release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Docker images build successfully
- [ ] Security scan passes

## 🤝 Community Guidelines

### Code of Conduct

- **Be respectful**: Treat everyone with respect
- **Be constructive**: Provide helpful feedback
- **Be collaborative**: Work together effectively
- **Be inclusive**: Welcome diverse perspectives

### Communication

- **Issues**: Use GitHub Issues for bugs and features
- **Discussions**: Use GitHub Discussions for questions
- **Security**: Use GitHub Security Advisories for security issues

## 📚 Additional Resources

### Development Tools

- **IDE**: VS Code with Python extension
- **Debugging**: pdb, ipdb for debugging
- **Profiling**: cProfile, memory_profiler
- **Testing**: pytest, pytest-cov, pytest-benchmark

### Learning Resources

- **FastAPI**: [FastAPI Documentation](https://fastapi.tiangolo.com/)
- **Poetry**: [Poetry Documentation](https://python-poetry.org/docs/)
- **Docker**: [Docker Documentation](https://docs.docker.com/)
- **pytest**: [pytest Documentation](https://docs.pytest.org/)

## 🎉 Thank You!

Thank you for contributing to Character AI! Your contributions help make this project better for everyone.

---

**Questions?** Feel free to open an issue or start a discussion!
