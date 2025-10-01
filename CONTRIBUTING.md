# Contributing to Character AI

Thank you for your interest in contributing to the Character AI! This guide will help you get started with contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Poetry (for dependency management)
- Git
- Docker (optional, for testing)

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/your-username/character.ai.git
   cd character.ai
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

# Run specific test file
poetry run pytest tests/test_voice_manager.py

# Run integration tests
make test-integration
```

### Test Requirements

- All tests must pass before submitting a PR
- New features must include tests
- Aim for high test coverage
- Use descriptive test names

### Writing Tests

```python
# Example test structure
def test_character_creation():
    """Test character creation with valid data."""
    character = Character(
        name="TestBot",
        species=Species.ROBOT,
        archetype=Archetype.HELPER
    )
    
    assert character.name == "TestBot"
    assert character.species == Species.ROBOT
    assert character.archetype == Archetype.HELPER
```

## 🔍 Code Quality

### Linting and Formatting

```bash
# Run linting and formatting
make lint

# Fix linting issues automatically
make format

# Run type checking
make type-check

# Run all quality checks
make check-all
```

### Code Style Guidelines

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write docstrings for all public functions
- Use meaningful variable and function names
- Keep functions small and focused

### Example Code Style

```python
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Character:
    """Represents an interactive character."""
    
    name: str
    species: Species
    archetype: Archetype
    personality_traits: List[PersonalityTrait]
    
    def get_display_name(self) -> str:
        """Get the display name for the character."""
        return f"{self.name} ({self.species.value})"
    
    def has_trait(self, trait: PersonalityTrait) -> bool:
        """Check if character has a specific personality trait."""
        return trait in self.personality_traits
```

## 📝 Documentation

### Writing Documentation

- Update relevant documentation when adding features
- Include code examples in docstrings
- Update API documentation for new endpoints
- Add to README if needed

### Documentation Structure

```
docs/
├── api/                    # API documentation
├── deployment/             # Deployment guides
├── examples/              # Code examples
└── development/           # Development guides
```

## 🐛 Bug Reports

### Before Reporting

1. Check if the issue already exists
2. Try to reproduce the issue
3. Check the logs for error messages
4. Test with the latest version

### Bug Report Template

```markdown
## Bug Description
Brief description of the bug.

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Ubuntu 20.04]
- Python Version: [e.g., 3.10.12]
- Platform Version: [e.g., 1.0.0]

## Additional Context
Any other context about the problem.
```

## ✨ Feature Requests

### Before Requesting

1. Check if the feature already exists
2. Consider if it fits the project's scope
3. Think about implementation complexity
4. Consider backward compatibility

### Feature Request Template

```markdown
## Feature Description
Brief description of the feature.

## Use Case
Why is this feature needed?

## Proposed Solution
How would you like this to work?

## Alternatives Considered
What other approaches have you considered?

## Additional Context
Any other context about the feature request.
```

## 🔄 Pull Request Process

### Before Submitting

1. **Ensure tests pass**
   ```bash
   make test
   ```

2. **Run linting and quality checks**
   ```bash
   make check-all
   ```

3. **Update documentation** if needed

4. **Add changelog entry** if applicable

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Integration tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Changelog updated
```

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** in staging environment
4. **Approval** from maintainers

## 🏗️ Architecture Guidelines

### Adding New Components

1. **Follow the existing structure**
   ```
   src/character_ai/
   ├── core/           # Core functionality
   ├── characters/     # Character management
   ├── algorithms/     # AI algorithms
   ├── web/           # Web API
   └── cli/           # Command line interface
   ```

2. **Use dependency injection** where possible
3. **Follow the protocol pattern** for interfaces
4. **Include comprehensive tests**

### Example Component Structure

```python
# src/character_ai/new_feature/
from typing import Protocol, Optional
from dataclasses import dataclass

class NewFeatureProtocol(Protocol):
    """Protocol for new feature implementation."""
    
    async def process(self, data: str) -> str:
        """Process the data."""
        ...

@dataclass
class NewFeatureConfig:
    """Configuration for new feature."""
    
    enabled: bool = True
    timeout_seconds: int = 30

class NewFeature:
    """Implementation of new feature."""
    
    def __init__(self, config: NewFeatureConfig):
        self.config = config
    
    async def process(self, data: str) -> str:
        """Process the data."""
        if not self.config.enabled:
            return data
        
        # Implementation here
        return processed_data
```

## 🔒 Security

### Security Guidelines

- **Never commit secrets** or API keys
- **Use environment variables** for configuration
- **Validate all inputs** from external sources
- **Follow secure coding practices**
- **Report security issues** privately

### Reporting Security Issues

If you discover a security vulnerability, please report it privately:

1. **GitHub Security Advisories**: Use GitHub's private vulnerability reporting
2. **Include**: Detailed description and steps to reproduce
3. **Do not**: Create public issues for security vulnerabilities

## 📊 Performance

### Performance Guidelines

- **Profile code** before optimizing
- **Use async/await** for I/O operations
- **Cache expensive operations** when appropriate
- **Monitor memory usage** in long-running processes
- **Write performance tests** for critical paths

### Example Performance Test

```python
import pytest
import time

def test_character_processing_performance():
    """Test that character processing meets performance requirements."""
    start_time = time.time()
    
    # Process character
    result = process_character(character_data)
    
    processing_time = time.time() - start_time
    
    # Should complete within 100ms
    assert processing_time < 0.1
    assert result is not None
```

## 🎯 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Release notes written
- [ ] Tagged in Git

## 🤝 Community

### Getting Help

- **GitHub Discussions**: For questions and discussions
- **GitHub Issues**: For bug reports and feature requests
- **Documentation**: Check the docs/ directory

### Code of Conduct

We follow the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Please read and follow it.

## 📚 Resources

### Useful Links

- [Python Style Guide](https://pep8.org/)
- [Type Hints](https://docs.python.org/3/library/typing.html)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [Pytest Documentation](https://docs.pytest.org/)

### Development Tools

- **IDE**: VS Code with Python extension
- **Linting**: Ruff, MyPy, Black
- **Testing**: Pytest with coverage
- **Documentation**: Sphinx

## 🙏 Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **GitHub contributors** page

Thank you for contributing to the Character AI! 🚀
