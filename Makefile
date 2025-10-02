# Character AI Makefile
# Focused on tests, bundling models, and building runtime image

.PHONY: help test test-dev clean setup setup-dev setup-ci bundle models-image runtime-image run-api security lint lint-dev format format-dev docker-build docker-run docker-test docker-clean docker-dev docker-compose-up docker-compose-down

# Default target
help:
	@echo "Character AI - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  setup               - Set up environment with security dependencies"
	@echo "  setup-dev           - Set up development environment (includes setup)"
	@echo "  setup-ci            - Set up CI environment (optimized for GitHub Actions)"
	@echo ""
	@echo "Development:"
	@echo "  test                - Run tests"
	@echo "  test-dev            - Run development test suite"
	@echo "  clean               - Clean build artifacts and cache"
	@echo "  security            - Run security checks (bandit, safety, detect-secrets)"
	@echo "  lint                 - Run linting checks (black, isort, ruff, mypy)"
	@echo "  lint-dev            - Run development linting checks"
	@echo "  format               - Format code (black, isort)"
	@echo "  format-dev          - Format development code"
	@echo ""
	@echo "Production:"
	@echo "  validate-prod        - Validate production environment"
	@echo "  smoke-test           - Run production smoke tests"
	@echo "  validate-env         - Validate environment variables"
	@echo "  optimize-ci          - Analyze and optimize CI/CD performance"
	@echo ""
	@echo "Docker Development:"
	@echo "  docker-build         - Build runtime Docker image locally"
	@echo "  docker-dev           - Build development Docker image (faster)"
	@echo "  docker-run           - Run Docker container locally"
	@echo "  docker-test          - Run tests inside Docker container"
	@echo "  docker-lint          - Run linting inside Docker container"
	@echo "  docker-clean         - Clean up Docker resources"
	@echo "  docker-compose-up    - Start full stack with docker-compose"
	@echo "  docker-compose-down  - Stop docker-compose services"
	@echo ""
	@echo "Models & Docker:"
	@echo "  bundle              - Build models bundle (models_bundle.tar.gz)"
	@echo "  models-image        - Build Docker models image (Dockerfile.models)"
	@echo "  runtime-image       - Build runtime image (Dockerfile) using models image"
	@echo "  run-api             - Run API locally (uvicorn)"
	@echo "  validate-profile    - Validate a character profile folder"
	@echo "  generate-schemas    - Export JSON Schemas for profiles/index/consent"
	@echo "  generate-api-docs    - Generate OpenAPI documentation"
	@echo "  recompute-embeddings- Recompute voice embeddings (CHAR=all or char name)"
	@echo ""

test:
	@echo "Running tests for GitHub CI..."
	# Run tests from tests/ directory
	CAI_ENABLE_API_STARTUP=0 PYTHONUNBUFFERED=1 PYTHONWARNINGS="ignore::RuntimeWarning:sys:1,ignore::RuntimeWarning:unittest.mock,ignore::RuntimeWarning:tracemalloc" poetry run pytest -v -p pytest_asyncio tests/

test-dev:
	@echo "Running development test suite..."
	# Run test suite from tests_dev/ directory
	CAI_ENABLE_API_STARTUP=0 PYTHONUNBUFFERED=1 PYTHONWARNINGS="ignore::RuntimeWarning:sys:1,ignore::RuntimeWarning:unittest.mock,ignore::RuntimeWarning:tracemalloc" poetry run pytest -v -p pytest_asyncio -m "not integration" tests_dev/
	@echo "Running integration tests..."
	CAI_ENABLE_API_STARTUP=0 PYTHONUNBUFFERED=1 PYTHONWARNINGS="ignore::RuntimeWarning:sys:1,ignore::RuntimeWarning:unittest.mock,ignore::RuntimeWarning:tracemalloc" poetry run pytest -v -p pytest_asyncio -m integration tests_dev/

test-coverage:
	@echo "Running tests with coverage..."
	CAI_ENABLE_API_STARTUP=0 PYTHONUNBUFFERED=1 PYTHONWARNINGS="ignore::RuntimeWarning:sys:1,ignore::RuntimeWarning:unittest.mock,ignore::RuntimeWarning:tracemalloc" poetry run pytest --cov=src/character_ai --cov-report=html --cov-report=term-missing -v -p pytest_asyncio tests/

clean:
	@echo "Cleaning build artifacts and cache..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .tox/
	rm -rf .cache/
	rm -rf .venv/
	rm -f poetry.lock
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "Cleanup complete!"

# Docker Development Commands
docker-build:
	@echo "Building runtime Docker image locally..."
	docker build -t character-ai:latest .

docker-dev:
	@echo "Building development Docker image (faster, with dev dependencies)..."
	docker build -t character-ai:dev \
		--build-arg BUILD_ENV=development \
		--target builder \
		.

docker-run:
	@echo "Running Docker container locally..."
	docker run --rm -p 8000:8000 \
		-e CAI_ENVIRONMENT=development \
		-e CAI_API__HOST=0.0.0.0 \
		-e CAI_API__PORT=8000 \
		character-ai:latest

docker-test:
	@echo "Running tests inside Docker container..."
	docker run --rm character-ai:latest \
		python -m pytest tests/ -v

docker-lint:
	@echo "Running linting inside Docker container..."
	docker run --rm character-ai:latest \
		sh -c "make lint"

docker-clean:
	@echo "Cleaning up Docker resources..."
	docker system prune -f
	docker builder prune -f
	docker image prune -f
	@echo "Docker cleanup complete!"

docker-compose-up:
	@echo "Starting full stack with docker-compose..."
	docker-compose up -d
	@echo "Services started. Check status with: docker-compose ps"

docker-compose-down:
	@echo "Stopping docker-compose services..."
	docker-compose down
	@echo "Services stopped."

# Setup targets
setup:
	@echo "Setting up environment with secure architecture (PyTorch 2.8.0 + Wav2Vec2 + Coqui TTS)..."
	export PATH="$$HOME/.pyenv/bin:$$PATH" && eval "$$(pyenv init -)" && poetry env use python && poetry install --only=main
	poetry run pip install torch>=2.8.0 torchaudio>=2.8.0 --force-reinstall
	poetry run pip install numpy==1.22.0 cryptography PyJWT pydantic-core==2.33.2 psutil==5.9.8 --force-reinstall
	poetry run pip install llama-cpp-python --force-reinstall --no-cache-dir
	poetry run pip install numpy==1.22.0 fsspec==2024.6.1 networkx==2.8.8 --force-reinstall
	poetry run pip install numpy==1.22.2 --force-reinstall --no-deps

setup-dev:
	@echo "Setting up development environment with secure architecture (PyTorch 2.8.0 + Wav2Vec2 + Coqui TTS)..."
	export PATH="$$HOME/.pyenv/bin:$$PATH" && eval "$$(pyenv init -)" && poetry env use python && poetry install
	poetry run pip install torch>=2.8.0 torchaudio>=2.8.0 --force-reinstall
	poetry run pip install numpy==1.22.0 cryptography PyJWT pydantic-core==2.33.2 psutil==5.9.8 --force-reinstall
	poetry run pip install llama-cpp-python --force-reinstall --no-cache-dir
	poetry run pip install numpy==1.22.0 fsspec==2024.6.1 networkx==2.8.8 --force-reinstall
	poetry run pip install numpy==1.22.2 --force-reinstall --no-deps
	poetry run pre-commit install

# CI-optimized setup that skips heavy PyTorch reinstalls
setup-ci:
	@echo "Setting up CI environment (optimized for GitHub Actions)..."
	poetry install --extras security --extras audio --extras ml
	poetry run pip install pytest pytest-asyncio pytest-cov pytest-mock pytest-benchmark black isort ruff mypy bandit detect-secrets pre-commit
	poetry run pip install numpy==1.22.2 --force-reinstall --no-cache-dir
	poetry run pre-commit install

# Security and Quality
security:
	@echo "Running security checks..."
	poetry run bandit -r src/ --severity-level medium
	poetry run safety check --short-report
	poetry run detect-secrets scan --baseline .secrets.baseline

lint:
	@echo "Running linting checks..."
	@echo "Auto-fixing issues..."
	poetry run ruff check src/ tests/ --fix
	poetry run black --check src/ tests/
	poetry run isort --check-only src/ tests/
	poetry run ruff check src/ tests/
	cd src && poetry run mypy . --ignore-missing-imports
	cd tests && poetry run mypy . --ignore-missing-imports

lint-dev:
	@echo "Running development linting checks..."
	@echo "Auto-fixing issues..."
	poetry run ruff check src/ tests_dev/ --fix
	poetry run black --check src/ tests_dev/
	poetry run isort --check-only src/ tests_dev/
	poetry run ruff check src/ tests_dev/
	cd src && poetry run mypy . --ignore-missing-imports
	cd tests_dev && poetry run mypy . --ignore-missing-imports

format:
	@echo "Formatting code..."
	poetry run black src/ tests/
	poetry run isort src/ tests/

format-dev:
	@echo "Formatting development code..."
	poetry run black src/ tests_dev/
	poetry run isort src/ tests_dev/

# Production validation targets
validate-prod:
	@echo "Validating production environment..."
	poetry run python scripts/validate_production.py

smoke-test:
	@echo "Running production smoke tests..."
	poetry run python scripts/validate_production.py

validate-env:
	@echo "Validating environment variables..."
	poetry run python scripts/validate_env_vars.py

optimize-ci:
	@echo "Analyzing CI/CD performance..."
	poetry run python scripts/optimize_ci.py

# Models and Bundling
bundle:
	@echo "Building models bundle..."
	poetry run python scripts/build_model_bundle.py

models-image:
	@echo "Building Docker models image..."
	docker build -f Dockerfile.models -t character-ai-models:latest .

runtime-image:
	@echo "Building runtime image using models image..."
	docker build -t character-ai:latest .

# API and Development
run-api:
	@echo "Starting API server..."
	poetry run uvicorn character_ai.web.toy_api:app --host 0.0.0.0 --port 8000 --reload

# Validation and Schema Generation
validate-profile:
	@echo "Validating character profile..."
	poetry run python scripts/profile_validate.py

generate-schemas:
	@echo "Generating JSON schemas..."
	poetry run python scripts/generate_schemas.py

generate-api-docs:
	@echo "Generating OpenAPI documentation..."
	poetry run python scripts/generate_openapi.py

recompute-embeddings:
	@echo "Recomputing voice embeddings..."
	poetry run python scripts/recompute_voice_embeddings.py $(CHAR)
