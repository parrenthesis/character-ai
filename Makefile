# Character AI Makefile
# Focused on tests, bundling models, and building runtime image

.PHONY: help test test-dev clean check-system-deps install-system-deps-instructions setup-python-version setup setup-dev setup-ci bundle models-image runtime-image run-api security lint lint-dev format format-dev docker-build docker-run docker-test docker-clean docker-dev docker-compose-up docker-compose-down test-voice-pipeline-list test-voice-pipeline-stt test-voice-pipeline-llm test-voice-pipeline-tts test-voice-pipeline-single test-voice-pipeline-all test-voice-pipeline-realtime test-voice-pipeline-realtime-desktop test-voice-pipeline-realtime-pi test-voice-pipeline-benchmark test-performance-benchmarks test-performance-benchmarks-direct test-voice-pipeline-suite

# Default target
help:
	@echo "Character AI - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  check-system-deps            - Check if system dependencies are installed"
	@echo "  install-system-deps-instructions - Show installation commands for system dependencies"
	@echo "  setup-python-version - Set up Python version for the project (creates .python-version)"
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
	@echo "Voice Pipeline Testing:"
	@echo "  test-voice-pipeline-list     - List available input files for Data"
	@echo "  test-voice-pipeline-stt      - Test STT component only"
	@echo "  test-voice-pipeline-llm      - Test LLM component only"
	@echo "  test-voice-pipeline-tts      - Test TTS component only"
	@echo "  test-voice-pipeline-single   - Test single file (what_are_you_doing.wav)"
	@echo "  test-voice-pipeline-all      - Test all available input files"
	@echo "  test-voice-pipeline-realtime - Test real-time voice interaction (10s)"
	@echo "  test-voice-pipeline-realtime-desktop - Test real-time with desktop hardware profile"
	@echo "  test-voice-pipeline-realtime-pi - Test real-time with Raspberry Pi hardware profile"
	@echo "  test-voice-pipeline-benchmark - Run performance benchmark test"
	@echo "  test-performance-benchmarks - Run automated performance benchmarks (pytest)"
	@echo "  test-performance-benchmarks-direct - Run performance benchmarks directly"
	@echo "  test-voice-pipeline-suite    - Run all voice pipeline tests in sequence"
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
	# Set environment variable to fix XTTS v2 compatibility with PyTorch 2.8
	export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
	# Run tests from tests/ directory
	CAI_ENABLE_API_STARTUP=0 PYTHONUNBUFFERED=1 PYTHONWARNINGS="ignore::RuntimeWarning:sys:1,ignore::RuntimeWarning:unittest.mock,ignore::RuntimeWarning:tracemalloc" poetry run pytest -v -p pytest_asyncio tests/

test-dev:
	@echo "Running development test suite..."
	# Set environment variable to fix XTTS v2 compatibility with PyTorch 2.8
	export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
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
check-system-deps:
	@echo "Checking system dependencies..."
	@MISSING=""; \
	command -v ffmpeg > /dev/null 2>&1 || MISSING="$$MISSING ffmpeg"; \
	command -v pkg-config > /dev/null 2>&1 || MISSING="$$MISSING pkg-config"; \
	if [ -n "$$MISSING" ]; then \
		echo "⚠️  Missing system dependencies:$$MISSING"; \
		echo "Run 'make install-system-deps-instructions' for install commands"; \
		exit 1; \
	else \
		echo "✅ All required system dependencies found"; \
	fi

install-system-deps-instructions:
	@echo "========================================="
	@echo "System Dependencies Installation Guide"
	@echo "========================================="
	@echo ""
	@echo "Required packages: ffmpeg, portaudio19-dev, libsndfile1, libasound2-dev"
	@echo ""
	@echo "📋 Ubuntu/Debian:"
	@echo "  sudo apt-get update"
	@echo "  sudo apt-get install -y ffmpeg portaudio19-dev libsndfile1 libasound2-dev"
	@echo ""
	@echo "📋 macOS (Homebrew):"
	@echo "  brew install ffmpeg portaudio libsndfile"
	@echo ""
	@echo "📋 CI/CD (GitHub Actions - add to workflow):"
	@echo "  - name: Install system dependencies"
	@echo "    run: |"
	@echo "      sudo apt-get update"
	@echo "      sudo apt-get install -y ffmpeg portaudio19-dev libsndfile1"
	@echo ""
	@echo "📋 Docker (add to Dockerfile):"
	@echo "  RUN apt-get update && apt-get install -y ffmpeg portaudio19-dev libsndfile1"
	@echo ""

setup-python-version:
	@echo "Setting up Python version for the project..."
	@if [ ! -f .python-version ]; then \
		if [ -d "$$HOME/.pyenv" ]; then \
			echo "Creating .python-version file..."; \
			export PYENV_ROOT="$$HOME/.pyenv"; \
			export PATH="$$PYENV_ROOT/bin:$$PATH"; \
			eval "$$(pyenv init -)"; \
			if pyenv versions | grep -q "3.10"; then \
				echo "3.10.12" > .python-version; \
				echo "Created .python-version with 3.10.12"; \
			elif pyenv versions | grep -q "3.11"; then \
				echo "3.11.7" > .python-version; \
				echo "Created .python-version with 3.11.7"; \
			else \
				echo "No compatible Python version found in pyenv"; \
				echo "Please install Python 3.10+ with: pyenv install 3.10.12"; \
				exit 1; \
			fi; \
		else \
			echo "pyenv not found. Please install pyenv or use system Python."; \
			echo "For system Python, ensure you have Python 3.10+ installed."; \
		fi; \
	else \
		echo ".python-version already exists: $$(cat .python-version)"; \
	fi

setup: setup-python-version
	@echo "Setting up environment with secure architecture (PyTorch 2.8.0 + Wav2Vec2 + Coqui TTS)..."
	@make check-system-deps 2>/dev/null || { \
		echo ""; \
		echo "⚠️  Some system dependencies are missing!"; \
		echo "TTS speed control and audio I/O may not work correctly."; \
		echo ""; \
		echo "Run: make install-system-deps-instructions"; \
		echo ""; \
	}
	# Set up pyenv environment
	@if [ -d "$$HOME/.pyenv" ]; then \
		echo "Using pyenv for Python version management..."; \
		export PYENV_ROOT="$$HOME/.pyenv"; \
		export PATH="$$PYENV_ROOT/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
		if [ -f .python-version ]; then \
			echo "Using Python version from .python-version: $$(cat .python-version)"; \
			poetry env use $$(cat .python-version); \
		else \
			echo "No .python-version found, using system Python"; \
			poetry env use python; \
		fi; \
	else \
		echo "pyenv not found, using system Python"; \
		poetry env use python; \
	fi
	# Install main dependencies
	poetry install --only=main
	# Set environment variable to fix XTTS v2 compatibility with PyTorch 2.8
	export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
	# Install problematic packages that conflict with TTS
	poetry run pip install numpy==1.22.0 scipy>=1.11.2 --force-reinstall
	# Install llama-cpp-python (requires build tools, can't be in pyproject.toml)
	poetry run pip install llama-cpp-python --force-reinstall --no-cache-dir
	# Fix numpy version after llama-cpp-python installs wrong version
	poetry run pip install numpy==1.22.0 --force-reinstall --no-deps

setup-dev: setup-python-version
	@echo "Setting up development environment (minimal for disk space)..."
	@make check-system-deps 2>/dev/null || { \
		echo ""; \
		echo "⚠️  Some system dependencies are missing!"; \
		echo "TTS speed control and audio I/O may not work correctly."; \
		echo ""; \
		echo "Run: make install-system-deps-instructions"; \
		echo ""; \
	}
	# Set up pyenv environment
	@if [ -d "$$HOME/.pyenv" ]; then \
		echo "Using pyenv for Python version management..."; \
		export PYENV_ROOT="$$HOME/.pyenv"; \
		export PATH="$$PYENV_ROOT/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
		if [ -f .python-version ]; then \
			echo "Using Python version from .python-version: $$(cat .python-version)"; \
			poetry env use $$(cat .python-version); \
		else \
			echo "No .python-version found, using system Python"; \
			poetry env use python; \
		fi; \
	else \
		echo "pyenv not found, using system Python"; \
		poetry env use python; \
	fi
	# Set environment variable to fix XTTS v2 compatibility with PyTorch 2.8
	export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
	# Install all dependencies (main + dev) - pyproject.toml handles versions
	poetry install
	# Install problematic packages that conflict with TTS
	poetry run pip install numpy==1.22.0 scipy>=1.11.2 --force-reinstall
	# Install llama-cpp-python (requires build tools, can't be in pyproject.toml)
	poetry run pip install llama-cpp-python --force-reinstall --no-cache-dir
	# Fix numpy version after llama-cpp-python installs wrong version
	poetry run pip install numpy==1.22.0 --force-reinstall --no-deps
	# Skip pre-commit install to avoid git issues
	@echo "Skipping pre-commit install to avoid disk space issues"

# CI-optimized setup that skips heavy PyTorch reinstalls
setup-ci: setup-python-version
	@echo "Setting up CI environment (optimized for GitHub Actions)..."
	@make check-system-deps || { \
		echo "❌ System dependencies missing in CI environment"; \
		echo "Add to workflow: sudo apt-get install -y ffmpeg portaudio19-dev libsndfile1"; \
		exit 1; \
	}
	# Set up pyenv environment (if available)
	@if [ -d "$$HOME/.pyenv" ]; then \
		echo "Using pyenv for Python version management..."; \
		export PYENV_ROOT="$$HOME/.pyenv"; \
		export PATH="$$PYENV_ROOT/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
		if [ -f .python-version ]; then \
			echo "Using Python version from .python-version: $$(cat .python-version)"; \
			poetry env use $$(cat .python-version); \
		else \
			echo "No .python-version found, using system Python"; \
			poetry env use python; \
		fi; \
	else \
		echo "pyenv not found, using system Python"; \
		poetry env use python; \
	fi
	# Set environment variable to fix XTTS v2 compatibility with PyTorch 2.8
	export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
	# Install only essential dependencies to save disk space
	poetry install --only main
	# Clean up disk space aggressively before installing dev tools
	@echo "Cleaning disk space before dev tools installation..."
	pip cache purge || true
	find ~/.cache -type f -name "*.pyc" -delete || true
	find ~/.cache -type d -name "__pycache__" -exec rm -rf {} + || true
	# Clean up any large files that might be consuming space
	find /home/runner -name "*.log" -size +1M -delete || true
	find /home/runner -name "*.tmp" -delete || true
	# Install essential dev tools for CI
	poetry run pip install pytest pytest-asyncio pytest-cov black isort ruff mypy bandit safety --no-cache-dir
	# Install missing dependencies for tests and linting
	poetry run pip install PyJWT cryptography types-PyYAML types-requests types-psutil --no-cache-dir
	# Fix numpy vulnerability - upgrade to 1.22.2
	poetry run pip install numpy==1.22.2 --force-reinstall --no-cache-dir
	# Clean up after installation
	pip cache purge || true
	# Skip pre-commit install in CI to avoid git issues
	@echo "Skipping pre-commit install in CI environment"

# Security and Quality
security:
	@echo "Running security checks..."
	poetry run bandit -r src/ --severity-level medium
	poetry run safety check --short-report --continue-on-error
	poetry run detect-secrets scan --baseline .secrets.baseline

lint:
	@echo "Running linting checks..."
	@echo "Auto-fixing issues..."
	poetry run ruff check src/ tests/ --fix
	poetry run black --check src/ tests/
	poetry run isort --check-only src/ tests/
	poetry run ruff check src/ tests/
	cd src && poetry run mypy . --ignore-missing-imports --install-types --non-interactive
	cd tests && poetry run mypy . --ignore-missing-imports --install-types --non-interactive

lint-dev:
	@echo "Running development linting checks..."
	@echo "Auto-fixing issues..."
	poetry run ruff check src/ tests_dev/ --fix
	poetry run black --check src/ tests_dev/
	poetry run isort --check-only src/ tests_dev/
	poetry run ruff check src/ tests_dev/
	cd src && poetry run mypy . --ignore-missing-imports --install-types --non-interactive
	cd tests_dev && poetry run mypy . --ignore-missing-imports --install-types --non-interactive

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

# Model management
download-models:
	@echo "Downloading STT/TTS models locally..."
	poetry run python scripts/download_models.py

verify-models:
	@echo "Verifying local model availability..."
	poetry run python scripts/verify_models.py

setup-offline: setup-dev
	@echo "Setting up fully offline environment..."
	@$(MAKE) verify-models
	@echo ""
	@echo "✅ Offline operation ready"

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

# Voice Pipeline Testing Commands
# Environment variable overrides for testing:
# - CAI_MAX_CPU_THREADS=2: Limit CPU usage to avoid maxing out development machine
# - CAI_ENABLE_CPU_LIMITING=true: Enable CPU limiting (auto-enabled in testing mode)
# - CAI_ENVIRONMENT=testing: Use testing-specific configurations
# Ensure AudioBox is ready before running voice tests
setup-audiobox:
	@echo "Setting up AudioBox for voice testing..."
	@# Only restart PulseAudio if it's completely broken (not just not responding)
	@pactl info >/dev/null 2>&1 || (echo "PulseAudio not responding, restarting..." && pulseaudio --kill && sleep 2 && pulseaudio --start && sleep 1)
	@# Load AudioBox sink if not already loaded (avoid duplicates)
	@pactl list sinks short | grep -q "alsa_output.hw_3_0" || (echo "Loading AudioBox sink..." && pactl load-module module-alsa-sink device=hw:3,0)
	@# Load AudioBox source if not already loaded (avoid duplicates)
	@pactl list sources short | grep -q "alsa_input.hw_3_0" || (echo "Loading AudioBox source..." && pactl load-module module-alsa-source device=hw:3,0)
	@# Only change default sink if AudioBox is not already default
	@pactl get-default-sink | grep -q "alsa_output.hw_3_0" || (echo "Setting AudioBox as default..." && pactl set-default-sink alsa_output.hw_3_0)
	@# Set AudioBox as default source
	@pactl get-default-source | grep -q "alsa_input.hw_3_0" || (echo "Setting AudioBox as default source..." && pactl set-default-source alsa_input.hw_3_0)
	@# Unsuspend the sink if it's suspended
	@pactl list sinks short | grep "alsa_output.hw_3_0.*SUSPENDED" && (echo "Unsuspending AudioBox..." && pactl suspend-sink alsa_output.hw_3_0 false) || true
	@# Unsuspend the source if it's suspended
	@pactl list sources short | grep "alsa_input.hw_3_0.*SUSPENDED" && (echo "Unsuspending AudioBox source..." && pactl suspend-source alsa_input.hw_3_0 false) || true
	@# Verify setup
	@pactl list sinks short | grep -q "alsa_output.hw_3_0" && pactl list sources short | grep -q "alsa_input.hw_3_0" && echo "✅ AudioBox setup complete" || echo "⚠️  AudioBox setup may have issues"
test-voice-pipeline-list: setup-audiobox
	@echo "Listing available input files for Data..."
	CAI_MAX_CPU_THREADS=2 CAI_ENABLE_CPU_LIMITING=true CAI_ENVIRONMENT=testing \
	TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
	poetry run cai test voice-pipeline --character data --franchise star_trek --list-inputs

test-voice-pipeline-stt: setup-audiobox
	@echo "Testing STT component only..."
	CAI_MAX_CPU_THREADS=2 CAI_ENABLE_CPU_LIMITING=true CAI_ENVIRONMENT=testing \
	TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
	poetry run cai test voice-pipeline --character data --franchise star_trek --test-stt-only --input tests_dev/audio_samples/star_trek/data/input/what_are_you_doing.wav

test-voice-pipeline-llm: setup-audiobox
	@echo "Testing LLM component only..."
	CAI_MAX_CPU_THREADS=2 CAI_ENABLE_CPU_LIMITING=true CAI_ENVIRONMENT=testing \
	TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
	poetry run cai test voice-pipeline --character data --franchise star_trek --test-llm-only --input tests_dev/audio_samples/star_trek/data/input/what_are_you_doing.wav

test-voice-pipeline-tts: setup-audiobox
	@echo "Testing TTS component only..."
	CAI_MAX_CPU_THREADS=2 CAI_ENABLE_CPU_LIMITING=true CAI_ENVIRONMENT=testing \
	TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
	poetry run cai test voice-pipeline --character data --franchise star_trek --test-tts-only --input tests_dev/audio_samples/star_trek/data/input/what_are_you_doing.wav

test-voice-pipeline-single: setup-audiobox
	@echo "Testing single file (what_are_you_doing.wav)..."
	CAI_MAX_CPU_THREADS=2 CAI_ENABLE_CPU_LIMITING=true CAI_ENVIRONMENT=testing \
	TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
	poetry run cai test voice-pipeline --character data --franchise star_trek --input tests_dev/audio_samples/star_trek/data/input/what_are_you_doing.wav

test-voice-pipeline-all: setup-audiobox
	@echo "Testing all available input files..."
	CAI_MAX_CPU_THREADS=2 CAI_ENABLE_CPU_LIMITING=true CAI_ENVIRONMENT=testing \
	TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
	poetry run cai test voice-pipeline --character data --franchise star_trek --test-all

test-voice-pipeline-realtime: setup-audiobox
	@echo "Testing real-time voice interaction (auto-detect hardware profile)..."
	@echo "Test will run for 10 seconds with countdown"
	CAI_MAX_CPU_THREADS=2 CAI_ENABLE_CPU_LIMITING=true CAI_ENVIRONMENT=testing \
	CAI_QUIET_MODE=0 CAI_LOG_LEVEL=DEBUG \
	TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
	poetry run cai test voice-pipeline --character data --franchise star_trek --realtime --duration 10

test-voice-pipeline-realtime-desktop: setup-audiobox
	@echo "Testing real-time voice interaction with desktop hardware profile..."
	@echo "Test will run for 10 seconds with countdown"
	CAI_MAX_CPU_THREADS=2 CAI_ENABLE_CPU_LIMITING=true CAI_ENVIRONMENT=testing \
	CAI_QUIET_MODE=1 CAI_LOG_LEVEL=DEBUG \
	TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
	poetry run cai test voice-pipeline --character data --franchise star_trek --realtime --duration 10 --hardware-profile desktop --quiet

test-voice-pipeline-realtime-pi: setup-audiobox
	@echo "Testing real-time voice interaction with Raspberry Pi hardware profile..."
	@echo "Test will run for 10 seconds with countdown"
	CAI_MAX_CPU_THREADS=2 CAI_ENABLE_CPU_LIMITING=true CAI_ENVIRONMENT=testing \
	CAI_QUIET_MODE=1 CAI_LOG_LEVEL=DEBUG \
	TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
	poetry run cai test voice-pipeline --character data --franchise star_trek --realtime --duration 10 --hardware-profile raspberry_pi --quiet

test-voice-pipeline-benchmark: setup-audiobox
	@echo "Running performance benchmark test..."
	@echo "This will test multiple components and measure latency"
	CAI_MAX_CPU_THREADS=2 CAI_ENABLE_CPU_LIMITING=true CAI_ENVIRONMENT=testing \
	TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
	poetry run cai test benchmark --character data --franchise star_trek --component all --iterations 3

test-performance-benchmarks: setup-audiobox
	@echo "Running automated performance benchmarks..."
	@echo "Testing Alexa-level latency targets:"
	@echo "  STT: <200ms, LLM: <800ms, TTS: <800ms, Total: <2s"
	CAI_MAX_CPU_THREADS=2 CAI_ENABLE_CPU_LIMITING=true CAI_ENVIRONMENT=testing \
	TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
	poetry run pytest tests_dev/test_performance_benchmarks.py -v -s

test-performance-benchmarks-direct:
	@echo "Running performance benchmarks directly..."
	@echo "This will show detailed performance metrics"
	CAI_MAX_CPU_THREADS=2 CAI_ENABLE_CPU_LIMITING=true CAI_ENVIRONMENT=testing \
	TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
	poetry run python tests_dev/test_performance_benchmarks.py

test-voice-pipeline-suite: setup-audiobox
	@echo "Running complete voice pipeline test suite..."
	@echo "=========================================="
	@echo "1. Listing available input files..."
	@$(MAKE) test-voice-pipeline-list
	@echo ""
	@echo "2. Testing STT component..."
	@$(MAKE) test-voice-pipeline-stt
	@echo ""
	@echo "3. Testing LLM component..."
	@$(MAKE) test-voice-pipeline-llm
	@echo ""
	@echo "4. Testing TTS component..."
	@$(MAKE) test-voice-pipeline-tts
	@echo ""
	@echo "5. Testing single file..."
	@$(MAKE) test-voice-pipeline-single
	@echo ""
	@echo "6. Testing all files..."
	@$(MAKE) test-voice-pipeline-all
	@echo ""
	@echo "7. Testing real-time interaction..."
	@$(MAKE) test-voice-pipeline-realtime
	@echo ""
	@echo "=========================================="
	@echo "Voice pipeline test suite completed!"
