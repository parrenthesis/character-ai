# Character AI Makefile
# Focused on tests, bundling models, and building runtime image

.PHONY: help test test-dev clean setup-dev bundle models-image runtime-image run-api security lint lint-dev format format-dev

# Default target
help:
	@echo "Character AI - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  setup               - Set up environment with security dependencies"
	@echo "  setup-dev           - Set up development environment (includes setup)"
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
	@echo "  llama.cpp"
	CAI_RUN_LLAMA_CPP=1 CAI_LLAMA_GGUF=$(PWD)/models/llm/tinyllama-1.1b-q4_k_m.gguf PYTHONUNBUFFERED=1 poetry run pytest -v -m integration tests_dev/test_llm_processors.py::test_llama_cpp_basic
	@echo "  Whisper (skipped due to NumPy dependency conflicts)"
	# CAI_RUN_WHISPER=1 PYTHONUNBUFFERED=1 poetry run pytest -v -m integration tests_dev/test_whisper_processor.py
	@echo "  XTTS (skipped due to NumPy dependency conflicts)"
	# CAI_RUN_TTS=1 PYTHONUNBUFFERED=1 poetry run pytest -v -m integration tests_dev/test_xtts_processor.py
	@echo "  Engine"
	CAI_RUN_ENGINE=1 PYTHONUNBUFFERED=1 poetry run pytest -v -m integration tests_dev/test_engine.py::test_realtime_engine_end_to_end
	@echo "  Simple Integration"
	CAI_RUN_INTEGRATION=1 CAI_LLAMA_GGUF=$(PWD)/models/llm/tinyllama-1.1b-q4_k_m.gguf PYTHONUNBUFFERED=1 poetry run pytest -v -m integration tests/integration/test_simple_integration.py

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf cache/
	rm -rf logs/
	rm -rf test_audio/
	rm -rf test_output/
	rm -f catalog_Test\ Collection_*.yaml
	rm -f catalog_Test_Collection_*.yaml
	rm -f *.log
	rm -f .DS_Store
	rm -f Thumbs.db
	rm -f models_bundle.tar.gz
	rm -rf .venv/
	rm -f poetry.lock
	@echo "Cleaning development artifacts..."
	rm -f ci_performance_report.json
	rm -f env_validation_report.json
	rm -f production_validation_report.json
	rm -f coverage.json
	rm -f coverage.xml
	rm -rf bundles/
	rm -f *.tar.gz
	rm -f *.zip
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name "*.so" -delete
	find . -type f -name "*.egg" -delete

setup:
	@echo "Setting up environment with security dependencies..."
	@echo "Checking Python version compatibility..."
	@if command -v python3.10 >/dev/null 2>&1; then \
		echo "Using system Python 3.9"; \
	elif command -v python3.10 >/dev/null 2>&1; then \
		echo "Using system Python 3.10"; \
	elif command -v python3.11 >/dev/null 2>&1; then \
		echo "Using system Python 3.11"; \
	elif [ -d "$$HOME/.pyenv" ]; then \
		echo "Using existing pyenv..."; \
		export PATH="$$HOME/.pyenv/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
		pyenv local 3.10; \
	elif command -v pyenv >/dev/null 2>&1; then \
		echo "Using existing pyenv..."; \
		export PATH="$$HOME/.pyenv/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
		pyenv install 3.10 2>/dev/null || echo "Python 3.10 already installed"; \
		pyenv local 3.10; \
	else \
		echo "Installing pyenv for Python version management..."; \
		curl https://pyenv.run | bash; \
		export PATH="$$HOME/.pyenv/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
		pyenv install 3.10; \
		pyenv local 3.10; \
	fi
	@echo "Current Python version:"
	@python3 --version
	@echo "Note: Poetry can't resolve TTS + numpy + security conflicts, using pip override..."
	@echo "Removing old lock file and installing core dependencies..."
	rm -f poetry.lock
	@if command -v python3.10 >/dev/null 2>&1; then \
		poetry env use python3.10; \
	elif command -v python3.10 >/dev/null 2>&1; then \
		poetry env use python3.10; \
	elif command -v python3.11 >/dev/null 2>&1; then \
		poetry env use python3.11; \
	elif [ -d "$$HOME/.pyenv" ]; then \
		export PATH="$$HOME/.pyenv/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
		PYTHON_PATH=$$(pyenv which python3.10 2>/dev/null || echo "$$HOME/.pyenv/versions/3.10/bin/python"); \
		poetry env use $$PYTHON_PATH; \
	elif command -v pyenv >/dev/null 2>&1; then \
		export PATH="$$HOME/.pyenv/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
		PYTHON_PATH=$$(pyenv which python3.10 2>/dev/null || echo "$$HOME/.pyenv/versions/3.10/bin/python"); \
		poetry env use $$PYTHON_PATH; \
	else \
		echo "No compatible Python version found. Please install Python 3.9+ manually."; \
		exit 1; \
	fi
	poetry install --only main
	@echo "Fixing dependency conflicts (TTS + numpy + security)..."
	poetry run pip install numpy==1.24.3 cryptography PyJWT pydantic-core==2.33.2 psutil --force-reinstall
	@echo "Installing llama-cpp-python with proper build configuration..."
	poetry run pip install llama-cpp-python --force-reinstall --no-cache-dir
	@echo "Fixing tiktoken circular import issue..."
	poetry run pip install tiktoken --force-reinstall --no-cache-dir
	@echo "Fixing NumPy version conflicts (TTS requires numpy==1.22.0)..."
	poetry run pip install "numpy==1.22.0" --force-reinstall --no-cache-dir
	@echo "Fixing psutil version conflict..."
	poetry run pip install "psutil>=5.9.0,<6.0.0" --force-reinstall --no-cache-dir
	@echo "Fixing regex version conflict..."
	poetry run pip install "regex>=2023.0.0,<2024.0.0" --force-reinstall --no-cache-dir
	@echo "Done."

setup-dev:
	@echo "Setting up development environment with security dependencies..."
	@echo "Checking Python version compatibility..."
	@if command -v python3.10 >/dev/null 2>&1; then \
		echo "Using system Python 3.9"; \
	elif command -v python3.10 >/dev/null 2>&1; then \
		echo "Using system Python 3.10"; \
	elif command -v python3.11 >/dev/null 2>&1; then \
		echo "Using system Python 3.11"; \
	elif [ -d "$$HOME/.pyenv" ]; then \
		echo "Using existing pyenv..."; \
		export PATH="$$HOME/.pyenv/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
		pyenv local 3.10; \
	elif command -v pyenv >/dev/null 2>&1; then \
		echo "Using existing pyenv..."; \
		export PATH="$$HOME/.pyenv/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
		pyenv install 3.10 2>/dev/null || echo "Python 3.10 already installed"; \
		pyenv local 3.10; \
	else \
		echo "Installing pyenv for Python version management..."; \
		curl https://pyenv.run | bash; \
		export PATH="$$HOME/.pyenv/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
		pyenv install 3.10; \
		pyenv local 3.10; \
	fi
	@echo "Current Python version:"
	@python3 --version
	@echo "Note: Poetry can't resolve TTS + numpy + security conflicts, using pip override..."
	@echo "Removing old lock file and installing core dependencies..."
	rm -f poetry.lock
	@if command -v python3.10 >/dev/null 2>&1; then \
		poetry env use python3.10; \
	elif command -v python3.10 >/dev/null 2>&1; then \
		poetry env use python3.10; \
	elif command -v python3.11 >/dev/null 2>&1; then \
		poetry env use python3.11; \
	elif [ -d "$$HOME/.pyenv" ]; then \
		export PATH="$$HOME/.pyenv/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
		PYTHON_PATH=$$(pyenv which python3.10 2>/dev/null || echo "$$HOME/.pyenv/versions/3.10/bin/python"); \
		poetry env use $$PYTHON_PATH; \
	elif command -v pyenv >/dev/null 2>&1; then \
		export PATH="$$HOME/.pyenv/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
		PYTHON_PATH=$$(pyenv which python3.10 2>/dev/null || echo "$$HOME/.pyenv/versions/3.10/bin/python"); \
		poetry env use $$PYTHON_PATH; \
	else \
		echo "No compatible Python version found. Please install Python 3.9+ manually."; \
		exit 1; \
	fi
	poetry install --only main
	@echo "Fixing dependency conflicts (TTS + numpy + security)..."
	poetry run pip install numpy==1.24.3 cryptography PyJWT pydantic-core==2.33.2 psutil --force-reinstall
	@echo "Installing development dependencies..."
	poetry install --with dev
	@echo "Installing llama-cpp-python with proper build configuration..."
	poetry run pip install llama-cpp-python --force-reinstall --no-cache-dir
	@echo "Fixing tiktoken circular import issue..."
	poetry run pip install tiktoken --force-reinstall --no-cache-dir
	@echo "Fixing NumPy version conflicts (TTS requires numpy==1.22.0)..."
	poetry run pip install "numpy==1.22.0" --force-reinstall --no-cache-dir
	@echo "Fixing psutil version conflict..."
	poetry run pip install "psutil>=5.9.0,<6.0.0" --force-reinstall --no-cache-dir
	@echo "Fixing regex version conflict..."
	poetry run pip install "regex>=2023.0.0,<2024.0.0" --force-reinstall --no-cache-dir
	@echo "Development environment ready."

# Models & Docker
bundle:
	@echo "Building models bundle..."
	poetry run python scripts/build_model_bundle.py

models-image:
	@echo "Building models image..."
	docker build -f Dockerfile.models -t icp-models:latest .

runtime-image:
	@echo "Building runtime image (requires models image)..."
	docker build -t icp-runtime:latest -f Dockerfile .

run-api:
	@echo "Running API locally..."
	poetry run uvicorn character.ai.web.toy_api:app --reload --port 8000

stop-api:
	@echo "Stopping API server..."
	@pkill -f "uvicorn.*toy_api" || echo "No API server running"

validate-profile:
	@echo "Validating character profile..."
	@if [ -z "$(DIR)" ]; then echo "Usage: make validate-profile DIR=./configs/characters/sparkle"; exit 2; fi
	poetry run python scripts/profile_validate.py $(DIR)

generate-schemas:
	@echo "Generating JSON Schemas..."
	poetry run python scripts/generate_schemas.py

recompute-embeddings:
	@echo "Recomputing voice embeddings..."
	@if [ -z "$(CHAR)" ]; then \
		poetry run python scripts/recompute_voice_embeddings.py; \
	else \
		poetry run python scripts/recompute_voice_embeddings.py $(CHAR); \
	fi

generate-api-docs:
	@echo "Generating API documentation..."
	poetry run python scripts/generate_openapi.py

security:
	@echo "Running security checks..."
	poetry run bandit -r src/
	@echo "Dependency vulnerability scan..."
	-poetry run safety check || echo "Dependency vulnerabilities found - see output above for details"
	poetry run detect-secrets scan --baseline .secrets.baseline

lint:
	@echo "Running linting checks..."
	@echo "Auto-fixing issues..."
	poetry run ruff check src/ tests/ --fix
	poetry run black --check src/ tests/
	poetry run isort --check-only src/ tests/
	poetry run ruff check src/ tests/
	poetry run mypy src/

lint-dev:
	@echo "Running development linting checks..."
	@echo "Auto-fixing issues..."
	poetry run ruff check src/ tests_dev/ --fix
	poetry run black --check src/ tests_dev/
	poetry run isort --check-only src/ tests_dev/
	poetry run ruff check src/ tests_dev/
	poetry run mypy src/

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
