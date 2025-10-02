# Multi-stage Dockerfile for Character AI Platform
# Stage 1: Base image with system dependencies
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Create non-root user
RUN groupadd -r cai && useradd -r -g cai cai

# Stage 2: Dependencies builder
FROM base as builder

# Set build environment
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install Poetry and dependencies
RUN pip install poetry==1.7.1 && \
    poetry config virtualenvs.create false && \
    poetry install --only=main --no-dev

# Stage 3: Runtime image
FROM base as runtime

# Set runtime environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CAI_PATHS__MODELS_DIR=/app/models \
    CAI_MODELS__LLAMA_BACKEND=llama_cpp \
    CAI_ENVIRONMENT=production

# Create application directories
RUN mkdir -p /app/models /app/data /app/logs /app/tmp
RUN chown -R cai:cai /app

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy model bundle from a prebuilt models image (optional for now)
# COPY --from=ghcr.io/parrenthesis/cai-models:latest /models_bundle.tar.gz /app/
# COPY --from=ghcr.io/parrenthesis/cai-models:latest /manifest.json /app/

# Copy application code
COPY --chown=cai:cai . /app
WORKDIR /app

# Install application with new architecture dependencies
RUN pip install --no-cache-dir -e .[llama_cpp,tts,audio,ml] --no-deps

# Install secure dependencies (PyTorch 2.8.0+ for security)
RUN pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 --force-reinstall && \
    pip install --no-cache-dir numpy==1.24.3 cryptography==41.0.7 PyJWT==2.8.0 --force-reinstall && \
    pip cache purge

# Install model bundle into /app/models and verify checksums (optional)
RUN python scripts/install_model_bundle.py || echo "No model bundle found, skipping installation"

# Create healthcheck script
RUN printf '#!/bin/bash\ncurl -f http://localhost:8000/health || exit 1\n' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh

# Switch to non-root user
USER cai

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /app/healthcheck.sh

# Default command
CMD ["python", "-m", "character.ai.web.toy_api"]
