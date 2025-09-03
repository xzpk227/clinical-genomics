# Multi-stage Dockerfile for the Clinical Phenotype Extraction and HPO Mapping Pipeline
#
# Stage 1 (builder): installs all Python dependencies from pyproject.toml
# Stage 2 (runtime): copies only what is needed to run the API
#
# Build:
#   docker build -t clinical-phenotype-pipeline .
#
# Run:
#   docker run -p 8000:8000 clinical-phenotype-pipeline
#
# Note: data/hpo_database.json, data/hpo_index.faiss, and data/hpo_id_map.json
# must be pre-built before building the image (run build_hpo_db.py and build_index.py).

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specification first (layer caching)
COPY pyproject.toml ./

# Install CPU-only torch first to avoid pulling CUDA/cuDNN (~1GB+)
RUN pip install --upgrade pip \
    && pip install --prefix=/install \
        torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --prefix=/install ".[dev]" --no-deps \
    && pip install --prefix=/install \
        fastapi>=0.111.0 \
        "uvicorn[standard]>=0.29.0" \
        "pydantic>=2.7.0" \
        "spacy>=3.7.4" \
        "negspacy>=1.0.4" \
        "sentence-transformers>=3.0.0" \
        "faiss-cpu>=1.8.0" \
        "hypothesis>=6.100.0" \
        "pytest>=8.2.0" \
        "pytest-asyncio>=0.23.6" \
        "httpx>=0.27.0"

# ---------------------------------------------------------------------------
# Stage 2: test — runs unit and integration tests (no data artifacts needed)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS test

WORKDIR /app

# Install CPU-only torch first to avoid pulling CUDA/cuDNN (~1GB+)
RUN pip install --upgrade pip \
    && pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install \
        "transformers>=4.41.0,<5.0.0" \
        "sentence-transformers>=3.0.0,<4.0.0" \
        "fastapi>=0.111.0" \
        "uvicorn[standard]>=0.29.0" \
        "pydantic>=2.7.0" \
        "spacy>=3.7.4" \
        "negspacy>=1.0.4" \
        "faiss-cpu>=1.8.0" \
        "hypothesis>=6.100.0" \
        "pytest>=8.2.0" \
        "pytest-asyncio>=0.23.6" \
        "httpx>=0.27.0"

# Copy application source and tests
COPY src/ ./src/
COPY tests/ ./tests/
COPY config/ ./config/
COPY data/evaluation/ ./data/evaluation/
COPY pyproject.toml ./

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Install the package itself (no deps — already installed above)
RUN pip install --no-deps .

CMD ["pytest", "tests/unit/", "tests/integration/", "-v", "--tb=short"]

# ---------------------------------------------------------------------------
# Stage 3: runtime — full API image (requires pre-built data artifacts)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY config/ ./config/

# Copy pre-built data artifacts (must exist before docker build)
COPY data/hpo_database.json ./data/hpo_database.json
COPY data/hpo_index.faiss ./data/hpo_index.faiss
COPY data/hpo_id_map.json ./data/hpo_id_map.json
COPY data/evaluation/ ./data/evaluation/

# Copy package metadata so the editable install resolves correctly
COPY pyproject.toml ./

# Install the package itself (non-editable, no deps — already installed above)
RUN pip install --no-deps .

# Environment
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
