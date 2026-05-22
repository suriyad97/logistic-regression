# ── Stage 1: build ────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# Build tools needed for shap (C++ extension)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# curl needed for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application source
COPY src/     ./src/
COPY scripts/ ./scripts/
COPY config/  ./config/

# Model artifacts are baked directly into the Docker image
COPY model/ /model/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MODEL_PATH=/model/model.pkl
ENV PREPROCESSOR_PATH=/model/preprocessor.pkl
ENV PORT=8000
ENV PATH=/root/.local/bin:$PATH

EXPOSE ${PORT}

# Liveness probe — FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# uvicorn: 2 workers, async, production mode
CMD ["sh", "-c", "uvicorn scripts.app:app --host 0.0.0.0 --port ${PORT} --workers 2 --log-level info"]
