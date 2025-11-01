# Minimal dev Dockerfile (real image to be hardened in later sprints)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_CACHE_DIR=/root/.cache/pip \
    TORCH_HOME=/app/runtime_cache/torch \
    HF_HOME=/app/runtime_cache/hf

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY neurose_vton /app/neurose_vton
COPY scripts /app/scripts

# Create runtime dirs and install dependencies at build time
RUN mkdir -p /app/runtime_cache /app/outputs && \
    pip install --upgrade pip && \
    pip install -e .[api]

# External bind mounts at runtime (read-only for models/code; read-write for outputs/cache)
VOLUME ["/app/storage", "/app/manual_downloads", "/app/third_party", "/app/outputs", "/app/runtime_cache"]

EXPOSE 8000

CMD ["./scripts/dev_server.sh"]
