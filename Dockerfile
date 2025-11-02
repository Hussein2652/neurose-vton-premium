# syntax=docker/dockerfile:1.6
# GPU-first base with Torch preinstalled to avoid re-downloading
ARG BASE_IMAGE=pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_CACHE_DIR=/root/.cache/pip \
    TORCH_HOME=/app/runtime_cache/torch \
    HF_HOME=/app/runtime_cache/hf

WORKDIR /app

# Copy lockfile first so dependency layer is independent of source changes
COPY requirements.lock /app/requirements.lock

## Torch is already present in the base image; install system deps and project deps only

# Create runtime dirs (separate from pip cache mount)
RUN mkdir -p /app/runtime_cache /app/outputs

# Upgrade pip using persistent pip cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip

# Install system deps (cached layer): compilers + runtime libs for CV/ONNX builds
# Use build cache for apt metadata and packages
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential \
      gcc g++ make cmake git \
      ninja-build \
      libgl1 libglib2.0-0 \
      pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install runtime deps from lock file (torch stays from base image)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /app/requirements.lock

# Copy source code and scripts (done after deps to preserve cache)
COPY neurose_vton /app/neurose_vton
COPY scripts /app/scripts
COPY pyproject.toml README.md /app/

# Install the project itself without touching deps (so torch isn't reinstalled)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e . --no-deps

# External bind mounts at runtime (read-only for models/code; read-write for outputs/cache)
VOLUME ["/app/storage", "/app/manual_downloads", "/app/third_party", "/app/outputs", "/app/runtime_cache"]

EXPOSE 8000

CMD ["./scripts/dev_server.sh"]
