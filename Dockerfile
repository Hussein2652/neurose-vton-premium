# syntax=docker/dockerfile:1.6
# Minimal dev Dockerfile (real image to be hardened in later sprints)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_CACHE_DIR=/root/.cache/pip \
    TORCH_HOME=/app/runtime_cache/torch \
    HF_HOME=/app/runtime_cache/hf

WORKDIR /app

# Copy only dependency metadata first to leverage Docker layer caching
COPY pyproject.toml /app/

# System deps (cached layer): compilers + runtime libs for CV/ONNX builds
# Use build cache for apt metadata and packages
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential \
      gcc g++ make cmake git \
      libgl1 libglib2.0-0 \
      pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create runtime dirs and install dependencies at build time (deps cached in layers)
RUN mkdir -p /app/runtime_cache /app/outputs && \
    --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip

# Allow selecting PyTorch channel (CPU by default). Override at build with:
#   --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
# Install PyTorch first from the chosen channel (with pip cache mount)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url ${PYTORCH_INDEX_URL} torch torchvision

# Install runtime deps explicitly to avoid re-resolving torch later
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    fastapi>=0.110 \
    "uvicorn[standard]">=0.24 \
    python-multipart>=0.0.6 \
    pillow>=10.0 \
    tqdm>=4.66 \
    opencv-python-headless>=4.8 \
    ultralytics>=8.0.0 \
    insightface>=0.7.3 \
    onnxruntime-gpu>=1.16.0 \
    smplx>=0.1.28

# Copy source code and scripts (done after deps to preserve cache)
COPY neurose_vton /app/neurose_vton
COPY scripts /app/scripts
COPY README.md /app/

# Install the project itself without touching deps (so torch isn't reinstalled)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e . --no-deps

# External bind mounts at runtime (read-only for models/code; read-write for outputs/cache)
VOLUME ["/app/storage", "/app/manual_downloads", "/app/third_party", "/app/outputs", "/app/runtime_cache"]

EXPOSE 8000

CMD ["./scripts/dev_server.sh"]
