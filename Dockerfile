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

# System deps (cached layer): compilers + runtime libs for CV/ONNX builds
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential \
      gcc g++ make cmake git \
      libgl1 libglib2.0-0 \
      pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create runtime dirs and install dependencies at build time (deps cached in layers)
RUN mkdir -p /app/runtime_cache /app/outputs && \
    pip install --upgrade pip

# Allow selecting PyTorch channel (CPU by default). Override at build with:
#   --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --index-url ${PYTORCH_INDEX_URL} torch torchvision

# Install project extras (API + Person Analysis)
RUN pip install -e .[api,person]

# Copy source code and scripts (done after deps to preserve cache)
COPY neurose_vton /app/neurose_vton
COPY scripts /app/scripts

# External bind mounts at runtime (read-only for models/code; read-write for outputs/cache)
VOLUME ["/app/storage", "/app/manual_downloads", "/app/third_party", "/app/outputs", "/app/runtime_cache"]

EXPOSE 8000

CMD ["./scripts/dev_server.sh"]
