# syntax=docker/dockerfile:1.6
# Clean, single-stage Dockerfile. No external repo refs. Strict caching.

FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_CACHE_DIR=/root/.cache/pip \
    TORCH_HOME=/app/runtime_cache/torch \
    HF_HOME=/app/runtime_cache/hf \
    NO_ALBUMENTATIONS_UPDATE=1 \
    PYTHONWARNINGS="ignore::UserWarning:albumentations.__init__,ignore::UserWarning:onnxruntime.capi.onnxruntime_inference_collection" \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PATH=/usr/local/cuda/bin:${PATH}

WORKDIR /app

# System deps first: required to build some wheels (e.g., insightface)
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

# Install Python deps; cached by lockfile content only
COPY requirements.lock /app/requirements.lock
ARG PIP_VERSION=24.2
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "pip==${PIP_VERSION}"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /app/requirements.lock

# Runtime dirs
RUN mkdir -p /app/runtime_cache /app/outputs

# App code and scripts
COPY neurose_vton /app/neurose_vton
COPY scripts /app/scripts
COPY pyproject.toml README.md /app/

# Install the project itself without touching deps (so Torch/deps remain cached)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e . --no-deps

VOLUME ["/app/storage", "/app/manual_downloads", "/app/third_party", "/app/outputs", "/app/runtime_cache"]

EXPOSE 8000

CMD ["./scripts/dev_server.sh"]
