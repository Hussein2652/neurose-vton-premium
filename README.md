# NEUROSE VTON — Premium Engine (Scaffold)

Author: Mohamed Hussein ElGendy — Founder & AI Architect

This repository is a clean rebuild of the NEUROSE virtual try‑on system targeting premium‑grade quality that surpasses Kling (Kolors) in realism, garment fidelity, and identity preservation.

This initial sprint (S1) delivers:
- Clean repo scaffold
- Deterministic runtime configuration
- Model registry resolving external weights (read‑only mounts)
- Minimal FastAPI with `/v1/tryon-fast`, `/v1/tryon-premium`, and `/health` (stubs)

External model mounts (read‑only):
- `storage/`
- `manual_downloads/`
- `third_party/`

Do not write into these folders. All runtime caches and outputs are local to the repo (`runtime_cache/`, `outputs/`).

## Quick start (dev)

1) Create and activate a Python 3.10+ environment.

2) Install package (editable):

```
pip install -e .[api]
```

3) Run dev server:

```
./scripts/dev_server.sh
```

API: http://127.0.0.1:8000
- `GET /health`
- `GET /v1/registry` (optionally `?alias=<name>`) — includes system info and model resolution report
- `POST /v1/tryon-fast`
- `POST /v1/tryon-premium`
- `GET /ui` — simple in-browser UI to upload images and preview intermediates (served from `/files`)

## Determinism
Set `seed` in requests (or use default from `NEUROSE_DEFAULT_SEED`). The server configures deterministic modes when available (Torch/CUDA), falling back gracefully on CPU‑only setups.

## Model Registry
All model lookups go through `neurose_vton.registry.weights`. Resolution order for each entry:
1. Explicit env var `NEUROSE_MODEL_<ALIAS>`
2. Search in `storage/`, then `manual_downloads/`, then `third_party/` by declared patterns

The registry never writes into these folders. Any per‑model cache must be placed in `runtime_cache/`.

## Structure
- `neurose_vton/config.py` – paths, env, cache
- `neurose_vton/registry/weights.py` – model path resolution
- `neurose_vton/utils/determinism.py` – seed and deterministic toggles
- `neurose_vton/pipeline/` – modular pipeline skeleton
- `neurose_vton/stages/` – modular stage implementations
- `neurose_vton/api/main.py` – FastAPI app

## Healthcheck
`GET /health` returns a seeded mini self‑test status. No heavy models are loaded in this scaffold.

## Notes
- This scaffold is model‑agnostic and network‑free. Integrate actual models in subsequent sprints (S2+).
- Keep external mounts read‑only. Never modify their contents.
## Docker (offline‑first, cache‑aware)

Using Docker Compose (recommended):

```
docker compose build
docker compose up
```

Live reload (Compose Watch, Docker Compose v2.22+):

```
docker compose watch
```

- Syncs local `neurose_vton/` and `scripts/` into the container.
- Uvicorn runs with `--reload`, so code changes restart the server.
- Rebuilds the image when `pyproject.toml` or `Dockerfile` changes.

Dependencies
- Installed at build time and cached in Docker layers. Rebuild only when `pyproject.toml` changes.
- Models are never baked in; they are mounted read-only from `storage/`, `manual_downloads/`, and `third_party/`.
- BuildKit caching: The Dockerfile uses BuildKit cache mounts for apt and pip, so repeated builds reuse cached metadata and wheels. Ensure BuildKit is enabled (Compose v2 enables it by default). You can export `DOCKER_BUILDKIT=1` explicitly if needed.

GPU (NVIDIA) setup
- Install NVIDIA drivers and NVIDIA Container Toolkit on the host.
- By default, the Compose build uses CUDA 11.8 wheels (cu118), which are a solid default for RTX 30‑series.
- To change CUDA version, set a build arg before building:
  - CUDA 12.1: `export PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121`
- Then rebuild: `docker compose build` and run: `docker compose up`.
- The compose file requests GPUs and sets `NVIDIA_VISIBLE_DEVICES=all`.

What it does:
- Mounts `storage/`, `manual_downloads/`, `third_party/` as read‑only.
- Mounts `outputs/` and `runtime_cache/` as read‑write.
- Mounts your host caches into `/host_cache/{pip,torch}` and a local wheelhouse into `/wheels`.
- On startup, the entrypoint seeds caches from host and installs Python deps offline from `/wheels` (no network). Set `ALLOW_NET_INSTALL=1` to allow network installs.

Optional: classic scripts are available (`scripts/docker_build.sh`, `scripts/docker_run.sh`), but Compose is preferred.
- Torch Hub uses `/app/runtime_cache/torch/hub`; the entrypoint seeds from `storage/models/torch/hub` if present.

Tip: Generate wheels once and reuse:

```
mkdir -p wheels
pip wheel --wheel-dir wheels .[api]
```
