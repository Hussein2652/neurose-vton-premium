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
