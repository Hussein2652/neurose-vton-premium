# AGENTS: neurose-vton-premium

This repository is organized for a fresh, modular rebuild of the NEUROSE VTON engine. Please follow these constraints when modifying code within this repo:

Principles
- Premium Quality First: Treat image quality, determinism, and traceability as first‑class features.
- Determinism: Every inference must be seed‑controlled and repeatable when possible.
- Transparency: Expose intermediate artifacts (pose, seg, depth, UV, warped cloth, etc.) via a structured trace when requested.
- Extensibility: Each pipeline stage is modular and swappable through clear interfaces.
- Zero Data Loss: External model folders are read‑only bind mounts (`storage/`, `manual_downloads/`, `third_party/`). Never modify or write into them.

Repository Layout (target)
- `neurose_vton/` – Python package containing configuration, registry, pipeline, stages, and API.
- `storage/`, `manual_downloads/`, `third_party/` – External, read‑only model mounts. Do not modify.
- `scripts/` – Dev helpers (local server, healthcheck).
- `tests/` – Minimal tests (fast, no network, no model downloads).

Coding Conventions
- Keep code Python 3.10+ compatible.
- Prefer simple, explicit abstractions over deep class hierarchies.
- Avoid heavy imports at module import time; defer to function scope where possible.
- All file system writes must target the runtime cache or outputs directories configured in `config.py` (never into external mounts).

Interfaces
- Model resolution must go through `neurose_vton.registry.weights`.
- Determinism utilities live in `neurose_vton.utils.determinism`.
- The orchestration entrypoint is `neurose_vton.pipeline.pipeline:TryOnPipeline`.
- API app is `neurose_vton.api.main:app` and exposes `/v1/tryon-fast`, `/v1/tryon-premium`, and `/health`.

Testing
- Tests must not require GPU, network, or model downloads.
- Prefer validating configs, path resolution, and deterministic seeding behavior.
