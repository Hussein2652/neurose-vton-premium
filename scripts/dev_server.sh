#!/usr/bin/env bash
set -euo pipefail

exec uvicorn neurose_vton.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --reload-exclude "runtime_cache/*" \
  --reload-exclude "runtime_cache/**" \
  --reload-exclude "outputs/*" \
  --reload-exclude "outputs/**"
