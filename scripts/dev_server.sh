#!/usr/bin/env bash
set -euo pipefail

export NEUROSE_DEFAULT_SEED="12345"
export NEUROSE_STRICT_DETERMINISM="1"
exec uvicorn neurose_vton.api.main:app --host 0.0.0.0 --port 8000 --reload
