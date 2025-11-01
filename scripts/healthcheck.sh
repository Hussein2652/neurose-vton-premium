#!/usr/bin/env bash
set -euo pipefail

if command -v jq >/dev/null 2>&1; then
  curl -sf http://127.0.0.1:8000/health | jq . >/dev/null && echo "Health OK"
else
  curl -sf http://127.0.0.1:8000/health >/dev/null && echo "Health OK"
fi
