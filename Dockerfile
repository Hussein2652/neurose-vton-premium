# Minimal dev Dockerfile (real image to be hardened in later sprints)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY neurose_vton /app/neurose_vton
COPY scripts /app/scripts

RUN pip install --upgrade pip && \
    pip install -e .[api]

# External read-only mounts (bind at runtime)
VOLUME ["/app/storage", "/app/manual_downloads", "/app/third_party", "/app/outputs", "/app/runtime_cache"]

EXPOSE 8000
CMD ["./scripts/dev_server.sh"]
