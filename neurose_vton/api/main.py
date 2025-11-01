from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

from ..config import PATHS, SETTINGS
from ..utils.determinism import set_seed, enable_determinism
from ..utils.logging_utils import configure_logging
from ..pipeline import TryOnConfig, TryOnPipeline
from ..registry import registry
from ..system import system_info


app = FastAPI(title="NEUROSE VTON Premium (Scaffold)")


@app.on_event("startup")
def on_startup() -> None:
    configure_logging(SETTINGS.log_level)
    enable_determinism(strict=SETTINGS.strict_determinism)


@app.get("/health")
def health() -> JSONResponse:
    # Seeded mini self-test (no heavy models in scaffold)
    set_seed(SETTINGS.seed)
    return JSONResponse({
        "status": "ok",
        "seed": SETTINGS.seed,
        "cache_dir": str(PATHS.runtime_cache),
        "outputs_dir": str(PATHS.outputs),
    })


def _persist_upload(tmp_dir: Path, f: UploadFile) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    target = tmp_dir / f.filename
    with target.open("wb") as out:
        out.write(f.file.read())
    return target


@app.get("/v1/registry")
def registry_status(alias: Optional[str] = None) -> JSONResponse:
    if alias:
        try:
            spec = registry.spec(alias)
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "unknown alias",
                    "alias": alias,
                    "available": registry.list(),
                },
            )
        resolved = registry.resolve(alias)
        env_var = spec.env_var or f"NEUROSE_MODEL_{alias.upper()}"
        return JSONResponse(
            {
                "alias": alias,
                "description": spec.description,
                "required": str(spec.required),
                "env_var": env_var,
                "patterns": spec.patterns,
                "exists": bool(resolved),
                "path": str(resolved) if resolved else "",
            }
        )

    report = registry.report()
    return JSONResponse(
        {
            "roots": {
                "storage": str(PATHS.storage),
                "manual": str(PATHS.manual),
                "third_party": str(PATHS.third_party),
            },
            "available": registry.list(),
            "report": report,
            "system": system_info(),
        }
    )


@app.post("/v1/tryon-fast")
async def tryon_fast(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    seed: Optional[int] = Form(None),
    return_intermediates: bool = Form(False),
) -> JSONResponse:
    tmp = PATHS.runtime_cache / "uploads"
    p_path = _persist_upload(tmp, person_image)
    g_path = _persist_upload(tmp, garment_image)

    cfg = TryOnConfig(steps=24, seed=seed, save_intermediates=return_intermediates)
    result = TryOnPipeline(cfg).run(p_path, g_path)
    return JSONResponse({
        "mode": "fast",
        "seed": seed or SETTINGS.seed,
        "trace_dir": str(result.trace_dir) if result.trace_dir else None,
        "output_path": str(result.output_path) if result.output_path else None,
    })


@app.post("/v1/tryon-premium")
async def tryon_premium(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    seed: Optional[int] = Form(None),
    return_intermediates: bool = Form(False),
) -> JSONResponse:
    tmp = PATHS.runtime_cache / "uploads"
    p_path = _persist_upload(tmp, person_image)
    g_path = _persist_upload(tmp, garment_image)

    cfg = TryOnConfig(steps=40, seed=seed, save_intermediates=return_intermediates)
    result = TryOnPipeline(cfg).run(p_path, g_path)
    return JSONResponse({
        "mode": "premium",
        "seed": seed or SETTINGS.seed,
        "trace_dir": str(result.trace_dir) if result.trace_dir else None,
        "output_path": str(result.output_path) if result.output_path else None,
    })
