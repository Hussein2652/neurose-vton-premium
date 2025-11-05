from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from starlette.staticfiles import StaticFiles

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
    # Ensure expected runtime folders exist for third-party tools
    try:
        PATHS.runtime_cache.mkdir(parents=True, exist_ok=True)
        PATHS.outputs.mkdir(parents=True, exist_ok=True)
        for key in ("YOLO_CONFIG_DIR", "XDG_CONFIG_HOME", "XDG_CACHE_HOME"):
            p = os.environ.get(key)
            if p:
                Path(p).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# Static mount for outputs to preview intermediates
app.mount("/files", StaticFiles(directory=str(PATHS.outputs)), name="files")


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


def _trace_url(trace_dir: Path | None) -> str | None:
    if not trace_dir:
        return None
    try:
        rel = trace_dir.relative_to(PATHS.outputs)
        return f"/files/{rel.as_posix()}"
    except Exception:
        return None


@app.get("/")
def index() -> RedirectResponse:
    return RedirectResponse(url="/ui")


@app.get("/web")
def web_alias() -> RedirectResponse:
    return RedirectResponse(url="/ui")


@app.get("/ui")
def ui() -> HTMLResponse:
    html = """
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>NEUROSE VTON — Try-On UI</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
      .row { display: flex; gap: 12px; align-items: flex-start; margin-bottom: 12px; flex-wrap: wrap; }
      .col { display: flex; flex-direction: column; gap: 6px; }
      .card { border: 1px solid #ddd; padding: 12px; border-radius: 8px; }
      img { max-width: 280px; border: 1px solid #ccc; border-radius: 4px; }
      code { background: #f6f8fa; padding: 2px 6px; border-radius: 4px; }
      .muted { color: #666; font-size: 0.9em; }
      button { padding: 8px 12px; }
    </style>
  </head>
  <body>
    <h1>NEUROSE VTON — Simple Try-On</h1>
    <div class=\"card\">
      <form id=\"form\">
        <div class=\"row\">
          <div class=\"col\">
            <label>Person image</label>
            <input type=\"file\" name=\"person_image\" accept=\"image/*\" required />
          </div>
          <div class=\"col\">
            <label>Garment image</label>
            <input type=\"file\" name=\"garment_image\" accept=\"image/*\" required />
          </div>
        </div>
        <div class=\"row\">
          <div class=\"col\">
            <label>Mode</label>
            <select name=\"mode\">
              <option value=\"fast\">fast (24 steps)</option>
              <option value=\"premium\" selected>premium (40 steps)</option>
            </select>
          </div>
          <div class=\"col\">
            <label>Seed (optional)</label>
            <input type=\"number\" name=\"seed\" placeholder=\"e.g. 12345\" />
          </div>
          <div class=\"col\">
            <label><input type=\"checkbox\" name=\"return_intermediates\" checked /> save intermediates</label>
          </div>
        </div>
        <div class=\"row\">
          <div class=\"col\">
            <label>Depth provider</label>
            <select name=\"depth_provider\">
              <option value=\"\">auto (env)</option>
              <option value=\"midas\">MiDaS</option>
            </select>
          </div>
          <div class=\"col\">
            <label>MiDaS variant</label>
            <input type=\"text\" name=\"midas_variant\" placeholder=\"DPT_BEiT_L_384\" />
          </div>
          <div class=\"col\">
            <label>SCHP TTA scales</label>
            <input type=\"text\" name=\"schp_tta_scales\" placeholder=\"1.0,0.75,1.25\" />
          </div>
          <div class=\"col\">
            <label><input type=\"checkbox\" name=\"schp_fp16\" /> SCHP fp16</label>
          </div>
        </div>
        <button type=\"submit\">Run Try-On</button>
      </form>
    </div>

    <div id=\"result\" class=\"card\" style=\"margin-top:16px; display:none;\"></div>

    <script>
      const form = document.getElementById('form');
      const out = document.getElementById('result');
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        out.style.display = 'block';
        out.innerHTML = '<div class=\"muted\">Running…</div>';
        const fd = new FormData(form);
        const mode = fd.get('mode');
        const url = mode === 'fast' ? '/v1/tryon-fast' : '/v1/tryon-premium';
        try {
          const res = await fetch(url, { method: 'POST', body: fd });
          const j = await res.json();
          const traceUrl = j.trace_url || null;
          let html = '';
          html += '<div><b>Response</b>:</div>';
          html += '<pre>'+JSON.stringify(j, null, 2)+'</pre>';
          if (traceUrl) {
            const p = traceUrl + '/person';
            html += '<div class=\"row\">';
            const imgs = [
              'seg_overlay.png','segmentation.png','hair_mask.png','arms_mask.png','hands_mask.png','depth.png','normals.png'
            ];
            for (const name of imgs) {
              html += '<div class=\"col\"><div class=\"muted\">'+name+'</div><img src=\"'+p+'/'+name+'\" onerror=\"this.style.display=\\'none\\'\" /></div>';
            }
            html += '</div>';
            html += '<div class=\"muted\">Trace folder: <code>'+traceUrl+'</code></div>';
            // Garment block
            const g = traceUrl + '/garment';
            html += '<hr /><div><b>Garment</b>:</div>';
            html += '<div class=\"row\">';
            const gimgs = [
              'garment_matte.png','garment_matte_refined.png','uv_atlas.png','print_mask.png'
            ];
            for (const name of gimgs) {
              html += '<div class=\"col\"><div class=\"muted\">'+name+'</div><img src=\"'+g+'/'+name+'\" onerror=\"this.style.display=\\'none\\'\" /></div>';
            }
            html += '</div>';
            html += '<div class=\"muted\">Garment models/status: <a href=\"'+g+'/models.json\" target=\"_blank\">models.json</a> · <a href=\"'+g+'/status.json\" target=\"_blank\">status.json</a> · <a href=\"'+g+'/attributes.json\" target=\"_blank\">attributes.json</a></div>';
          }
          out.innerHTML = html;
        } catch (err) {
          out.innerHTML = '<div style=\"color:#b00\">Error: '+(err?.message||err)+'</div>';
        }
      });
    </script>
  </body>
  </html>
    """
    return HTMLResponse(html)


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
    depth_provider: Optional[str] = Form(None),  # 'midas'
    midas_variant: Optional[str] = Form(None),   # e.g., 'DPT_BEiT_L_384'
    schp_tta_scales: Optional[str] = Form(None), # e.g., '1.0,0.75,1.25'
    schp_fp16: Optional[str] = Form(None),       # 'on' | '1' | '0'
) -> JSONResponse:
    tmp = PATHS.runtime_cache / "uploads"
    p_path = _persist_upload(tmp, person_image)
    g_path = _persist_upload(tmp, garment_image)

    # Per-request env overrides
    overrides: dict[str, str] = {}
    if depth_provider:
        # Only MiDaS is supported now
        overrides['NEUROSE_SKIP_ZOEDEPTH'] = '1'
    if midas_variant:
        overrides['NEUROSE_MIDAS_VARIANT'] = midas_variant
    if schp_tta_scales:
        overrides['NEUROSE_SCHP_TTA_SCALES'] = schp_tta_scales
    if schp_fp16 is not None:
        val = str(schp_fp16).lower()
        overrides['NEUROSE_SCHP_FP16'] = '1' if val in {'1', 'true', 'on', 'yes'} else '0'

    prev: dict[str, Optional[str]] = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            os.environ[k] = v
        cfg = TryOnConfig(steps=24, seed=seed, save_intermediates=return_intermediates)
        result = TryOnPipeline(cfg).run(p_path, g_path)
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
    return JSONResponse({
        "mode": "fast",
        "seed": seed or SETTINGS.seed,
        "trace_dir": str(result.trace_dir) if result.trace_dir else None,
        "trace_url": _trace_url(result.trace_dir),
        "output_path": str(result.output_path) if result.output_path else None,
    })


@app.post("/v1/tryon-premium")
async def tryon_premium(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    seed: Optional[int] = Form(None),
    return_intermediates: bool = Form(False),
    depth_provider: Optional[str] = Form(None),
    midas_variant: Optional[str] = Form(None),
    schp_tta_scales: Optional[str] = Form(None),
    schp_fp16: Optional[str] = Form(None),
) -> JSONResponse:
    tmp = PATHS.runtime_cache / "uploads"
    p_path = _persist_upload(tmp, person_image)
    g_path = _persist_upload(tmp, garment_image)

    overrides: dict[str, str] = {}
    if depth_provider:
        overrides['NEUROSE_SKIP_ZOEDEPTH'] = '1'
    if midas_variant:
        overrides['NEUROSE_MIDAS_VARIANT'] = midas_variant
    if schp_tta_scales:
        overrides['NEUROSE_SCHP_TTA_SCALES'] = schp_tta_scales
    if schp_fp16 is not None:
        val = str(schp_fp16).lower()
        overrides['NEUROSE_SCHP_FP16'] = '1' if val in {'1', 'true', 'on', 'yes'} else '0'

    prev: dict[str, Optional[str]] = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            os.environ[k] = v
        cfg = TryOnConfig(steps=40, seed=seed, save_intermediates=return_intermediates)
        result = TryOnPipeline(cfg).run(p_path, g_path)
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
    return JSONResponse({
        "mode": "premium",
        "seed": seed or SETTINGS.seed,
        "trace_dir": str(result.trace_dir) if result.trace_dir else None,
        "trace_url": _trace_url(result.trace_dir),
        "output_path": str(result.output_path) if result.output_path else None,
    })
