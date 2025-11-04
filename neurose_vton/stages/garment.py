from __future__ import annotations

from pathlib import Path
from .base import StageOutput
from ..registry import registry
from .garment_backends import (
    Sam2MatteBackend,
    MattingRefineBackend,
    VitLAttributesBackend,
    UVMapperBackend,
    TPSRaftBackend,
    ClipPatchBackend,
)
import json
import logging


class GarmentAnalysis:
    def __init__(self) -> None:
        pass

    def run(self, garment_path: Path, seed: int, trace_dir: Path | None = None) -> StageOutput:
        log = logging.getLogger("neurose_vton.garment")
        # Reserve a stage-specific trace folder without writing placeholders
        out_dir = None
        if trace_dir is not None:
            out_dir = trace_dir / "garment"
            out_dir.mkdir(parents=True, exist_ok=True)
        resolved = {
            "sam2": str(registry.resolve("sam2") or ""),
            "matting_refine": str(registry.resolve("matting_refine") or ""),
            "vitl_apparel": str(registry.resolve("vitl_apparel") or ""),
            "uv_mapper": str(registry.resolve("uv_mapper") or ""),
            "tps_raft": str(registry.resolve("tps_raft") or ""),
            "clip_patch": str(registry.resolve("clip_patch") or ""),
        }
        status: dict[str, dict] = {}
        artifacts: dict[str, Path] = {}

        # 1) Product matte via SAM2
        matte_png = None
        try:
            sam2 = Sam2MatteBackend(model_dir=Path(resolved["sam2"]) if resolved["sam2"] else None)
            matte_png = sam2.compute(garment_path, out_dir) if out_dir else None
            status["matte"] = {"ok": bool(matte_png and Path(matte_png).exists()), "backend": "sam2"}
            if matte_png and Path(matte_png).exists():
                artifacts["garment_matte"] = Path(matte_png)
        except Exception as e:
            log.warning("SAM2 matte failed: %s", e)
            status["matte"] = {"ok": False, "backend": "sam2", "error": str(e)}

        # 2) Refine matte
        try:
            refine = MattingRefineBackend(model_dir=Path(resolved["matting_refine"]) if resolved["matting_refine"] else None)
            refined_png = refine.compute(Path(matte_png) if matte_png else Path(""), garment_path, out_dir) if out_dir else None
            status["matte_refine"] = {"ok": bool(refined_png and Path(refined_png).exists()), "backend": "matting_refine"}
            if refined_png and Path(refined_png).exists():
                artifacts["garment_matte_refined"] = Path(refined_png)
        except Exception as e:
            log.warning("Matting refine failed: %s", e)
            status["matte_refine"] = {"ok": False, "backend": "matting_refine", "error": str(e)}

        # 3) Attributes
        try:
            vit = VitLAttributesBackend(model_dir=Path(resolved["vitl_apparel"]) if resolved["vitl_apparel"] else None)
            attrs = vit.compute(garment_path)
            status["attributes"] = {"ok": bool(attrs), "backend": "vitl_apparel"}
            if attrs and out_dir:
                (out_dir / "attributes.json").write_text(json.dumps(attrs, indent=2))
                artifacts["attributes"] = out_dir / "attributes.json"
        except Exception as e:
            log.warning("Attributes failed: %s", e)
            status["attributes"] = {"ok": False, "backend": "vitl_apparel", "error": str(e)}

        # 4) UV mapper
        try:
            uv = UVMapperBackend(model_dir=Path(resolved["uv_mapper"]) if resolved["uv_mapper"] else None)
            uv_out = uv.compute(garment_path, Path(matte_png) if matte_png else None, out_dir) if out_dir else None
            status["uv"] = {"ok": bool(uv_out), "backend": "uv_mapper"}
            if uv_out:
                for k, p in uv_out.items():
                    if Path(p).exists():
                        artifacts[f"uv_{k}"] = Path(p)
        except Exception as e:
            log.warning("UV failed: %s", e)
            status["uv"] = {"ok": False, "backend": "uv_mapper", "error": str(e)}

        # 5) Geometry alignment
        try:
            align = TPSRaftBackend(model_dir=Path(resolved["tps_raft"]) if resolved["tps_raft"] else None)
            align_out = align.compute(garment_path, out_dir) if out_dir else None
            status["alignment"] = {"ok": bool(align_out), "backend": "tps_raft"}
            if align_out:
                for k, p in align_out.items():
                    if Path(p).exists():
                        artifacts[f"align_{k}"] = Path(p)
        except Exception as e:
            log.warning("Alignment failed: %s", e)
            status["alignment"] = {"ok": False, "backend": "tps_raft", "error": str(e)}

        # 6) Logo/print mask
        try:
            clip = ClipPatchBackend(model_dir=Path(resolved["clip_patch"]) if resolved["clip_patch"] else None)
            print_mask = clip.compute(garment_path, out_dir) if out_dir else None
            status["print_mask"] = {"ok": bool(print_mask and Path(print_mask).exists()), "backend": "clip_patch"}
            if print_mask and Path(print_mask).exists():
                artifacts["print_mask"] = Path(print_mask)
        except Exception as e:
            log.warning("Print mask failed: %s", e)
            status["print_mask"] = {"ok": False, "backend": "clip_patch", "error": str(e)}

        # Save model resolution + status
        if out_dir:
            (out_dir / "models.json").write_text(json.dumps(resolved, indent=2))
            (out_dir / "status.json").write_text(json.dumps(status, indent=2))

        return StageOutput(
            data={
                "garment_matte": str(artifacts.get("garment_matte", "")),
                "uv": {k: str(v) for k, v in artifacts.items() if k.startswith("uv_")},
                "alignment": {k: str(v) for k, v in artifacts.items() if k.startswith("align_")},
                "print_mask": str(artifacts.get("print_mask", "")),
                "models": resolved,
            },
            artifacts=artifacts,
        )
