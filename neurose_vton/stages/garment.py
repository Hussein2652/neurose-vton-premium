from __future__ import annotations

from pathlib import Path
from .base import StageOutput
from ..registry import registry


class GarmentAnalysis:
    def __init__(self) -> None:
        pass

    def run(self, garment_path: Path, seed: int, trace_dir: Path | None = None) -> StageOutput:
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
        return StageOutput(
            data={
                "garment_matte": None,
                "uv_atlas": None,
                "tps_grid": None,
                "flow_field": None,
                "print_masks": None,
                "models": resolved,
            },
            artifacts={},
        )
