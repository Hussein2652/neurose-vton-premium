from __future__ import annotations

from pathlib import Path
from .base import StageOutput
from ..registry import registry


class GarmentAnalysis:
    def __init__(self) -> None:
        pass

    def run(self, garment_path: Path, seed: int) -> StageOutput:
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
