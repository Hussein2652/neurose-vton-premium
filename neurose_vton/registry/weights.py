from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ..config import PATHS


SEARCH_ROOTS = [PATHS.storage, PATHS.manual, PATHS.third_party]


@dataclass
class ModelSpec:
    alias: str
    description: str
    patterns: list[str] = field(default_factory=list)
    env_var: Optional[str] = None
    required: bool = False

    def resolved_env(self, env: dict[str, str]) -> Optional[Path]:
        key = self.env_var or f"NEUROSE_MODEL_{self.alias.upper()}"
        val = env.get(key)
        if not val:
            return None
        p = Path(val).expanduser().resolve()
        return p if p.exists() else None


class ModelRegistry:
    def __init__(self, specs: Iterable[ModelSpec]):
        self._specs = {s.alias: s for s in specs}

    def list(self) -> list[str]:
        return sorted(self._specs.keys())

    def spec(self, alias: str) -> ModelSpec:
        return self._specs[alias]

    def resolve(self, alias: str, env: Optional[dict[str, str]] = None) -> Optional[Path]:
        env = env or os.environ
        spec = self.spec(alias)

        # 1) Env override
        p = spec.resolved_env(env)
        if p is not None:
            return p

        # 2) Pattern search in external roots
        for root in SEARCH_ROOTS:
            if not root.exists():
                continue
            found = self._match_first(root, spec.patterns)
            if found is not None:
                return found
        return None

    def report(self, env: Optional[dict[str, str]] = None) -> dict[str, dict[str, str]]:
        env = env or os.environ
        out: dict[str, dict[str, str]] = {}
        for a, s in self._specs.items():
            resolved = self.resolve(a, env=env)
            out[a] = {
                "description": s.description,
                "required": str(s.required),
                "exists": str(bool(resolved)),
                "path": str(resolved) if resolved else "",
            }
        return out

    @staticmethod
    def _match_first(root: Path, patterns: list[str]) -> Optional[Path]:
        if not patterns:
            return None
        # glob-like walk with fnmatch over all files and dirs
        for pattern in patterns:
            for p in root.rglob("*"):
                rel = str(p.relative_to(root))
                if fnmatch.fnmatch(rel, pattern):
                    return p
        return None


# Canonical model aliases. Patterns are intentionally broad; env overrides are preferred.
SPECS: list[ModelSpec] = [
    # 1) Person analysis
    ModelSpec("insightface", "Face detection/embeddings (InsightFace / InstantID)", [
        "**/unpacked/antelopev2",
        "**/antelopev2*",
        "**/antelopev2*/*",
        "**/insightface*/*",
        "**/instantid*/*",
    ]),
    ModelSpec("openpose", "OpenPose BODY_25 or YOLOv8-pose weights", [
        "**/openpose*/*",
        "**/body_25*/*",
        "**/yolov8*pose*",
        "**/yolov8*pose*/*",
    ]),
    ModelSpec("schp", "SCHP human parsing", ["**/schp*/*", "**/human*parsing*/*"]),
    ModelSpec("depth", "Zoe-Depth / DPT-Hybrid", [
        "**/zoe*depth*/*",
        "**/ZoeD*",
        "**/dpt*hybrid*/*",
        "**/*midas*/*",
    ]),
    ModelSpec("smplx", "SMPL-X body model", ["**/smplx*/*", "**/smpl*/*"]),
    ModelSpec("relight_sh", "Spherical harmonics relight head", ["**/sh*relight*/*"]),

    # 2) Garment analysis
    ModelSpec("sam2", "Segment Anything v2 (product segmentation)", ["**/sam2*/*", "**/segment*anything*/*"]),
    ModelSpec("matting_refine", "Matting refine model", ["**/matting*refine*/*", "**/bgm*/*"]),
    ModelSpec("vitl_apparel", "ViT-L apparel attribute head", ["**/vit*l*/*", "**/apparel*/*"]),
    ModelSpec("uv_mapper", "Learned UV mapping network", ["**/uv*map*/*"]),
    ModelSpec("tps_raft", "TPS + RAFT-flow alignment", ["**/tps*/*", "**/raft*/*"]),
    ModelSpec("clip_patch", "CLIP-ViT patch/logo head", ["**/clip*vit*/*", "**/logo*/*"]),

    # 4) Diffusion core
    ModelSpec("idm_vton_unet", "IDM-VTON UNet core", ["**/idm*vton*/*"]),
    ModelSpec("stableviton", "StableVITON blocks", ["**/stable*viton*/*"]),
    ModelSpec("controlnet_pose", "ControlNet Pose", ["**/controlnet*pose*/*"]),
    ModelSpec("controlnet_seg", "ControlNet Seg", ["**/controlnet*seg*/*"]),
    ModelSpec("controlnet_depth", "ControlNet Depth", ["**/controlnet*depth*/*"]),
    ModelSpec("lama_fabric", "LaMa-fabric seam cleanup", ["**/lama*fabric*/*", "**/lama*/*"]),

    # 6) Harmonization & Upscale
    ModelSpec("realesrgan_x2plus", "RealESRGAN x2plus", ["**/realesrgan*x2*/*", "**/realesrgan*/*"]),
]


registry = ModelRegistry(SPECS)
