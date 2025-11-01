from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import PATHS, SETTINGS
from ..utils.determinism import set_seed, enable_determinism
from ..utils.logging_utils import configure_logging
from ..stages import (
    CompositionPlanning,
    DiffusionCore,
    GarmentAnalysis,
    HarmonizeUpscale,
    PersonAnalysis,
    PhysicalRefinement,
)


@dataclass
class TryOnConfig:
    steps: int = 24
    fit: str = "true"  # "tight" | "true" | "loose"
    save_intermediates: bool = False
    seed: Optional[int] = None


@dataclass
class TryOnResult:
    output_path: Optional[Path]
    trace_dir: Optional[Path]


class TryOnPipeline:
    def __init__(self, cfg: TryOnConfig):
        self.cfg = cfg
        configure_logging(SETTINGS.log_level)
        enable_determinism(strict=SETTINGS.strict_determinism)

    def _trace_dir(self) -> Path:
        trace_id = uuid.uuid4().hex[:8]
        d = PATHS.outputs / f"trace_{trace_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def run(self, person_image: Path, garment_image: Path) -> TryOnResult:
        seed = self.cfg.seed if self.cfg.seed is not None else SETTINGS.seed
        set_seed(seed)

        trace_dir = self._trace_dir() if self.cfg.save_intermediates else None

        # Stage 1: Person analysis
        person = PersonAnalysis().run(person_image, seed, trace_dir=trace_dir)
        # Stage 2: Garment analysis
        garment = GarmentAnalysis().run(garment_image, seed)
        # Stage 3: Composition planning
        plan = CompositionPlanning().run(person, garment, self.cfg.fit)
        # Stage 4: Diffusion core
        core = DiffusionCore(self.cfg.steps).run(plan, seed)
        # Stage 5: Physical refinement
        refined = PhysicalRefinement().run(core)
        # Stage 6: Harmonization & upscale
        final = HarmonizeUpscale().run(refined)

        out_path = None
        if trace_dir:
            # Save a minimal manifest until real images exist
            manifest = {
                "seed": seed,
                "steps": self.cfg.steps,
                "fit": self.cfg.fit,
                "stages": {
                    "person": list(person.data.keys()),
                    "garment": list(garment.data.keys()),
                    "plan": list(plan.data.keys()),
                    "core": list(core.data.keys()),
                },
            }
            (trace_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return TryOnResult(output_path=out_path, trace_dir=trace_dir)
