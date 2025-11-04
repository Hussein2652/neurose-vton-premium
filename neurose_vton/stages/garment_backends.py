from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import logging


class Sam2MatteBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, garment_path: Path, out_dir: Path) -> Optional[Path]:
        log = logging.getLogger("neurose_vton.garment.sam2")
        if not self.model_dir or not self.model_dir.exists():
            log.warning("SAM2 model not resolved; skipping matte")
            return None
        # Scaffold only: actual model inference intentionally omitted (no placeholders)
        log.info("SAM2 resolved at %s; matte not produced in scaffold", str(self.model_dir))
        return None


class MattingRefineBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, matte_path: Path, garment_path: Path, out_dir: Path) -> Optional[Path]:
        log = logging.getLogger("neurose_vton.garment.matting")
        if not self.model_dir or not self.model_dir.exists():
            log.warning("MattingRefine model not resolved; skipping refine")
            return None
        if not matte_path.exists():
            log.warning("Input matte missing; skipping refine")
            return None
        log.info("MattingRefine resolved at %s; refine not produced in scaffold", str(self.model_dir))
        return None


class VitLAttributesBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, garment_path: Path) -> Optional[Dict[str, Any]]:
        log = logging.getLogger("neurose_vton.garment.attrs")
        if not self.model_dir or not self.model_dir.exists():
            log.warning("ViT-L apparel model not resolved; skipping attributes")
            return None
        log.info("ViT-L resolved at %s; attributes not produced in scaffold", str(self.model_dir))
        return None


class UVMapperBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, garment_path: Path, matte_path: Optional[Path], out_dir: Path) -> Optional[Dict[str, Path]]:
        log = logging.getLogger("neurose_vton.garment.uv")
        if not self.model_dir or not self.model_dir.exists():
            log.warning("UV mapper not resolved; skipping UV")
            return None
        log.info("UV mapper resolved at %s; UV not produced in scaffold", str(self.model_dir))
        return None


class TPSRaftBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, garment_path: Path, out_dir: Path) -> Optional[Dict[str, Path]]:
        log = logging.getLogger("neurose_vton.garment.align")
        if not self.model_dir or not self.model_dir.exists():
            log.warning("TPS/RAFT models not resolved; skipping alignment")
            return None
        log.info("TPS/RAFT resolved at %s; alignment not produced in scaffold", str(self.model_dir))
        return None


class ClipPatchBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, garment_path: Path, out_dir: Path) -> Optional[Path]:
        log = logging.getLogger("neurose_vton.garment.prints")
        if not self.model_dir or not self.model_dir.exists():
            log.warning("CLIP-ViT patch head not resolved; skipping print mask")
            return None
        log.info("CLIP-ViT resolved at %s; print mask not produced in scaffold", str(self.model_dir))
        return None

