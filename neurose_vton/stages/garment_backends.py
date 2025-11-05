from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import logging
import os
import sys


class Sam2MatteBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, garment_path: Path, out_dir: Path) -> Optional[Path]:
        log = logging.getLogger("neurose_vton.garment.sam2")
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
        # Try real SAM2 inference from a local repo if available
        repo = os.environ.get("NEUROSE_REPO_SAM2")
        if repo and Path(repo).exists():
            sys.path.insert(0, repo)
            sys.path.insert(0, str(Path(repo) / "sam2"))
            try:
                import torch  # type: ignore
                import numpy as np  # type: ignore
                from PIL import Image
                # Try common SAM2 predictor/build import paths
                SAM2ImagePredictor = None  # type: ignore
                build_sam2 = None  # type: ignore
                imp_errors = []
                for mod_pred, mod_build in [
                    ("sam2.sam2_image_predictor", "sam2.build_sam2"),
                    ("sam2.sam2.sam2_image_predictor", "sam2.sam2.build_sam2"),
                ]:
                    try:
                        _pred = __import__(mod_pred, fromlist=["SAM2ImagePredictor"])  # type: ignore
                        _build = __import__(mod_build, fromlist=["build_sam2"])  # type: ignore
                        SAM2ImagePredictor = getattr(_pred, "SAM2ImagePredictor")  # type: ignore
                        build_sam2 = getattr(_build, "build_sam2")  # type: ignore
                        break
                    except Exception as _e:
                        imp_errors.append(str(_e))
                        continue
                # Dynamic fallback: locate predictor and builder files
                if SAM2ImagePredictor is None or build_sam2 is None:
                    import importlib.util
                    repo_path = Path(repo)
                    # predictor
                    pred_file = None
                    for cand in repo_path.rglob("sam2_image_predictor.py"):
                        pred_file = cand
                        break
                    if pred_file is not None:
                        spec = importlib.util.spec_from_file_location("_sam2_pred_mod", str(pred_file))
                        if spec and spec.loader:
                            pred_mod = importlib.util.module_from_spec(spec)
                            sys.modules["_sam2_pred_mod"] = pred_mod
                            spec.loader.exec_module(pred_mod)  # type: ignore
                            SAM2ImagePredictor = getattr(pred_mod, "SAM2ImagePredictor", None)
                    # builder
                    build_file = None
                    for name in ("build_sam2.py", "build_sam.py"):
                        for cand in repo_path.rglob(name):
                            build_file = cand
                            break
                        if build_file:
                            break
                    if build_file is not None:
                        specb = importlib.util.spec_from_file_location("_sam2_build_mod", str(build_file))
                        if specb and specb.loader:
                            bmod = importlib.util.module_from_spec(specb)
                            sys.modules["_sam2_build_mod"] = bmod
                            specb.loader.exec_module(bmod)  # type: ignore
                            build_sam2 = getattr(bmod, "build_sam2", None) or getattr(bmod, "build_sam", None)
                if SAM2ImagePredictor is None or build_sam2 is None:
                    raise ImportError("SAM2 imports failed: " + "; ".join(imp_errors))
                # Locate weight
                weight = None
                if self.model_dir and self.model_dir.exists():
                    for pat in ("*.pt", "*.pth"):
                        for cand in self.model_dir.glob(pat):
                            weight = cand
                            break
                        if weight:
                            break
                if weight is None:
                    log.warning("SAM2 weight not found in %s", str(self.model_dir) if self.model_dir else "<unset>")
                else:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    # Locate a config YAML; prefer one that matches 'hiera' and 'large'
                    repo_path = Path(repo)
                    cfg = None
                    preferred = []
                    for cand in repo_path.rglob("*.yaml"):
                        name = cand.name.lower()
                        if "sam2" in str(cand.parent).lower() or "config" in str(cand.parent).lower():
                            preferred.append(cand)
                    # rank by keywords
                    def rank(p: Path) -> int:
                        n = p.name.lower()
                        score = 0
                        if "hiera" in n:
                            score += 2
                        if "large" in n or n.endswith("_l.yaml") or "_l_" in n:
                            score += 2
                        if "sam2" in n:
                            score += 1
                        return score
                    if preferred:
                        preferred.sort(key=rank, reverse=True)
                        cfg = preferred[0]
                    else:
                        # fallback: any yaml in repo
                        any_yaml = list(repo_path.rglob("*.yaml"))
                        if any_yaml:
                            cfg = any_yaml[0]
                    if cfg is None:
                        raise FileNotFoundError("SAM2 config .yaml not found in repo")
                    # Build model with flexible signature + Hydra support
                    built = False
                    import inspect
                    params = list(inspect.signature(build_sam2).parameters.keys())  # type: ignore
                    cfg_name = cfg.stem
                    # 1) Try by parameter names
                    try:
                        kwargs = {}
                        if 'config_name' in params:
                            kwargs['config_name'] = cfg_name
                        elif 'config_file' in params:
                            kwargs['config_file'] = str(cfg)
                        elif 'config' in params:
                            kwargs['config'] = str(cfg)
                        if 'checkpoint' in params:
                            kwargs['checkpoint'] = str(weight)
                        model = build_sam2(**kwargs)  # type: ignore
                        built = True
                    except Exception:
                        pass
                    # 2) Try positional fallbacks
                    if not built:
                        try:
                            model = build_sam2(str(cfg), str(weight))  # type: ignore
                            built = True
                        except Exception:
                            pass
                    # 3) Hydra initialize in cfg directory and try name
                    if not built:
                        try:
                            from hydra import initialize_config_dir  # type: ignore
                            with initialize_config_dir(version_base=None, config_dir=str(cfg.parent), job_name="sam2_build"):
                                try:
                                    model = build_sam2(config_name=cfg.name, checkpoint=str(weight))  # type: ignore
                                    built = True
                                except Exception:
                                    try:
                                        model = build_sam2(config_name=cfg.stem, checkpoint=str(weight))  # type: ignore
                                        built = True
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                    if not built:
                        raise RuntimeError(f"SAM2 build failed for cfg={cfg}")
                    predictor = SAM2ImagePredictor(model)  # type: ignore
                    predictor.to(device)
                    image = Image.open(garment_path).convert("RGB")
                    predictor.set_image(np.array(image))  # type: ignore
                    # Use a central box prompt covering 90% area
                    W, H = image.size
                    bx = int(0.05 * W)
                    by = int(0.05 * H)
                    bw = int(0.90 * W)
                    bh = int(0.90 * H)
                    box = np.array([bx, by, bx + bw, by + bh])  # type: ignore
                    masks, _, _ = predictor.predict(box=box, multimask_output=False)  # type: ignore
                    m = (masks[0].astype("uint8") * 255)  # type: ignore
                    out = out_dir / "garment_matte.png"
                    import cv2  # type: ignore
                    cv2.imwrite(str(out), m)
                    log.info("SAM2 matte saved: %s", str(out))
                    return out
            except Exception as e:
                log.warning("SAM2 repo present but inference failed: %s", e)
        # No repo or failed: obey classical fallback flag
        classical = os.environ.get("NEUROSE_GARMENT_CLASSICAL_FALLBACK", "0").lower() in {"1", "true", "on"}
        if not classical:
            return None
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
            img = cv2.imread(str(garment_path), cv2.IMREAD_COLOR)
            if img is None:
                return None
            h, w = img.shape[:2]
            bx = int(0.05 * w)
            by = int(0.05 * h)
            bw = int(0.90 * w)
            bh = int(0.90 * h)
            rect = (bx, by, bw, bh)
            mask = np.zeros((h, w), np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            kernel = np.ones((5, 5), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
            m = cv2.GaussianBlur(m, (5, 5), 0)
            out = out_dir / "garment_matte.png"
            cv2.imwrite(str(out), m)
            log.info("Classical matte saved: %s", str(out))
            return out
        except Exception:
            return None


class MattingRefineBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, matte_path: Path, garment_path: Path, out_dir: Path) -> Optional[Path]:
        log = logging.getLogger("neurose_vton.garment.matting")
        if not matte_path.exists():
            log.warning("Input matte missing; skipping refine")
            return None
        # Try real RVM refine if repo available
        repo = os.environ.get("NEUROSE_REPO_RVM")
        if repo and Path(repo).exists() and self.model_dir and self.model_dir.exists():
            sys.path.insert(0, repo)
            try:
                import torch  # type: ignore
                import cv2  # type: ignore
                # Try common RVM import paths
                try:
                    from model import MattingNetwork  # type: ignore
                except Exception:
                    from rvm.model import MattingNetwork  # type: ignore
                net = MattingNetwork("mobilenetv3").eval()  # type: ignore
                weight = None
                for pat in ("*.pt", "*.pth"):
                    for cand in self.model_dir.glob(pat):
                        weight = cand
                        break
                    if weight:
                        break
                if weight is None:
                    raise RuntimeError("RVM weight not found")
                net.load_state_dict(torch.load(str(weight), map_location="cpu"))  # type: ignore
                # Proper single-image refine would require a static path; not implemented yet
                raise RuntimeError("RVM single-image refine not implemented; provide classical fallback or video pipeline")
            except Exception as e:
                log.warning("RVM repo present but refine failed: %s", e)
        # Classical edge-aware refine if enabled
        classical = os.environ.get("NEUROSE_GARMENT_CLASSICAL_FALLBACK", "0").lower() in {"1", "true", "on"}
        if not classical:
            return None
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception as e:
            log.warning("OpenCV/NumPy unavailable for refine: %s", e)
            return None
        matte = cv2.imread(str(matte_path), cv2.IMREAD_GRAYSCALE)
        if matte is None:
            return None
        img = cv2.imread(str(garment_path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        # Normalize alpha to [0,1]
        a = (matte.astype(np.float32) / 255.0)
        # Edge-aware smoothing using bilateral filter on alpha guided by image
        # Compute guidance as luminance
        y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        a_blur = cv2.bilateralFilter(a, d=9, sigmaColor=0.1, sigmaSpace=5)
        # Feather edges
        a_ref = np.clip(a_blur, 0.0, 1.0)
        a_ref_u8 = (a_ref * 255.0).astype(np.uint8)
        out = out_dir / "garment_matte_refined.png"
        try:
            cv2.imwrite(str(out), a_ref_u8)
            log.info("Refined matte saved: %s", str(out))
            return out
        except Exception as e:
            log.warning("Failed to write refined matte: %s", e)
            return None


class VitLAttributesBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, garment_path: Path) -> Optional[Dict[str, Any]]:
        log = logging.getLogger("neurose_vton.garment.attrs")
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception as e:
            log.warning("OpenCV/NumPy unavailable for attributes: %s", e)
            return None
        img = cv2.imread(str(garment_path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        H = hsv[:, :, 0] / 180.0  # 0..1
        S = hsv[:, :, 1] / 255.0
        V = hsv[:, :, 2] / 255.0
        # Circular mean for hue
        ang = H * 2.0 * np.pi
        mean_ang = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
        if mean_ang < 0:
            mean_ang += 2.0 * np.pi
        dominant_hue = float(mean_ang / (2.0 * np.pi))
        # Texture metrics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(edges.mean())
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        pattern_score = float(lap.var())
        attrs = {
            "dominant_hue": dominant_hue,
            "saturation_mean": float(S.mean()),
            "brightness_mean": float(V.mean()),
            "edge_density": edge_density,
            "pattern_score": pattern_score,
            "model": str(self.model_dir) if self.model_dir else "",
        }
        return attrs


class UVMapperBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, garment_path: Path, matte_path: Optional[Path], out_dir: Path) -> Optional[Dict[str, Path]]:
        log = logging.getLogger("neurose_vton.garment.uv")
        # Only run classical UV when fallback is enabled
        classical = os.environ.get("NEUROSE_GARMENT_CLASSICAL_FALLBACK", "0").lower() in {"1", "true", "on"}
        if not classical:
            return None
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception as e:
            log.warning("OpenCV/NumPy unavailable for UV: %s", e)
            return None
        img = cv2.imread(str(garment_path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        H, W = img.shape[:2]
        # Load matte if provided to limit region
        mask = np.ones((H, W), np.uint8) * 255
        if matte_path and matte_path.exists():
            m = cv2.imread(str(matte_path), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                _, mask = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            return None
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        w = max(1, x1 - x0 + 1)
        h = max(1, y1 - y0 + 1)
        # Canonical UV in bbox
        grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        u = (grid_x - x0) / float(w)
        v = (grid_y - y0) / float(h)
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
        # Visual atlas RGB = (u, v, 0)
        atlas = np.zeros((H, W, 3), np.uint8)
        atlas[..., 0] = (u * 255.0).astype(np.uint8)
        atlas[..., 1] = (v * 255.0).astype(np.uint8)
        atlas[mask == 0] = 0
        atlas_path = out_dir / "uv_atlas.png"
        cv2.imwrite(str(atlas_path), atlas)
        uv_npz = out_dir / "uv_map.npz"
        np.savez_compressed(str(uv_npz), u=u, v=v, mask=(mask > 0))
        return {"atlas": atlas_path, "map": uv_npz}


class TPSRaftBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, garment_path: Path, out_dir: Path) -> Optional[Dict[str, Path]]:
        log = logging.getLogger("neurose_vton.garment.align")
        if not self.model_dir or not self.model_dir.exists():
            log.warning("TPS/RAFT models not resolved; skipping alignment")
            return None
        # Choose default RAFT checkpoint: raft-things.pth preferred, fall back to raft-sintel.pth
        weight = None
        prefer = [
            self.model_dir / "raft-things.pth",
            self.model_dir / "raft_sintel.pth",
            self.model_dir / "raft-sintel.pth",
        ]
        for p in prefer:
            if p.exists():
                weight = p
                break
        if weight is None:
            # search any .pth under directory as last resort
            try:
                for cand in self.model_dir.rglob("*.pth"):
                    weight = cand
                    break
            except Exception:
                pass
        if weight is None:
            log.warning("RAFT weight not found in %s; skipping alignment", str(self.model_dir))
            return None
        log.info("TPS/RAFT using weight: %s (no outputs in scaffold)", str(weight))
        return None


class ClipPatchBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, garment_path: Path, out_dir: Path) -> Optional[Path]:
        log = logging.getLogger("neurose_vton.garment.prints")
        classical = os.environ.get("NEUROSE_GARMENT_CLASSICAL_FALLBACK", "0").lower() in {"1", "true", "on"}
        if not classical:
            return None
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception as e:
            log.warning("OpenCV/NumPy unavailable for print mask: %s", e)
            return None
        img = cv2.imread(str(garment_path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # High-frequency content via Laplacian magnitude
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        mag = np.abs(lap)
        # Adaptive threshold
        m = mag / (mag.max() + 1e-8)
        thr = float(np.clip(m.mean() + m.std(), 0.2, 0.8))
        mask = (m > thr).astype(np.uint8) * 255
        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        out = out_dir / "print_mask.png"
        cv2.imwrite(str(out), mask)
        log.info("Print mask saved: %s", str(out))
        return out
