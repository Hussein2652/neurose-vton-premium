from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import subprocess
import shutil
import sys
import os
import logging


def _load_image(path: Path):
    try:
        from PIL import Image
        return Image.open(path).convert("RGB")
    except Exception:
        return None


@dataclass
class FaceResult:
    bbox: Optional[List[float]]
    embedding: Optional[List[float]]
    landmarks: Optional[List[List[float]]]


class FaceIDBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir
        self._insight_model = None
        try:
            import insightface  # type: ignore

            # InsightFace model loading typically uses model_zoo or app.FaceAnalysis
            # We defer heavy init until compute to avoid startup cost.
            self._lib = insightface
        except Exception:
            self._lib = None

    def compute(self, image_path: Path) -> Optional[FaceResult]:
        log = logging.getLogger("neurose_vton.person.face")
        if self._lib is None:
            log.warning("InsightFace not installed; skipping face analysis")
            return None
        try:
            import numpy as np  # type: ignore
            from insightface.app import FaceAnalysis  # type: ignore
            # Choose model pack name by available files
            pack = "buffalo_l"
            root_path = self.model_dir
            if self.model_dir:
                if (self.model_dir / "antelopev2").exists():
                    pack = "antelopev2"
                elif (self.model_dir / "unpacked" / "antelopev2").exists():
                    pack = "antelopev2"
                    root_path = self.model_dir / "unpacked"
            app = FaceAnalysis(name=pack, root=str(root_path) if root_path else None)
            app.prepare(ctx_id=0 if self._has_cuda() else -1)
            img = _load_image(image_path)
            if img is None:
                log.error("Failed to load image for face analysis: %s", image_path)
                return None
            arr = np.array(img)[:, :, ::-1]  # BGR for insightface
            faces = app.get(arr)
            if not faces:
                log.info("No face detected")
                return FaceResult(bbox=None, embedding=None, landmarks=None)
            f = faces[0]
            bbox = [float(x) for x in f.bbox]
            emb = [float(x) for x in f.normed_embedding.tolist()] if hasattr(f, "normed_embedding") else None
            # Landmarks: prefer dense if available, else 5-point kps
            lms: Optional[List[List[float]]] = None
            if hasattr(f, "landmark_2d_106") and f.landmark_2d_106 is not None:
                lms = [[float(x), float(y)] for x, y in f.landmark_2d_106.tolist()]
            elif hasattr(f, "kps") and f.kps is not None:
                lms = [[float(x), float(y)] for x, y in f.kps.tolist()]
            log.info("Face detected; bbox=%s emb=%s lms=%s", bbox is not None, emb is not None, lms is not None)
            return FaceResult(bbox=bbox, embedding=emb, landmarks=lms)
        except Exception as e:
            log.exception("Face analysis failed: %s", e)
            return None

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch  # type: ignore
            return torch.cuda.is_available()
        except Exception:
            return False


class PoseBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, image_path: Path) -> Optional[Dict[str, Any]]:
        log = logging.getLogger("neurose_vton.person.pose")
        # Try ultralytics YOLOv8-pose first
        try:
            from ultralytics import YOLO  # type: ignore
            # Prefer explicit local .pt if provided/resolved
            weight = None
            if self.model_dir is not None:
                # self.model_dir may be a file or directory; handle both
                if self.model_dir.is_file():
                    weight = str(self.model_dir)
                else:
                    for cand in self.model_dir.rglob("yolov8*pose*.pt"):
                        weight = str(cand)
                        break
            model = YOLO(weight or "yolov8n-pose.pt")
            results = model(source=str(image_path), imgsz=640, device=0 if self._has_cuda() else "cpu")
            # Convert first result to BODY_25-like list
            kps = []
            for r in results:
                if hasattr(r, "keypoints") and r.keypoints is not None:
                    k = r.keypoints.xy[0].tolist()
                    kps = [[float(x), float(y)] for x, y in k]
                    break
            log.info("Pose estimated with YOLOv8; points=%d", len(kps))
            return {"format": "YOLOv8", "keypoints": kps}
        except Exception as e:
            log.warning("YOLOv8 pose failed (%s); falling back", e)
            pass
        # Fallback: unavailable
        return None

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch  # type: ignore
            return torch.cuda.is_available()
        except Exception:
            return False


class ParsingBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, image_path: Path) -> Optional[Path]:
        log = logging.getLogger("neurose_vton.person.parsing")
        try:
            # Locate SCHP repo and weights
            repo = None
            weight = None
            # Candidates for repo
            repo_candidates: List[Path] = []
            if self.model_dir:
                repo_candidates += [self.model_dir / "Self-Correction-Human-Parsing-master", self.model_dir]
            # Common locations in this repo
            repo_candidates += [
                Path("third_party/schp/Self-Correction-Human-Parsing-master").resolve(),
                Path("manual_downloads/schp_downloads/Self-Correction-Human-Parsing-master").resolve(),
            ]
            for c in repo_candidates:
                if (c / "simple_extractor.py").exists():
                    repo = c
                    break
            if repo is None:
                log.warning("SCHP repo not found; skipping parsing")
                return None
            # Search for repo folder
            weight_candidates: List[Path] = []
            # storage models
            weight_candidates += list(Path("storage/models/schp_lip").rglob("exp-schp-201908261155-lip.pth"))
            # manual downloads
            weight_candidates += [
                Path("manual_downloads/schp_downloads/exp-schp-201908261155-lip.pth").resolve(),
            ]
            if self.model_dir:
                weight_candidates += [self.model_dir / "exp-schp-201908261155-lip.pth"]
            for p in weight_candidates:
                if p.exists():
                    weight = p
                    break
            if weight is None:
                log.warning("SCHP weight not found; skipping parsing")
                return None

            # Prepare temp IO dirs
            from ..config import PATHS
            import uuid
            tmp_id = uuid.uuid4().hex[:8]
            in_dir = PATHS.runtime_cache / f"schp_in_{tmp_id}"
            out_dir = PATHS.runtime_cache / f"schp_out_{tmp_id}"
            in_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            # Copy input
            in_img = in_dir / image_path.name
            shutil.copyfile(image_path, in_img)

            # Build command
            gpu_arg = '0'
            try:
                import torch  # type: ignore
                if not torch.cuda.is_available():
                    gpu_arg = 'None'
            except Exception:
                gpu_arg = 'None'

            cmd = [
                sys.executable,
                str(repo / "simple_extractor.py"),
                "--dataset", "lip",
                "--model-restore", str(weight),
                "--gpu", gpu_arg,
                "--input-dir", str(in_dir),
                "--output-dir", str(out_dir),
            ]
            # Execute
            env = os.environ.copy()
            # Prepend repo to PYTHONPATH so its local imports resolve
            env["PYTHONPATH"] = f"{repo}:{env.get('PYTHONPATH','')}"
            proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
            if proc.returncode != 0:
                log.error("SCHP extractor failed: %s", proc.stderr[-500:])
                return None
            else:
                log.info("SCHP extractor ok: %s", proc.stdout.splitlines()[-1] if proc.stdout else "done")

            # Output PNG has same stem
            out_img = out_dir / (image_path.stem + ".png")
            if out_img.exists():
                return out_img
            return None
        except Exception as e:
            log.exception("SCHP parsing exception: %s", e)
            return None


class DepthBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, image_path: Path) -> Optional[Any]:
        log = logging.getLogger("neurose_vton.person.depth")
        # Try MiDaS via torch.hub using local cache (runtime_cache/torch/hub), seeded from storage if mounted
        try:
            import torch  # type: ignore
            import numpy as np  # type: ignore
            from PIL import Image
            try:
                torch.hub.set_dir(str(Path("/app/runtime_cache/torch/hub").resolve()))
            except Exception:
                pass
            midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid')
            midas.eval()
            transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            transform = transforms.dpt_transform
            img = Image.open(image_path).convert('RGB')
            input_batch = transform(img).unsqueeze(0)
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=img.size[::-1], mode='bicubic', align_corners=False
                ).squeeze().cpu().numpy()
            # Normalize to 0-255
            p = prediction
            p = (p - p.min()) / (p.max() - p.min() + 1e-8)
            depth_u8 = (p * 255).astype(np.uint8)
            # Compute simple normals from depth gradient
            gy, gx = np.gradient(p)
            nz = np.ones_like(p)
            normals = np.stack([gx, gy, nz], axis=-1)
            n = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
            normals = (normals / n + 1.0) * 0.5
            normals_u8 = (normals * 255).astype(np.uint8)
            log.info("Depth estimated via MiDaS cached hub")
            return {"depth": depth_u8, "normals": normals_u8}
        except Exception as e:
            log.warning("Depth backend failed (%s); using placeholder", e)
            return None


class SmplxBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, image_path: Path) -> Optional[Path]:
        log = logging.getLogger("neurose_vton.person.smplx")
        # Try simple neutral mesh export using smplx if available
        try:
            import torch  # type: ignore
            import smplx  # type: ignore
            model_dir = None
            # Prefer explicit model_dir if contains .npz
            candidates: List[Path] = []
            if self.model_dir:
                candidates.append(self.model_dir)
            candidates += [Path("manual_downloads/smplx_downloads").resolve()]
            for d in candidates:
                for name in ["SMPLX_NEUTRAL_2020.npz", "SMPLX_NEUTRAL.npz"]:
                    if (d / name).exists():
                        model_dir = d
                        break
                if model_dir:
                    break
            if model_dir is None:
                log.warning("SMPL-X neutral npz not found; skipping")
                return None
            model = smplx.create(model_path=str(model_dir), model_type='smplx', gender='neutral', use_pca=False)
            out = model()
            verts = out.vertices[0].detach().cpu().numpy()
            faces = model.faces
            from tempfile import mkdtemp
            tmpd = Path(mkdtemp())
            obj = tmpd / "smplx.obj"
            with obj.open('w') as f:
                for v in verts:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            log.info("SMPL-X mesh generated")
            return obj
        except Exception as e:
            log.warning("SMPL-X backend failed (%s); using placeholder", e)
            return None


class LightingBackend:
    def compute(self, image_path: Path) -> Optional[Dict[str, Any]]:
        # Simple placeholder: return zeros; real impl would estimate SH from normals/depth/face
        return {"sh9": [0.0] * 9}
