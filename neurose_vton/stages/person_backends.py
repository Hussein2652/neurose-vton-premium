from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import subprocess
import shutil
import sys
import os


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
        if self._lib is None:
            return None
        try:
            import numpy as np  # type: ignore
            from insightface.app import FaceAnalysis  # type: ignore

            app = FaceAnalysis(name="buffalo_l", root=str(self.model_dir) if self.model_dir else None)
            app.prepare(ctx_id=0 if self._has_cuda() else -1)
            img = _load_image(image_path)
            if img is None:
                return None
            arr = np.array(img)[:, :, ::-1]  # BGR for insightface
            faces = app.get(arr)
            if not faces:
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
            return FaceResult(bbox=bbox, embedding=emb, landmarks=lms)
        except Exception:
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
        # Try ultralytics YOLOv8-pose first
        try:
            from ultralytics import YOLO  # type: ignore
            model = YOLO("yolov8n-pose.pt")  # relies on local weights cache if present
            results = model(source=str(image_path), imgsz=640, device=0 if self._has_cuda() else "cpu")
            # Convert first result to BODY_25-like list
            kps = []
            for r in results:
                if hasattr(r, "keypoints") and r.keypoints is not None:
                    k = r.keypoints.xy[0].tolist()
                    kps = [[float(x), float(y)] for x, y in k]
                    break
            return {"format": "YOLOv8", "keypoints": kps}
        except Exception:
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
        try:
            # Locate SCHP repo and weights
            if not self.model_dir:
                return None
            # Search for repo folder
            candidates = [
                self.model_dir / "Self-Correction-Human-Parsing-master",
                self.model_dir,
            ]
            repo = None
            for c in candidates:
                if (c / "simple_extractor.py").exists():
                    repo = c
                    break
            if repo is None:
                return None
            weight = None
            for p in [
                self.model_dir / "exp-schp-201908261155-lip.pth",
                self.model_dir.parent / "manual_downloads" / "schp_downloads" / "exp-schp-201908261155-lip.pth",
            ]:
                if p.exists():
                    weight = p
                    break
            if weight is None:
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
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

            # Output PNG has same stem
            out_img = out_dir / (image_path.stem + ".png")
            if out_img.exists():
                return out_img
            return None
        except Exception:
            return None


class DepthBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, image_path: Path) -> Optional[Any]:
        # Try MiDaS via torch.hub if cached locally
        try:
            import torch  # type: ignore
            import numpy as np  # type: ignore
            from PIL import Image
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
            return {"depth": depth_u8, "normals": normals_u8}
        except Exception:
            return None


class SmplxBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, image_path: Path) -> Optional[Path]:
        # Try simple neutral mesh export using smplx if available
        try:
            import torch  # type: ignore
            import smplx  # type: ignore
            model_path = None
            if self.model_dir and self.model_dir.exists():
                # Look for SMPLX_NEUTRAL_2020.npz or SMPLX_NEUTRAL.npz
                for name in ["SMPLX_NEUTRAL.npz", "SMPLX_NEUTRAL_2020.npz"]:
                    cand = self.model_dir / name
                    if cand.exists():
                        model_path = self.model_dir
                        break
            if model_path is None:
                return None
            model = smplx.create(model_path=str(model_path), model_type='smplx', gender='neutral', use_pca=False)
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
            return obj
        except Exception:
            return None


class LightingBackend:
    def compute(self, image_path: Path) -> Optional[Dict[str, Any]]:
        # Simple placeholder: return zeros; real impl would estimate SH from normals/depth/face
        return {"sh9": [0.0] * 9}
