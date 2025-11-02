from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import subprocess
import shutil
import sys
import os
import logging
from ..config import PATHS


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

    def _prepare_models(self) -> tuple[str, Optional[Path]]:
        """Return (pack_name, writable_root) for InsightFace models.
        If storage models are available under a read-only mount, copy them into
        a writable cache: PATHS.runtime_cache/insightface/models/<pack>.
        """
        # Decide pack name based on provided model_dir path
        pack = "buffalo_l"
        if self.model_dir:
            pstr = str(self.model_dir).lower()
            if "antelopev2" in pstr:
                pack = "antelopev2"
            elif "buffalo" in pstr:
                pack = "buffalo_l"
        writable_root = PATHS.runtime_cache / "insightface"
        try:
            if self.model_dir and self.model_dir.exists():
                src = self.model_dir
                dst = writable_root / "models" / pack
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    # Copy model files into writable cache
                    if src.is_dir():
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        dst.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst / src.name)
        except Exception as e:
            logging.getLogger("neurose_vton.person.face").warning("Model seed failed: %s", e)
        return pack, writable_root

    def compute(self, image_path: Path) -> Optional[FaceResult]:
        log = logging.getLogger("neurose_vton.person.face")
        if self._lib is None:
            log.warning("InsightFace not installed; skipping face analysis")
            return None
        try:
            import numpy as np  # type: ignore
            from insightface.app import FaceAnalysis  # type: ignore
            # Prepare models under writable cache root to avoid read-only mounts
            pack, writable_root = self._prepare_models()
            app = FaceAnalysis(name=pack, root=str(writable_root))
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
            # Convert first result to COCO17 list
            coco17: List[List[float]] = []
            for r in results:
                if hasattr(r, "keypoints") and r.keypoints is not None:
                    k = r.keypoints.xy[0].tolist()
                    coco17 = [[float(x), float(y)] for x, y in k]
                    break
            log.info("Pose estimated with YOLOv8; points=%d", len(coco17))
            # Derive BODY_25 approximation
            def _mid(a: List[float], b: List[float]) -> List[float]:
                return [float((a[0] + b[0]) / 2.0), float((a[1] + b[1]) / 2.0)]
            body25: List[List[float]] = []
            if len(coco17) >= 17:
                nose = coco17[0]
                leye, reye = coco17[1], coco17[2]
                lear, rear = coco17[3], coco17[4]
                lsho, rsho = coco17[5], coco17[6]
                lelb, relb = coco17[7], coco17[8]
                lwri, rwri = coco17[9], coco17[10]
                lhip, rhip = coco17[11], coco17[12]
                lknee, rknee = coco17[13], coco17[14]
                lank, rank = coco17[15], coco17[16]
                neck = _mid(lsho, rsho)
                midhip = _mid(lhip, rhip)
                # Map as close as possible
                body25 = [
                    nose,                 # 0 Nose
                    neck,                 # 1 Neck
                    rsho, relb, rwri,    # 2-4 Right arm
                    lsho, lelb, lwri,    # 5-7 Left arm
                    midhip,               # 8 MidHip
                    rhip, rknee, rank,    # 9-11 Right leg
                    lhip, lknee, lank,    # 12-14 Left leg
                    reye, leye, rear, lear,  # 15-18 facial
                    lank, lank, lank,     # 19-21 Left foot (approx from ankle)
                    rank, rank, rank,     # 22-24 Right foot (approx from ankle)
                ]
            return {
                "format": "yolov8+coco17_to_body25",
                "yolo": {"format": "COCO17", "keypoints": coco17},
                "body_25": {"format": "BODY_25", "keypoints": body25, "approx": True},
            }
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
            # Prefer GPU if available; requires CUDA toolkit in image
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
            # Ensure CUDA_HOME for torch cpp extensions
            env.setdefault("CUDA_HOME", "/usr/local/cuda")
            env.setdefault("PATH", f"/usr/local/cuda/bin:{env.get('PATH','')}")
            env.setdefault("LD_LIBRARY_PATH", f"/usr/local/cuda/lib64:{env.get('LD_LIBRARY_PATH','')}")
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
            # Handle .jpeg quirk in SCHP which slices last 4 chars only
            name_lower = image_path.name.lower()
            for ext in (".jpeg", ".jpg", ".png"):
                if name_lower.endswith(ext):
                    base = image_path.name[: -len(ext)]
                    alt = out_dir / (base + ".png")
                    if alt.exists():
                        return alt
                    break
            # Fallback: return the first png in out_dir if any
            cand = list(out_dir.glob("*.png"))
            if cand:
                return cand[0]
            return None
        except Exception as e:
            log.exception("SCHP parsing exception: %s", e)
            return None


class DepthBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, image_path: Path) -> Optional[Any]:
        log = logging.getLogger("neurose_vton.person.depth")
        # 1) Try ZoeDepth with local checkpoint; strictly offline (no runtime installs)
        # Allow user to skip via env for speed/stability
        skip_zoe = os.environ.get("NEUROSE_SKIP_ZOEDEPTH", "0") in {"1", "true", "True"}
        if not skip_zoe:
            try:
                res = self._compute_zoedepth(image_path)
                if res is not None:
                    return res
            except Exception as e:
                log.warning("ZoeDepth failed (%s); falling back to MiDaS", e)
        else:
            log.info("Skipping ZoeDepth due to NEUROSE_SKIP_ZOEDEPTH")

        # 2) Fallback: MiDaS via torch.hub (cached)
        try:
            import torch  # type: ignore
            import numpy as np  # type: ignore
            from PIL import Image
            try:
                storage_ckpt = Path("storage/models/torch/hub/checkpoints/dpt_hybrid_384.pt").resolve()
                runtime_ckpt = Path("/app/runtime_cache/torch/hub/checkpoints/dpt_hybrid_384.pt").resolve()
                if storage_ckpt.exists() and not runtime_ckpt.exists():
                    runtime_ckpt.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(storage_ckpt, runtime_ckpt)
            except Exception:
                pass
            try:
                torch.hub.set_dir(str(Path("/app/runtime_cache/torch/hub").resolve()))
            except Exception:
                pass
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid')
            midas.to(device).eval()
            img = Image.open(image_path).convert('RGB')
            W, H = img.size
            # Manual preprocessing (ImageNet normalization, 384x384)
            import numpy as np  # type: ignore
            im_resized = img.resize((384, 384))
            arr = np.asarray(im_resized).astype('float32') / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype='float32')
            std = np.array([0.229, 0.224, 0.225], dtype='float32')
            arr = (arr - mean) / std
            chw = arr.transpose(2, 0, 1)  # CHW
            input_batch = torch.from_numpy(chw).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = midas(input_batch)
                if pred.dim() == 3:
                    pred = pred.unsqueeze(1)
                elif pred.dim() == 4 and pred.shape[1] != 1:
                    pred = pred[:, :1, ...]
                pred = torch.nn.functional.interpolate(pred, size=(H, W), mode='bicubic', align_corners=False)
            p = pred[0, 0].detach().cpu().numpy()
            pmin = float(p.min())
            pmax = float(p.max())
            p = (p - pmin) / (pmax - pmin + 1e-8)
            depth_u8 = (p * 255.0).astype(np.uint8)
            # Compute normals from 2D depth map
            gy, gx = np.gradient(p)
            nz = np.ones_like(p)
            normals = np.stack([gx, gy, nz], axis=-1)
            n = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
            normals = (normals / n + 1.0) * 0.5
            normals_u8 = (normals * 255.0).astype(np.uint8)
            log.info("Depth estimated via MiDaS cached hub")
            return {"depth": depth_u8, "normals": normals_u8, "_backend": "midas"}
        except Exception as e:
            log.warning("Depth backend failed (%s); using placeholder", e)
            return None


    def _compute_zoedepth(self, image_path: Path) -> Optional[Any]:
        import torch  # type: ignore
        import numpy as np  # type: ignore
        from PIL import Image
        # Resolve checkpoint: prefer provided model_dir; else known storage path
        ckpt: Optional[Path] = None
        candidates: List[Path] = []
        if self.model_dir:
            if self.model_dir.is_file():
                candidates.append(self.model_dir)
            else:
                candidates += list(self.model_dir.rglob("ZoeD_*NK*.pt"))
                candidates += list(self.model_dir.rglob("ZoeD_*.pt"))
        candidates.append(Path("storage/models/zoedepth_m12_nk/v1/ZoeD_M12_NK.pt").resolve())
        for c in candidates:
            if c.exists():
                ckpt = c
                break
        if not ckpt or not ckpt.exists():
            return None
        # Import zoedepth either from installed site-packages or local folders
        try:
            from zoedepth.models.builder import build_model  # type: ignore
            from zoedepth.utils.config import get_config  # type: ignore
        except Exception:
            # Try local candidates in order: third_party, cached extracted zip
            # Highest priority: explicit env override
            env_repo = os.environ.get("NEUROSE_ZOEDEPTH_REPO")
            local_repo = Path(env_repo).resolve() if env_repo else Path("third_party/zoedepth").resolve()
            cache_repo = (PATHS.runtime_cache / "zoedepth_repo").resolve()
            if not local_repo.exists():
                # If a zip exists under manual_downloads, extract it into runtime_cache (no writes to mounts)
                zips: List[Path] = []
                try:
                    manual_zroot = (PATHS.manual / "zoedepth_downloads").resolve()
                    if manual_zroot.exists():
                        for z in manual_zroot.glob("ZoeDepth*.zip"):
                            zips.append(z)
                except Exception:
                    pass
                if zips and not cache_repo.exists():
                    import zipfile
                    cache_repo.mkdir(parents=True, exist_ok=True)
                    try:
                        with zipfile.ZipFile(str(zips[0]), 'r') as zf:
                            zf.extractall(str(cache_repo))
                    except Exception:
                        pass
            # Prefer third_party if present; else cached extracted
            if local_repo.exists():
                sys.path.insert(0, str(local_repo))
            elif cache_repo.exists():
                # The zip typically unpacks to a folder like ZoeDepth-main
                # Add both cache_repo and its first-level subdir if exists
                sys.path.insert(0, str(cache_repo))
                try:
                    subs = list(cache_repo.iterdir())
                    for s in subs:
                        if s.is_dir():
                            sys.path.insert(0, str(s))
                            break
                except Exception:
                    pass
            from zoedepth.models.builder import build_model  # type: ignore
            from zoedepth.utils.config import get_config  # type: ignore

        # Pick NK variant if checkpoint name suggests it; prevents wrong weights mapping
        variant = "zoedepth_nk" if "NK" in ckpt.name.upper() else "zoedepth"
        conf = get_config(variant, "infer")
        # Best-effort to prevent any pretrained URL download from config
        for key in ("pretrained_resource", "pretrained_model", "pretrained_resource_model"):
            try:
                setattr(conf, key, None)
            except Exception:
                pass
        try:
            if hasattr(conf, "zoedepth") and isinstance(conf.zoedepth, dict):
                conf.zoedepth["pretrained_resource"] = None
            if hasattr(conf, "zoedepth_nk") and isinstance(conf.zoedepth_nk, dict):
                conf.zoedepth_nk["pretrained_resource"] = None
        except Exception:
            pass
        model = build_model(conf)
        # Patch missing drop_path in some repos
        try:
            import torch.nn as nn  # type: ignore
            for mod in model.modules():
                if not hasattr(mod, 'drop_path'):
                    try:
                        setattr(mod, 'drop_path', nn.Identity())
                    except Exception:
                        pass
        except Exception:
            pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        sd = torch.load(str(ckpt), map_location=device)
        state = sd.get("state_dict", sd)
        model.load_state_dict(state, strict=False)

        img = Image.open(image_path).convert('RGB')
        arr = np.asarray(img).astype('float32') / 255.0
        arr = arr.transpose(2, 0, 1)
        input_tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
        # Self-test forward
        with torch.no_grad():
            _ = model(torch.zeros(1, 3, 64, 64, device=device))
            out = model(input_tensor)
            if isinstance(out, (list, tuple)):
                depth = out[0]
            elif hasattr(out, 'keys'):
                depth = out.get('depth') or list(out.values())[0]
            else:
                depth = out
            if hasattr(depth, 'detach'):
                depth = depth.detach().squeeze()
            if hasattr(depth, 'cpu'):
                depth = depth.cpu().numpy()
        p = np.asarray(depth)
        pmin, pmax = float(p.min()), float(p.max())
        p = (p - pmin) / (pmax - pmin + 1e-8)
        depth_u8 = (p * 255.0).astype(np.uint8)
        gy, gx = np.gradient(p)
        nz = np.ones_like(p)
        normals = np.stack([gx, gy, nz], axis=-1)
        n = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
        normals = (normals / n + 1.0) * 0.5
        normals_u8 = (normals * 255.0).astype(np.uint8)
        return {"depth": depth_u8, "normals": normals_u8, "_backend": "zoedepth"}

class SmplxBackend:
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, image_path: Path) -> Optional[Path]:
        log = logging.getLogger("neurose_vton.person.smplx")
        # Try simple neutral mesh export using smplx if available
        try:
            import torch  # type: ignore
            import smplx  # type: ignore
            # Build a cache model directory with expected structure: <root>/smplx/SMPLX_NEUTRAL_2020.npz
            cache_root = PATHS.runtime_cache / "smplx_models"
            target_dir = cache_root / "smplx"
            target_dir.mkdir(parents=True, exist_ok=True)
            # Locate source npz
            src_npz = None
            candidates: List[Path] = []
            if self.model_dir:
                candidates.append(self.model_dir)
            candidates += [Path("manual_downloads/smplx_downloads").resolve()]
            for d in candidates:
                for name in ["SMPLX_NEUTRAL_2020.npz", "SMPLX_NEUTRAL.npz"]:
                    if (d / name).exists():
                        src_npz = d / name
                        break
                if src_npz:
                    break
            if src_npz is None:
                log.warning("SMPL-X neutral npz not found; skipping")
                return None
            # Copy into cache structure
            dst_npz = target_dir / src_npz.name
            if not dst_npz.exists():
                shutil.copy2(src_npz, dst_npz)
            # Some smplx versions expect SMPLX_NEUTRAL.npz specifically
            neutral_alias = target_dir / "SMPLX_NEUTRAL.npz"
            if not neutral_alias.exists():
                shutil.copy2(dst_npz, neutral_alias)
            model = smplx.create(model_path=str(cache_root), model_type='smplx', gender='neutral', use_pca=False)
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
