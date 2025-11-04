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
        # Prefer antelopev2 for stable embeddings; allow explicit override
        pack = "antelopev2"
        if self.model_dir:
            pstr = str(self.model_dir).lower()
            if "buffalo" in pstr and "antelopev2" not in pstr:
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
            try:
                app = FaceAnalysis(name=pack, root=str(writable_root), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            except Exception:
                app = FaceAnalysis(name=pack, root=str(writable_root))
            # Stable det size and CUDA ctx if available
            try:
                app.prepare(ctx_id=0 if self._has_cuda() else -1, det_size=(640, 640))
            except Exception:
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

    def compute(self, image_path: Path, target_xy: Optional[List[float]] = None) -> Optional[Dict[str, Any]]:
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
            # Strictly offline: require a local weight
            if weight is None:
                log.error("YOLOv8 pose weight not found locally; set model_dir to your yolov8-pose .pt")
                return None
            model = YOLO(weight)
            results = model(source=str(image_path), imgsz=640, device=0 if self._has_cuda() else "cpu")
            # Choose the person whose bbox center is closest to target_xy (if provided), else the first
            coco17: List[List[float]] = []
            try:
                for r in results:
                    if getattr(r, "keypoints", None) is None or getattr(r, "boxes", None) is None:
                        continue
                    k_all = r.keypoints.xy  # [N, 17, 2]
                    b_all = r.boxes.xyxy   # [N, 4]
                    n = int(getattr(k_all, 'shape', [0])[0]) if hasattr(k_all, 'shape') else 0
                    if n == 0:
                        continue
                    sel = 0
                    if target_xy is not None and len(target_xy) == 2:
                        import torch  # type: ignore
                        cx = (b_all[:, 0] + b_all[:, 2]) * 0.5
                        cy = (b_all[:, 1] + b_all[:, 3]) * 0.5
                        tx, ty = float(target_xy[0]), float(target_xy[1])
                        d = (cx - tx) ** 2 + (cy - ty) ** 2
                        sel = int(torch.argmin(d).item())
                    # Extract COCO17 for selected idx
                    k = k_all[sel].tolist()
                    coco17 = [[float(x), float(y)] for x, y in k]
                    break
            except Exception:
                # Fallback to first available keypoints tensor
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
        # Inline fast path toggle
        try:
            if os.environ.get("NEUROSE_SCHP_INLINE", "1") in {"1", "true", "True"}:
                out = self._compute_inline(image_path)
                if out is not None:
                    return out
                else:
                    if os.environ.get("NEUROSE_SCHP_STRICT", "1") in {"1", "true", "True"}:
                        log.warning("SCHP inline returned no result; STRICT enabled, skipping extractor fallback")
                        return None
                    log.warning("SCHP inline returned no result; falling back to extractor")
        except Exception as e:
            if os.environ.get("NEUROSE_SCHP_STRICT", "1") in {"1", "true", "True"}:
                log.warning("SCHP inline failed: %s; STRICT enabled, skipping extractor fallback", e)
                return None
            log.warning("SCHP inline failed: %s; falling back to extractor", e)
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
                try:
                    for cand in self.model_dir.rglob("exp-schp-201908261155-lip.pth"):
                        weight_candidates.append(cand)
                except Exception:
                    pass
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

    def _compute_inline(self, image_path: Path) -> Optional[Path]:
        """Inline SCHP parsing on GPU/CPU without spawning a subprocess.
        Loads model once and reuses it across requests. Saves a PNG next to trace.
        """
        log = logging.getLogger("neurose_vton.person.parsing")
        # Resolve repo
        repo_candidates: List[Path] = []
        if self.model_dir:
            repo_candidates += [self.model_dir / "Self-Correction-Human-Parsing-master", self.model_dir]
        repo_candidates += [
            Path("third_party/schp/Self-Correction-Human-Parsing-master").resolve(),
            Path("manual_downloads/schp_downloads/Self-Correction-Human-Parsing-master").resolve(),
        ]
        repo = None
        for c in repo_candidates:
            if (c / "networks.py").exists() or (c / "networks" / "__init__.py").exists():
                repo = c
                break
        if repo is None:
            return None
        # Resolve weight
        weight = None
        for p in [
            Path("storage/models/schp_lip").resolve(),
            Path("manual_downloads/schp_downloads").resolve(),
            self.model_dir.resolve() if self.model_dir else None,
        ]:
            if not p:
                continue
            for cand in p.rglob("exp-schp-201908261155-lip.pth"):
                weight = cand
                break
            if weight:
                break
        if weight is None or not weight.exists():
            return None

        # Lazy singleton model
        model, input_size = _SchpInlineModel.get(repo, weight)
        if model is None:
            return None
        try:
            log.info("SCHP inline using repo=%s weight=%s input=%sx%s", str(repo), str(weight), int(input_size[1]), int(input_size[0]))
        except Exception:
            pass

        # Load image via cv2 to match SCHP BGR preprocessing
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        h, w = img_bgr.shape[:2]
        # Compute center/scale like dataset
        aspect_ratio = input_size[1] * 1.0 / input_size[0]
        x, y, bw, bh = 0.0, 0.0, float(w - 1), float(h - 1)
        cx, cy = x + bw * 0.5, y + bh * 0.5
        if bw > aspect_ratio * bh:
            bh = bw * 1.0 / aspect_ratio
        elif bw < aspect_ratio * bh:
            bw = bh * aspect_ratio
        scale = np.array([bw, bh], dtype=np.float32)
        center = np.array([cx, cy], dtype=np.float32)
        # Multi-scale + flip TTA, accumulate in original resolution
        sys.path.insert(0, str(repo))
        from utils.transforms import get_affine_transform, transform_logits  # type: ignore
        import torch  # type: ignore
        use_fp16 = _SchpInlineModel.use_fp16() and str(_SchpInlineModel.device()) != 'cpu'
        # Parse scales from env
        scales_env = os.environ.get("NEUROSE_SCHP_TTA_SCALES", "1.0")
        try:
            scale_factors = [float(s.strip()) for s in scales_env.split(',') if s.strip()]
            if not scale_factors:
                scale_factors = [1.0]
        except Exception:
            scale_factors = [1.0]
        do_flip = os.environ.get("NEUROSE_SCHP_TTA_FLIP", "1") in {"1", "true", "True"}

        num_classes = 20
        logits_sum = None  # numpy (H, W, C)
        count = 0

        def forward_logits_for(center_np, scale_np, flip: bool) -> Optional[np.ndarray]:
            # Affine warp to network input size at this scale
            trans = get_affine_transform(center_np, scale_np, 0, input_size)
            inp = cv2.warpAffine(
                img_bgr,
                trans,
                (int(input_size[1]), int(input_size[0])),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            if flip:
                inp = cv2.flip(inp, 1)
            arr = inp.astype(np.float32) / 255.0
            mean = np.array([0.406, 0.456, 0.485], dtype=np.float32)
            std = np.array([0.225, 0.224, 0.229], dtype=np.float32)
            arr = (arr - mean) / std
            chw = arr.transpose(2, 0, 1)  # CHW (BGR)
            x_t = torch.from_numpy(chw).unsqueeze(0).to(_SchpInlineModel.device())
            if use_fp16:
                x_t = x_t.half()
            with torch.no_grad():
                o = model(x_t)
            # Extract logits tensor robustly, matching simple_extractor: output[0][-1][0]
            logit_t = None
            if isinstance(o, (list, tuple)) and len(o) > 0:
                o0 = o[0]
                if isinstance(o0, (list, tuple)) and len(o0) > 0:
                    last = o0[-1]
                    logit_t = last[0] if isinstance(last, (list, tuple)) and len(last) > 0 else last
                else:
                    logit_t = o0
            elif isinstance(o, dict):
                logit_t = o.get('out') or o.get('logits') or o.get('result') or next(iter(o.values()))
            else:
                logit_t = o
            if isinstance(logit_t, (list, tuple)) and len(logit_t) > 0:
                logit_t = logit_t[0]
            if not (isinstance(logit_t, torch.Tensor) and logit_t.dim() == 4):
                return None
            logit_t = torch.nn.functional.interpolate(logit_t.float(), size=tuple(input_size.tolist()), mode='bilinear', align_corners=True)
            if flip:
                # Unflip logits spatially and swap L/R class channels
                logit_t = torch.flip(logit_t, dims=[3])
                idx = list(range(num_classes))
                # Swap 14<->15 (arms), 16<->17 (legs), 18<->19 (shoes)
                if num_classes >= 20:
                    idx[14], idx[15] = idx[15], idx[14]
                    idx[16], idx[17] = idx[17], idx[16]
                    idx[18], idx[19] = idx[19], idx[18]
                import torch as _t
                idx_t = _t.tensor(idx, dtype=_t.long, device=logit_t.device)
                logit_t = logit_t[:, idx_t, :, :]
            logits_hw = logit_t[0].permute(1, 2, 0).cpu().numpy()  # HWC
            # Map back to original resolution for this scale
            logits_res = transform_logits(logits_hw, center_np, scale_np, w, h, input_size=input_size.tolist())
            return logits_res.astype(np.float32, copy=False)

        for sf in scale_factors:
            scale_f = scale * float(sf)
            # base
            lr = forward_logits_for(center, scale_f, flip=False)
            if lr is not None:
                logits_sum = (lr if logits_sum is None else logits_sum + lr)
                count += 1
            # flip
            if do_flip:
                lr_f = forward_logits_for(center, scale_f, flip=True)
                if lr_f is not None:
                    logits_sum = (lr_f if logits_sum is None else logits_sum + lr_f)
                    count += 1
        if logits_sum is None or count == 0:
            return None
        logits_avg = logits_sum / float(count)
        parsing = np.argmax(logits_avg, axis=2).astype(np.uint8)
        # Save next to trace output (handled by caller) with color palette like SCHP
        from tempfile import mkdtemp
        tmpd = Path(mkdtemp())
        out_png = tmpd / (image_path.stem + ".png")
        from PIL import Image
        # Build palette (same as SCHP get_palette)
        num_classes = 20
        palette = [0] * (num_classes * 3)
        for j in range(num_classes):
            lab = j
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        img_p = Image.fromarray(parsing, mode='P')
        img_p.putpalette(palette)
        img_p.save(out_png)
        return out_png


class _SchpInlineModel:
    _cache: Dict[str, Any] = {}

    @classmethod
    def device(cls):
        try:
            import torch  # type: ignore
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            class _D:  # type: ignore
                pass
            return "cpu"

    @classmethod
    def use_fp16(cls) -> bool:
        return os.environ.get("NEUROSE_SCHP_FP16", "1") in {"1", "true", "True"}

    @classmethod
    def get(cls, repo: Path, weight: Path):
        key = f"{repo}::{weight}"
        if key in cls._cache:
            return cls._cache[key]
        # Build model
        sys.path.insert(0, str(repo))
        try:
            import torch  # type: ignore
            import numpy as np  # type: ignore
            import importlib
            networks = importlib.import_module("networks")
            model = networks.init_model('resnet101', num_classes=20, pretrained=None)  # LIP 20 classes
            state = torch.load(str(weight), map_location="cpu").get('state_dict', {})
            from collections import OrderedDict
            new_state = OrderedDict()
            for k, v in state.items():
                name = k[7:] if k.startswith('module.') else k
                new_state[name] = v
            model.load_state_dict(new_state, strict=False)
            dev = cls.device()
            model.to(dev).eval()
            if cls.use_fp16() and str(dev) != 'cpu':
                model.half()
            # Input size for LIP
            input_size = np.asarray([473, 473], dtype=np.int32)
            cls._cache[key] = (model, input_size)
            return cls._cache[key]
        except Exception as e:
            logging.getLogger("neurose_vton.person.parsing").warning("Inline SCHP init failed: %s", e)
            return (None, None)


class DepthBackend:
    _disabled: bool = False  # latch to avoid repeated ZoeDepth attempts after a hard failure
    def __init__(self, model_dir: Optional[Path]) -> None:
        self.model_dir = model_dir

    def compute(self, image_path: Path) -> Optional[Any]:
        log = logging.getLogger("neurose_vton.person.depth")
        # 1) Try ZoeDepth with local checkpoint; strictly offline (no runtime installs)
        # Allow user to skip via env for speed/stability
        skip_zoe = os.environ.get("NEUROSE_SKIP_ZOEDEPTH", "0") in {"1", "true", "True"}
        if not skip_zoe and not DepthBackend._disabled:
            try:
                res = self._compute_zoedepth(image_path)
                if res is not None:
                    return res
            except Exception as e:
                log.warning("ZoeDepth failed (%s); falling back to MiDaS", e)
                DepthBackend._disabled = True
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
            # Allow variant selection via env, default to highest quality
            variant = os.environ.get("NEUROSE_MIDAS_VARIANT", "DPT_BEiT_L_384").strip()
            try:
                midas = torch.hub.load('intel-isl/MiDaS', variant)
            except Exception:
                # Fallback to DPT_Hybrid if selected variant missing
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
        # Helper to disable stochastic depth / droppath regardless of timm version
        import torch.nn as nn  # type: ignore
        def _disable_droppath_config(conf_obj: Any) -> None:
            try:
                # common keys used by zoedepth variants
                for k in ("drop_path_rate", "stochastic_depth"):
                    if hasattr(conf_obj, k):
                        try:
                            setattr(conf_obj, k, 0.0)
                        except Exception:
                            pass
                # nested dicts
                for key in ("zoedepth", "zoedepth_nk"):
                    d = getattr(conf_obj, key, None)
                    if isinstance(d, dict):
                        d["drop_path_rate"] = 0.0
            except Exception:
                pass
        def _replace_droppath_modules(m: nn.Module) -> None:
            try:
                try:
                    from timm.models.layers import DropPath as DP  # type: ignore
                except Exception:
                    try:
                        from timm.layers import DropPath as DP  # type: ignore
                    except Exception:
                        DP = None  # type: ignore
                for name, child in list(m.named_children()):
                    try:
                        if hasattr(child, 'drop_path') and child.drop_path is not None:
                            setattr(child, 'drop_path', nn.Identity())
                    except Exception:
                        pass
                    if DP is not None:
                        try:
                            if isinstance(child, DP):
                                setattr(m, name, nn.Identity())
                                child = getattr(m, name)
                        except Exception:
                            pass
                    _replace_droppath_modules(child)
            except Exception:
                pass
        # 1) Resolve checkpoint
        ckpt: Optional[Path] = None
        if self.model_dir:
            if self.model_dir.is_file():
                ckpt = self.model_dir
            else:
                for pat in ("**/ZoeD_*NK*.pt", "**/ZoeD_*.pt"):
                    try:
                        found = next(self.model_dir.glob(pat))
                        ckpt = found
                        break
                    except StopIteration:
                        pass
        if ckpt is None:
            cand = Path("storage/models/zoedepth_m12_nk/v1/ZoeD_M12_NK.pt").resolve()
            if cand.exists():
                ckpt = cand
        if ckpt is None or not ckpt.exists():
            return None

        # 2a) Try loading via local hubconf using a local pretrained_resource (preferred)
        repo_paths: List[Path] = []
        env_repo = os.environ.get("NEUROSE_ZOEDEPTH_REPO")
        if env_repo:
            repo_paths.append(Path(env_repo).resolve())
        repo_paths.append(Path("third_party/zoedepth").resolve())
        cache_repo = (PATHS.runtime_cache / "zoedepth_repo").resolve()
        repo_paths.append(cache_repo)

        variant = "zoedepth_nk" if "NK" in ckpt.name.upper() else "zoedepth"
        hub_entries = ["ZoeD_M12_NK", "ZoeD_NK"] if variant == "zoedepth_nk" else ["ZoeD_M12_N", "ZoeD_N"]
        for repo in repo_paths:
            hubconf = repo / "hubconf.py"
            if hubconf.exists():
                for entry in hub_entries:
                    try:
                        model = torch.hub.load(str(repo), entry, source='local', pretrained=True,
                                               config_mode='infer', pretrained_resource=f"local::{ckpt}")
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model.to(device).eval()
                        # Ensure missing drop_path attributes are present across all submodules
                        try:
                            import torch.nn as nn  # type: ignore
                            for mod in getattr(model, 'modules', lambda: [])():
                                try:
                                    setattr(mod, 'drop_path', nn.Identity())
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # Inference per README: prefer model.infer_pil / infer
                        img = Image.open(image_path).convert('RGB')
                        try:
                            if hasattr(model, 'infer_pil'):
                                depth_np = model.infer_pil(img)
                            elif hasattr(model, 'infer'):
                                from zoedepth.utils.misc import pil_to_batched_tensor  # type: ignore
                                X = pil_to_batched_tensor(img).to(device)
                                depth_t = model.infer(X)
                                depth_np = depth_t.squeeze().detach().cpu().float().numpy()
                            else:
                                # raw forward as last resort
                                arr = np.asarray(img).astype('float32') / 255.0
                                H, W = arr.shape[:2]
                                x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
                                with torch.inference_mode():
                                    out = model(x)
                                    if isinstance(out, (list, tuple)):
                                        depth = out[0]
                                    elif isinstance(out, dict):
                                        depth = out.get('metric_depth') or out.get('depth') or next(iter(out.values()))
                                    else:
                                        depth = out
                                    depth_np = torch.nn.functional.interpolate(depth, size=(H, W), mode='bicubic', align_corners=False).squeeze().detach().cpu().float().numpy()
                        except Exception:
                            # fallback to raw forward
                            arr = np.asarray(img).astype('float32') / 255.0
                            H, W = arr.shape[:2]
                            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
                            with torch.inference_mode():
                                out = model(x)
                                if isinstance(out, (list, tuple)):
                                    depth = out[0]
                                elif isinstance(out, dict):
                                    depth = out.get('metric_depth') or out.get('depth') or next(iter(out.values()))
                                else:
                                    depth = out
                                depth_np = torch.nn.functional.interpolate(depth, size=(H, W), mode='bicubic', align_corners=False).squeeze().detach().cpu().float().numpy()
                        p = depth_np
                        pmin, pmax = float(p.min()), float(p.max())
                        p = (p - pmin) / (pmax - pmin + 1e-8)
                        depth_u8 = (p * 255.0).astype(np.uint8)
                        gy, gx = np.gradient(p)
                        nz = np.ones_like(p)
                        normals = np.stack([gx, gy, nz], axis=-1)
                        n = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
                        normals = (normals / n + 1.0) * 0.5
                        normals_u8 = (normals * 255.0).astype(np.uint8)
                        return {"depth": depth_u8, "normals": normals_u8, "_backend": ("zoedepth_m12_nk" if variant == "zoedepth_nk" else "zoedepth")}
                    except Exception:
                        continue
        # 2b) Import zoedepth package; if missing, try local repo via NEUROSE_ZOEDEPTH_REPO and third_party
        build_model = None
        zget_config = None
        try:
            from zoedepth.models.builder import build_model as _bm  # type: ignore
            from zoedepth.utils.config import get_config as _gc  # type: ignore
            build_model, zget_config = _bm, _gc
        except Exception:
            for repo in repo_paths:
                if not repo.exists():
                    continue
                sys.path.insert(0, str(repo))
                try:
                    subs = list(repo.iterdir())
                    for s in subs:
                        if s.is_dir():
                            sys.path.insert(0, str(s))
                            break
                except Exception:
                    pass
            from zoedepth.models.builder import build_model as _bm  # type: ignore
            from zoedepth.utils.config import get_config as _gc  # type: ignore
            build_model, zget_config = _bm, _gc

        # 3) Build config and model using local checkpoint (no network)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        conf = zget_config(variant, "infer")
        local_spec = f"local::{ckpt}"
        try:
            if hasattr(conf, "pretrained_resource"):
                setattr(conf, "pretrained_resource", local_spec)
            if hasattr(conf, "zoedepth") and isinstance(conf.zoedepth, dict):
                conf.zoedepth["pretrained_resource"] = local_spec
            if hasattr(conf, "zoedepth_nk") and isinstance(conf.zoedepth_nk, dict):
                conf.zoedepth_nk["pretrained_resource"] = local_spec
        except Exception:
            pass
        # Try to disable droppath in config before build
        _disable_droppath_config(conf)
        try:
            model = build_model(conf).to(device).eval()
        except Exception as e:
            logging.getLogger("neurose_vton.person.depth").warning("ZoeDepth build failed: %s", e)
            # Retry build without auto-loading weights; we'll load state manually
            try:
                if hasattr(conf, "pretrained_resource"):
                    setattr(conf, "pretrained_resource", None)
                if hasattr(conf, "zoedepth") and isinstance(conf.zoedepth, dict):
                    conf.zoedepth["pretrained_resource"] = None
                if hasattr(conf, "zoedepth_nk") and isinstance(conf.zoedepth_nk, dict):
                    conf.zoedepth_nk["pretrained_resource"] = None
            except Exception:
                pass
            model = build_model(conf).to(device).eval()
        # Disable droppath modules/attrs post-build
        _replace_droppath_modules(model)
        # 4) Load checkpoint
        sd = torch.load(str(ckpt), map_location=device)
        state = sd.get("state_dict", sd.get("model", sd))
        # Sanitize incoming state: keep only matching keys/shapes
        try:
            target = model.state_dict()
            filtered = {}
            dropped = 0
            for k, v in state.items():
                if k in target and hasattr(v, 'shape') and hasattr(target[k], 'shape') and tuple(v.shape) == tuple(target[k].shape):
                    filtered[k] = v
                else:
                    dropped += 1
            if dropped:
                logging.getLogger("neurose_vton.person.depth").warning("ZoeDepth state filtered: kept=%d dropped=%d", len(filtered), dropped)
            state = filtered if filtered else state
        except Exception:
            pass
        try:
            import sys as _sys
            _sys.setrecursionlimit(max(_sys.getrecursionlimit(), 10000))
        except Exception:
            pass
        try:
            model.load_state_dict(state, strict=False)
        except Exception as e:
            logging.getLogger("neurose_vton.person.depth").warning("ZoeDepth load_state failed: %s — applying drop_path patch", e)
            # Patch missing drop_path in some repos
            try:
                import torch.nn as nn  # type: ignore
                for mod in getattr(model, 'modules', lambda: [])():
                    if not hasattr(mod, 'drop_path'):
                        try:
                            setattr(mod, 'drop_path', nn.Identity())
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                model.load_state_dict(state, strict=False)
            except Exception:
                # Last resort: attempt loading without positional index tensors
                try:
                    state2 = {k: v for k, v in state.items() if 'relative_position_index' not in k}
                    model.load_state_dict(state2, strict=False)
                except Exception:
                    raise

        # 5) Inference: prefer public API (infer_pil / infer) per README
        img = Image.open(image_path).convert('RGB')
        arr = np.asarray(img).astype('float32') / 255.0
        H, W = arr.shape[:2]
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        try:
            with torch.inference_mode():
                if hasattr(model, 'infer_pil'):
                    depth_np = model.infer_pil(img)
                elif hasattr(model, 'infer'):
                    from zoedepth.utils.misc import pil_to_batched_tensor  # type: ignore
                    X = pil_to_batched_tensor(img).to(device)
                    depth_t = model.infer(X)
                    depth_np = depth_t.squeeze().detach().cpu().float().numpy()
                else:
                    out = model(x)
                    if isinstance(out, (list, tuple)):
                        depth = out[0]
                    elif isinstance(out, dict):
                        depth = out.get('metric_depth') or out.get('depth') or next(iter(out.values()))
                    else:
                        depth = out
                    depth_np = torch.nn.functional.interpolate(depth, size=(H, W), mode='bicubic', align_corners=False).squeeze().detach().cpu().float().numpy()
        except Exception as e:
            log = logging.getLogger("neurose_vton.person.depth")
            log.warning("ZoeDepth forward failed: %s", e)
            # Retry once after aggressively setting drop_path on all modules/classes
            try:
                import torch.nn as nn  # type: ignore
                for mod in getattr(model, 'modules', lambda: [])():
                    if not hasattr(mod, 'drop_path'):
                        try:
                            setattr(mod, 'drop_path', nn.Identity())
                        except Exception:
                            try:
                                setattr(mod.__class__, 'drop_path', nn.Identity())
                            except Exception:
                                pass
                with torch.inference_mode():
                    if hasattr(model, 'infer_pil'):
                        depth_np = model.infer_pil(img)
                    elif hasattr(model, 'infer'):
                        from zoedepth.utils.misc import pil_to_batched_tensor  # type: ignore
                        X = pil_to_batched_tensor(img).to(device)
                        depth_t = model.infer(X)
                        depth_np = depth_t.squeeze().detach().cpu().float().numpy()
                    else:
                        out = model(x)
                        if isinstance(out, (list, tuple)):
                            depth = out[0]
                        elif isinstance(out, dict):
                            depth = out.get('metric_depth') or out.get('depth') or next(iter(out.values()))
                        else:
                            depth = out
                        depth_np = torch.nn.functional.interpolate(depth, size=(H, W), mode='bicubic', align_corners=False).squeeze().detach().cpu().float().numpy()
            except Exception:
                raise
        # 6) Normalize and normals
        p = depth_np
        pmin, pmax = float(p.min()), float(p.max())
        p = (p - pmin) / (pmax - pmin + 1e-8)
        depth_u8 = (p * 255.0).astype(np.uint8)
        gy, gx = np.gradient(p)
        nz = np.ones_like(p)
        normals = np.stack([gx, gy, nz], axis=-1)
        n = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
        normals = (normals / n + 1.0) * 0.5
        normals_u8 = (normals * 255.0).astype(np.uint8)
        return {"depth": depth_u8, "normals": normals_u8, "_backend": ("zoedepth_m12_nk" if "NK" in ckpt.name.upper() else "zoedepth")}
        # Patch missing drop_path in some repos
        try:
            import torch.nn as nn  # type: ignore
            for mod in getattr(model, 'modules', lambda: [])():
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
