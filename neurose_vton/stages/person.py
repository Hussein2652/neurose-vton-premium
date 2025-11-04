from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import time
import json
import logging

from .base import StageOutput
from ..registry import registry
from .person_backends import (
    FaceIDBackend,
    PoseBackend,
    ParsingBackend,
    DepthBackend,
    SmplxBackend,
    LightingBackend,
)
from ..config import PATHS


class PersonAnalysis:
    def __init__(self) -> None:
        pass

    def _save_json(self, path: Path, obj: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2))

    def _save_blank_png(self, path: Path, w: int = 64, h: int = 64, channels: int = 1) -> None:
        try:
            from PIL import Image
            import numpy as np
            arr = np.zeros((h, w, channels), dtype=np.uint8)
            if channels == 1:
                arr = arr.squeeze(-1)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr).save(path)
        except Exception:
            # If PIL not available, create an empty marker file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")

    def _save_placeholder_mesh(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Minimal valid OBJ with a single triangle as placeholder
        content = """
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1 2 3
""".strip()
        path.write_text(content)

    def _save_occluder_masks(self, seg_path: Path, out_dir: Path) -> None:
        hair = out_dir / "hair_mask.png"
        arms = out_dir / "arms_mask.png"
        hands = out_dir / "hands_mask.png"
        try:
            from PIL import Image
            import numpy as np
            import cv2  # type: ignore
            seg = Image.open(seg_path)
            arr = np.array(seg)
            # LIP label indices per SCHP (see dataset_settings in SCHP):
            # Hair=2, Left-arm=14, Right-arm=15, Glove=3 approximates hands
            hair_mask = (arr == 2).astype(np.uint8) * 255
            arms_mask = ((arr == 14) | (arr == 15)).astype(np.uint8) * 255
            hands_mask = (arr == 3).astype(np.uint8) * 255
            # Light morphological closing to remove pinholes and thin gaps
            kernel = np.ones((3, 3), np.uint8)
            hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
            hands_mask = cv2.morphologyEx(hands_mask, cv2.MORPH_CLOSE, kernel)
            Image.fromarray(hair_mask).save(hair)
            Image.fromarray(arms_mask).save(arms)
            Image.fromarray(hands_mask).save(hands)
        except Exception:
            # Create placeholders if segmentation not loadable
            self._save_blank_png(hair, 64, 64, 1)
            self._save_blank_png(arms, 64, 64, 1)
            self._save_blank_png(hands, 64, 64, 1)

    def _save_seg_overlay(self, image_path: Path, seg_path: Path, out_path: Path, alpha: float = 0.4) -> None:
        try:
            from PIL import Image
            import numpy as np
            base = Image.open(image_path).convert("RGB")
            seg = Image.open(seg_path)
            seg = seg.resize(base.size, Image.NEAREST)
            seg_arr = np.array(seg)
            # Simple color map for up to 20 classes
            palette = np.array([
                [0,0,0],[255,0,0],[255,128,0],[255,255,0],[128,255,0],[0,255,0],
                [0,255,128],[0,255,255],[0,128,255],[0,0,255],[128,0,255],[255,0,255],
                [255,0,128],[128,128,128],[128,64,0],[0,128,64],[64,0,128],[0,64,128],
                [192,192,0],[0,192,192]
            ], dtype=np.uint8)
            colors = palette[seg_arr % len(palette)]
            overlay = Image.fromarray(colors).convert("RGBA")
            # Apply alpha
            a = int(max(0.0, min(1.0, alpha)) * 255)
            overlay.putalpha(a)
            out = base.convert("RGBA")
            out = Image.alpha_composite(out, overlay)
            out.convert("RGB").save(out_path)
        except Exception:
            pass

    def run(self, image_path: Path, seed: int, trace_dir: Optional[Path] = None) -> StageOutput:
        log = logging.getLogger("neurose_vton.person")
        log.info("PersonAnalysis start | image=%s seed=%s trace=%s", image_path, seed, bool(trace_dir))
        # Resolve model availability via registry (no heavy loads here)
        resolved = {
            "insightface": str(registry.resolve("insightface") or ""),
            "openpose": str(registry.resolve("openpose") or ""),
            "schp": str(registry.resolve("schp") or ""),
            "depth": str(registry.resolve("depth") or ""),
            "smplx": str(registry.resolve("smplx") or ""),
            "relight_sh": str(registry.resolve("relight_sh") or ""),
        }

        artifacts: Dict[str, Path] = {}
        status_map: Dict[str, Any] = {}
        def _dev() -> str:
            try:
                import torch  # type: ignore
                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        def _cuda_stats() -> Dict[str, int]:
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    return {
                        "mem_alloc_mb": int(torch.cuda.memory_allocated() / 1e6),
                        "mem_reserved_mb": int(torch.cuda.memory_reserved() / 1e6),
                    }
            except Exception:
                pass
            return {"mem_alloc_mb": 0, "mem_reserved_mb": 0}
        if trace_dir is not None:
            out_dir = trace_dir / "person"
            # Face detection + landmarks + embedding
            face_det_json = out_dir / "face_detection.json"
            face_lms_json = out_dir / "face_landmarks.json"
            face_emb_json = out_dir / "face_embedding.json"
            face_backend = FaceIDBackend(model_dir=Path(resolved["insightface"]) if resolved["insightface"] else None)
            t0 = time.time()
            face_res = face_backend.compute(image_path)
            t_face = int((time.time() - t0) * 1000)
            if face_res is None:
                self._save_json(face_det_json, {"bbox": None})
                self._save_json(face_lms_json, {"landmarks": []})
                self._save_json(face_emb_json, {"embedding": None})
                log.warning("Face step: fallback")
            else:
                self._save_json(face_det_json, {"bbox": face_res.bbox})
                self._save_json(face_lms_json, {"landmarks": face_res.landmarks or []})
                self._save_json(face_emb_json, {"embedding": face_res.embedding})
                log.info("Face step: ok | det,lms,emb saved")
            artifacts["face_detection"] = face_det_json
            artifacts["face_landmarks"] = face_lms_json
            artifacts["face_embedding"] = face_emb_json
            s = {"ok": bool(face_res), "backend": "insightface", "device": _dev(), "ms": t_face}
            s.update(_cuda_stats())
            status_map["face"] = s
            # Try compute pose
            pose_json = out_dir / "pose.json"
            pose_coco_json = out_dir / "pose_coco.json"
            pose_body25_json = out_dir / "pose_body25.json"
            pose_backend = PoseBackend(model_dir=Path(resolved["openpose"]) if resolved["openpose"] else None)
            t0 = time.time()
            # If we have a face bbox, choose the YOLO person detection closest to its center
            target_xy = None
            try:
                if isinstance(face_res, type(None)):
                    target_xy = None
                else:
                    bb = getattr(face_res, 'bbox', None)
                    if bb and len(bb) == 4:
                        target_xy = [float((bb[0] + bb[2]) * 0.5), float((bb[1] + bb[3]) * 0.5)]
            except Exception:
                target_xy = None
            pose_res = pose_backend.compute(image_path, target_xy=target_xy)
            t_pose = int((time.time() - t0) * 1000)
            if pose_res is None:
                self._save_json(pose_json, {"keypoints": [], "format": "BODY_25", "note": "placeholder"})
                self._save_json(pose_coco_json, {"keypoints": [], "format": "COCO17", "note": "placeholder"})
                self._save_json(pose_body25_json, {"keypoints": [], "format": "BODY_25", "note": "placeholder"})
                log.warning("Pose step: fallback")
            else:
                self._save_json(pose_json, pose_res)
                # Split outputs if available
                try:
                    if isinstance(pose_res, dict):
                        if "yolo" in pose_res and isinstance(pose_res["yolo"], dict):
                            self._save_json(pose_coco_json, pose_res["yolo"]) 
                        if "body_25" in pose_res and isinstance(pose_res["body_25"], dict):
                            self._save_json(pose_body25_json, pose_res["body_25"]) 
                except Exception:
                    pass
                try:
                    n = 0
                    if isinstance(pose_res, dict):
                        if "keypoints" in pose_res:
                            n = len(pose_res.get("keypoints", []))
                        elif "body_25" in pose_res and isinstance(pose_res["body_25"], dict):
                            n = len(pose_res["body_25"].get("keypoints", []))
                        elif "yolo" in pose_res and isinstance(pose_res["yolo"], dict):
                            n = len(pose_res["yolo"].get("keypoints", []))
                    log.info("Pose step: ok | %d keypoints", n)
                except Exception:
                    log.info("Pose step: ok")
            artifacts["pose"] = pose_json
            artifacts["pose_coco"] = pose_coco_json
            artifacts["pose_body25"] = pose_body25_json
            s = {"ok": bool(pose_res), "backend": "yolov8", "device": _dev(), "ms": t_pose}
            s.update(_cuda_stats())
            status_map["pose"] = s

            # Segmentation via SCHP if available
            seg_png = out_dir / "segmentation.png"
            parsing_backend = ParsingBackend(model_dir=Path(resolved["schp"]) if resolved["schp"] else None)
            t0 = time.time()
            seg_path = parsing_backend.compute(image_path)
            t_pars = int((time.time() - t0) * 1000)
            if seg_path and seg_path.exists():
                try:
                    import shutil
                    shutil.copyfile(seg_path, seg_png)
                    log.info("Parsing step: ok | segmentation saved")
                except Exception:
                    log.warning("Parsing step: copy failed; skipping")
            else:
                log.warning("Parsing step: no output")
            artifacts["segmentation"] = seg_png
            # Derive occluder masks from segmentation if possible
            if seg_png.exists():
                self._save_occluder_masks(seg_png, out_dir)
            # Save overlay
            if seg_png.exists():
                self._save_seg_overlay(image_path, seg_png, out_dir / "seg_overlay.png")
                log.info("Parsing step: occluder masks saved")
            s = {"ok": bool(seg_path and seg_path.exists()), "backend": "schp", "device": _dev(), "ms": t_pars}
            s.update(_cuda_stats())
            status_map["parsing"] = s

            # Depth + normals via depth backend if available
            depth_png = out_dir / "depth.png"
            normals_png = out_dir / "normals.png"
            depth_backend = DepthBackend(model_dir=Path(resolved["depth"]) if resolved["depth"] else None)
            t0 = time.time()
            depth_res = depth_backend.compute(image_path)
            t_depth = int((time.time() - t0) * 1000)
            if depth_res is not None:
                try:
                    from PIL import Image
                    import numpy as np
                    Image.fromarray(depth_res["depth"]).save(depth_png)
                    Image.fromarray(depth_res["normals"]).save(normals_png)
                    log.info("Depth step: ok | depth+normals saved")
                except Exception:
                    log.warning("Depth step: write failed; skipping")
            else:
                log.warning("Depth step: no output")
            artifacts["depth"] = depth_png
            artifacts["normals"] = normals_png
            bd = "midas"
            try:
                if isinstance(depth_res, dict) and depth_res.get("_backend"):
                    bd = str(depth_res.get("_backend"))
            except Exception:
                pass
            s = {"ok": bool(depth_res), "backend": bd, "device": _dev(), "ms": t_depth}
            s.update(_cuda_stats())
            status_map["depth"] = s

            # SMPL-X mesh if available
            mesh_obj = out_dir / "body_mesh.obj"
            smplx_backend = SmplxBackend(model_dir=Path(resolved["smplx"]) if resolved["smplx"] else None)
            t0 = time.time()
            smplx_obj = smplx_backend.compute(image_path)
            t_smplx = int((time.time() - t0) * 1000)
            if smplx_obj and smplx_obj.exists():
                try:
                    import shutil
                    shutil.copyfile(smplx_obj, mesh_obj)
                    log.info("SMPL-X step: ok | mesh saved")
                except Exception:
                    log.warning("SMPL-X step: copy failed; skipping")
            else:
                log.warning("SMPL-X step: no mesh")
            artifacts["body_mesh"] = mesh_obj
            s = {"ok": bool(smplx_obj and smplx_obj.exists()), "backend": "smplx", "device": _dev(), "ms": t_smplx}
            s.update(_cuda_stats())
            status_map["smplx"] = s

            # Lighting SH estimation from normals + RGB
            light_json = out_dir / "light_sh.json"
            light_backend = LightingBackend()
            t0 = time.time()
            light_res = light_backend.compute(image_path, normals_png if normals_png.exists() else None, seg_png if seg_png.exists() else None)
            t_light = int((time.time() - t0) * 1000)
            if light_res is not None:
                self._save_json(light_json, light_res)
                log.info("Lighting step: ok | SH saved")
                artifacts["light_sh"] = light_json
                s = {"ok": True, "backend": "sh-relight", "device": "cpu", "ms": t_light}
            else:
                log.warning("Lighting step: no output")
                s = {"ok": False, "backend": "sh-relight", "device": "cpu", "ms": t_light}
            status_map["lighting"] = s

            # Models resolution snapshot
            models_json = out_dir / "models.json"
            status_json = out_dir / "status.json"
            self._save_json(models_json, resolved)
            self._save_json(status_json, status_map)
            log.info("PersonAnalysis complete | status=%s", status_map)
            artifacts["models"] = models_json

        return StageOutput(
            data={
                "pose": None,
                "segmentation": None,
                "depth": None,
                "normals": None,
                "body_mesh": None,
                "light_sh": None,
                "models": resolved,
            },
            artifacts=artifacts,
        )
