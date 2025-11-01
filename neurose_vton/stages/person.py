from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import json

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
            seg = Image.open(seg_path)
            arr = np.array(seg)
            # LIP label indices per SCHP (see dataset_settings in SCHP):
            # Hair=2, Left-arm=14, Right-arm=15, Glove=3 approximates hands
            hair_mask = (arr == 2).astype(np.uint8) * 255
            arms_mask = ((arr == 14) | (arr == 15)).astype(np.uint8) * 255
            hands_mask = (arr == 3).astype(np.uint8) * 255
            Image.fromarray(hair_mask).save(hair)
            Image.fromarray(arms_mask).save(arms)
            Image.fromarray(hands_mask).save(hands)
        except Exception:
            # Create placeholders if segmentation not loadable
            self._save_blank_png(hair, 64, 64, 1)
            self._save_blank_png(arms, 64, 64, 1)
            self._save_blank_png(hands, 64, 64, 1)

    def run(self, image_path: Path, seed: int, trace_dir: Optional[Path] = None) -> StageOutput:
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
        if trace_dir is not None:
            out_dir = trace_dir / "person"
            # Face detection + landmarks + embedding
            face_det_json = out_dir / "face_detection.json"
            face_lms_json = out_dir / "face_landmarks.json"
            face_emb_json = out_dir / "face_embedding.json"
            face_backend = FaceIDBackend(model_dir=Path(resolved["insightface"]) if resolved["insightface"] else None)
            face_res = face_backend.compute(image_path)
            if face_res is None:
                self._save_json(face_det_json, {"bbox": None})
                self._save_json(face_lms_json, {"landmarks": []})
                self._save_json(face_emb_json, {"embedding": None})
            else:
                self._save_json(face_det_json, {"bbox": face_res.bbox})
                self._save_json(face_lms_json, {"landmarks": face_res.landmarks or []})
                self._save_json(face_emb_json, {"embedding": face_res.embedding})
            artifacts["face_detection"] = face_det_json
            artifacts["face_landmarks"] = face_lms_json
            artifacts["face_embedding"] = face_emb_json
            # Try compute pose
            pose_json = out_dir / "pose.json"
            pose_backend = PoseBackend(model_dir=Path(resolved["openpose"]) if resolved["openpose"] else None)
            pose_res = pose_backend.compute(image_path)
            if pose_res is None:
                self._save_json(pose_json, {"keypoints": [], "format": "BODY_25", "note": "placeholder"})
            else:
                self._save_json(pose_json, pose_res)
            artifacts["pose"] = pose_json

            # Segmentation via SCHP if available
            seg_png = out_dir / "segmentation.png"
            parsing_backend = ParsingBackend(model_dir=Path(resolved["schp"]) if resolved["schp"] else None)
            seg_path = parsing_backend.compute(image_path)
            if seg_path and seg_path.exists():
                try:
                    import shutil
                    shutil.copyfile(seg_path, seg_png)
                except Exception:
                    self._save_blank_png(seg_png, 64, 64, 1)
            else:
                self._save_blank_png(seg_png, 64, 64, 1)
            artifacts["segmentation"] = seg_png
            # Derive occluder masks from segmentation if possible
            self._save_occluder_masks(seg_png, out_dir)

            # Depth + normals via depth backend if available
            depth_png = out_dir / "depth.png"
            normals_png = out_dir / "normals.png"
            depth_backend = DepthBackend(model_dir=None)
            depth_res = depth_backend.compute(image_path)
            if depth_res is not None:
                try:
                    from PIL import Image
                    import numpy as np
                    Image.fromarray(depth_res["depth"]).save(depth_png)
                    Image.fromarray(depth_res["normals"]).save(normals_png)
                except Exception:
                    self._save_blank_png(depth_png, 64, 64, 1)
                    self._save_blank_png(normals_png, 64, 64, 3)
            else:
                self._save_blank_png(depth_png, 64, 64, 1)
                self._save_blank_png(normals_png, 64, 64, 3)
            artifacts["depth"] = depth_png
            artifacts["normals"] = normals_png

            # SMPL-X mesh if available
            mesh_obj = out_dir / "body_mesh.obj"
            smplx_backend = SmplxBackend(model_dir=Path(resolved["smplx"]) if resolved["smplx"] else None)
            smplx_obj = smplx_backend.compute(image_path)
            if smplx_obj and smplx_obj.exists():
                try:
                    import shutil
                    shutil.copyfile(smplx_obj, mesh_obj)
                except Exception:
                    self._save_placeholder_mesh(mesh_obj)
            else:
                self._save_placeholder_mesh(mesh_obj)
            artifacts["body_mesh"] = mesh_obj

            # Lighting SH attempt
            light_json = out_dir / "light_sh.json"
            light_backend = LightingBackend()
            light_res = light_backend.compute(image_path) or {"sh9": [0.0] * 9}
            self._save_json(light_json, light_res)
            artifacts["light_sh"] = light_json

            # Models resolution snapshot
            models_json = out_dir / "models.json"
            self._save_json(models_json, resolved)
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
