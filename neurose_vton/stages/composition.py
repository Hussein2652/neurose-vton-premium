from __future__ import annotations

from .base import StageOutput


class CompositionPlanning:
    def __init__(self) -> None:
        pass

    def run(self, person: StageOutput, garment: StageOutput, fit: str = "true") -> StageOutput:
        return StageOutput(
            data={
                "occlusion_order": ["hair", "face", "hands", "garment", "torso"],
                "fit": fit,
                "constraints": {
                    "pose": person.data.get("pose"),
                    "seg": person.data.get("segmentation"),
                    "depth": person.data.get("depth"),
                    "warped_latent": None,
                    "face_id": None,
                    "uv": garment.data.get("uv_atlas"),
                },
            },
            artifacts={},
        )
