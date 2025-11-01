from __future__ import annotations

from .base import StageOutput


class DiffusionCore:
    def __init__(self, steps: int) -> None:
        self.steps = steps

    def run(self, plan: StageOutput, seed: int) -> StageOutput:
        return StageOutput(
            data={
                "steps": self.steps,
                "scheduler": "dpmpp_2m_karras",
                "result_image": None,
            },
            artifacts={},
        )
