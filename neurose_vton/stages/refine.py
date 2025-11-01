from __future__ import annotations

from .base import StageOutput


class PhysicalRefinement:
    def run(self, core: StageOutput) -> StageOutput:
        return StageOutput(data={"refined": True}, artifacts={})
