from __future__ import annotations

from .base import StageOutput


class HarmonizeUpscale:
    def run(self, refined: StageOutput) -> StageOutput:
        return StageOutput(data={"upscaled": True}, artifacts={})
