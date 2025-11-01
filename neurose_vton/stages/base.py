from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class StageOutput:
    data: Dict[str, Any]
    artifacts: Dict[str, Path]

