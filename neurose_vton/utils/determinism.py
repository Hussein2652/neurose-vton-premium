from __future__ import annotations

import os
import random
from typing import Optional


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def enable_determinism(strict: bool = True) -> dict[str, str]:
    """Best-effort deterministic settings. Returns env changes to apply for CUDA if needed."""
    env_changes: dict[str, str] = {}
    try:
        import torch  # type: ignore

        if strict and hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=not strict)
        # cuDNN
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        # CUDA kernels
        env_changes["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        for k, v in env_changes.items():
            os.environ[k] = v
    except Exception:
        pass
    return env_changes
