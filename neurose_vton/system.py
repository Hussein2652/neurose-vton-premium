from __future__ import annotations

from typing import Any, Dict


def system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"torch": False, "cuda": False, "cuda_device_count": 0}
    try:
        import torch  # type: ignore

        info["torch"] = True
        if hasattr(torch, "cuda"):
            info["cuda"] = bool(torch.cuda.is_available())
            try:
                info["cuda_device_count"] = int(torch.cuda.device_count())
            except Exception:
                info["cuda_device_count"] = 0
    except Exception:
        pass
    return info

