from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    repo_root: Path = REPO_ROOT
    storage: Path = REPO_ROOT / "storage"
    manual: Path = REPO_ROOT / "manual_downloads"
    third_party: Path = REPO_ROOT / "third_party"
    runtime_cache: Path = REPO_ROOT / "runtime_cache"
    outputs: Path = REPO_ROOT / "outputs"

    def ensure_runtime_dirs(self) -> None:
        self.runtime_cache.mkdir(parents=True, exist_ok=True)
        self.outputs.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    seed: int
    strict_determinism: bool
    log_level: str

    @staticmethod
    def load(env: dict[str, str] | None = None) -> "Settings":
        env = env or os.environ
        seed = int(env.get("NEUROSE_DEFAULT_SEED", "12345"))
        strict = env.get("NEUROSE_STRICT_DETERMINISM", "1") in {"1", "true", "True"}
        log_level = env.get("NEUROSE_LOG_LEVEL", "INFO")
        return Settings(seed=seed, strict_determinism=strict, log_level=log_level)


PATHS = Paths()
SETTINGS = Settings.load()
PATHS.ensure_runtime_dirs()
