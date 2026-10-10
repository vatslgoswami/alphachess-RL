"""Load the shared ``config.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the config as a nested dict. ``path`` defaults to the repo's config.yaml."""
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def repo_path(relative: str | Path) -> Path:
    """Resolve a config path (relative to the repo root) to an absolute path."""
    p = Path(relative)
    return p if p.is_absolute() else REPO_ROOT / p
