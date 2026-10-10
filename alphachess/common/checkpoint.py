"""Checkpoint save/load. Format: docs/INTERFACES.md §6."""

from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path
from typing import Any


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save_checkpoint(path: str | Path, model, optimizer, step: int, config: dict, notes: str = ""):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "config": config,
            "git_commit": git_commit(),
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "notes": notes,
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    import torch

    return torch.load(path, map_location=map_location, weights_only=False)
