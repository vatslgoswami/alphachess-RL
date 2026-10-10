"""Types shared across components. See docs/INTERFACES.md §5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sample:
    """One training example: encoded position, policy target, value target."""

    planes: np.ndarray  # (C, 8, 8)
    pi_idx: np.ndarray  # (k,) int16 move indices
    pi_p: np.ndarray  # (k,) float32 probabilities, sum to 1
    z: float  # value target in [-1, 1], side to move's perspective
