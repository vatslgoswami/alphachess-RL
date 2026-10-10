"""Synchronous value iteration on γ-values: ``V(s) = max_a -γ V(next)``, starting from 0.

Every sweep backs up every position from the previous sweep's values (the primer's section 3).
Checkmates are fixed at -1, stalemates at 0, and leaf nodes at their known ``±γ^dtm``. With
γ < 1 the fixed point is ``±γ^dtm`` for wins/losses and 0 for draws, so it carries the same
information as the retrograde table. :func:`values_to_wdl_dtm` converts it back for comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .graph import Graph
from .solve import gamma_values


@dataclass
class VIResult:
    values: np.ndarray  # (n,) γ-values of internal nodes
    sweeps: int  # sweeps until a sweep changed nothing (that last sweep included)
    backups: int  # sweeps * positions
    seconds: float
    changed_per_sweep: list[int]


def value_iteration(graph: Graph, gamma: float = 0.9, max_sweeps: int = 10_000) -> VIResult:
    if not 0 < gamma < 1:
        raise ValueError("value iteration here needs 0 < gamma < 1 so that DTM is recoverable")
    started = time.perf_counter()
    n = graph.n
    leaf_values = gamma_values(graph.leaf_wdl, graph.leaf_dtm, gamma)
    values = np.zeros(n + graph.num_leaves)
    values[n:] = leaf_values
    deg = graph.outdegree
    has_moves = deg > 0
    mates = (~has_moves) & graph.in_check
    values[:n][mates] = -1.0
    starts = graph.succ_ptr[:-1][has_moves]
    changed_per_sweep = []
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        backed = -gamma * values[graph.succ]
        new = values[:n].copy()
        new[has_moves] = np.maximum.reduceat(backed, starts)
        changed = int(np.count_nonzero(new != values[:n]))
        changed_per_sweep.append(changed)
        values[:n] = new
        if changed == 0:
            break
    return VIResult(
        values[:n].copy(), sweeps, sweeps * n, time.perf_counter() - started, changed_per_sweep
    )


def values_to_wdl_dtm(values: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Recover (wdl, dtm) from ``±γ^dtm`` values (0 means draw)."""
    wdl = np.sign(values).astype(np.int8)
    with np.errstate(divide="ignore"):
        dtm = np.where(wdl != 0, np.rint(np.log(np.abs(values)) / np.log(gamma)), -1)
    return wdl, dtm.astype(np.int16)
