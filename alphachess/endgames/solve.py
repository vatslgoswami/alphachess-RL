"""Retrograde analysis: the count-down BFS from the primer (docs/Vatsl_Endgame_DP_Primer.html).

Values are from the side to move's perspective: +1 win, 0 draw, -1 loss, with DTM in plies
(winner mates fastest, loser resists longest). This solves the negamax Bellman equation
``V(s) = max_a -V(next)`` exactly, but backs up a position only when one of its successors has
just been resolved, instead of sweeping every position like value iteration (``vi.py``).

Level ``d`` holds the positions resolved with DTM ``d``. From each one we walk predecessors:

- a predecessor of a *lost* position is a win with DTM ``d + 1`` (first time wins: shortest);
- a predecessor of a *won* position loses one escape; when its counter of unresolved moves hits
  zero, every move leads to an opponent win, so it is lost with DTM ``d + 1`` (last child
  processed is the longest resistance).

Whatever is never resolved is a draw.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .graph import Graph, gather

WIN, DRAW, LOSS = 1, 0, -1


@dataclass
class Solution:
    wdl: np.ndarray  # (n + L,) int8
    dtm: np.ndarray  # (n + L,) int16, -1 for draws
    best_edge: np.ndarray  # (n,) edge id of the optimal move, -1 for terminal positions
    seconds: float
    levels: int  # number of DTM levels processed


def solve(graph: Graph) -> Solution:
    started = time.perf_counter()
    n, total = graph.n, graph.n + graph.num_leaves
    wdl = np.zeros(total, dtype=np.int8)
    dtm = np.full(total, -1, dtype=np.int16)
    resolved = np.zeros(total, dtype=bool)
    count = np.zeros(total, dtype=np.int64)
    count[:n] = graph.outdegree

    buckets: dict[int, list[np.ndarray]] = {}

    def push(d: int, nodes: np.ndarray) -> None:
        if nodes.size:
            buckets.setdefault(d, []).append(nodes)

    # Seeds: checkmates (loss, DTM 0), stalemates (draw), and leaves with known values.
    no_moves = np.flatnonzero(graph.outdegree == 0)
    mates = no_moves[graph.in_check[no_moves]]
    wdl[mates], dtm[mates], resolved[no_moves] = LOSS, 0, True
    push(0, mates)
    leaf_ids = n + np.arange(graph.num_leaves)
    wdl[leaf_ids], dtm[leaf_ids], resolved[leaf_ids] = graph.leaf_wdl, graph.leaf_dtm, True
    for leaf in leaf_ids[graph.leaf_wdl != DRAW]:
        push(int(dtm[leaf]), np.array([leaf]))

    d = 0
    while buckets:
        frontier = np.concatenate(buckets.pop(d, [np.empty(0, dtype=np.int64)]))
        lost = frontier[wdl[frontier] == LOSS]
        won = frontier[wdl[frontier] == WIN]

        preds = np.unique(gather(graph.pred_ptr, graph.pred, lost))
        preds = preds[~resolved[preds]]
        wdl[preds], dtm[preds], resolved[preds] = WIN, d + 1, True
        push(d + 1, preds)

        preds = gather(graph.pred_ptr, graph.pred, won)
        np.subtract.at(count, preds, 1)
        preds = np.unique(preds)
        preds = preds[(count[preds] == 0) & ~resolved[preds]]
        wdl[preds], dtm[preds], resolved[preds] = LOSS, d + 1, True
        push(d + 1, preds)
        d += 1

    best_edge = best_edges(graph, wdl, dtm)
    return Solution(wdl, dtm, best_edge, time.perf_counter() - started, d)


def edge_scores(graph: Graph, wdl: np.ndarray, dtm: np.ndarray) -> np.ndarray:
    """Score of every move from the mover's side: fast wins high, slow losses above fast ones."""
    child_wdl = wdl[graph.succ].astype(np.int64)
    child_dtm = dtm[graph.succ].astype(np.int64)
    return np.where(
        child_wdl == LOSS, 10_000 - child_dtm, np.where(child_wdl == WIN, -10_000 + child_dtm, 0)
    )


def best_edges(graph: Graph, wdl: np.ndarray, dtm: np.ndarray) -> np.ndarray:
    """Edge id of the optimal move per node (first of equally good moves), -1 if none."""
    scores = edge_scores(graph, wdl, dtm)
    owner = graph.edge_owner()
    # Sort by (owner, -score, edge id) and keep the first edge of each owner.
    order = np.lexsort((np.arange(len(scores)), -scores, owner))
    first = np.ones(len(order), dtype=bool)
    first[1:] = owner[order][1:] != owner[order][:-1]
    best = np.full(graph.n, -1, dtype=np.int64)
    best[owner[order][first]] = order[first]
    return best


def gamma_values(wdl: np.ndarray, dtm: np.ndarray, gamma: float) -> np.ndarray:
    """Discounted value ``±gamma**dtm`` (0 for draws), as in the primer."""
    power = np.power(gamma, np.maximum(dtm, 0).astype(np.float64))
    return np.where(wdl == WIN, power, np.where(wdl == LOSS, -power, 0.0))
