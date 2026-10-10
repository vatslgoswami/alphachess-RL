"""Successor graph of a three-piece endgame, built once with python-chess.

Nodes ``0 .. n-1`` are the legal canonical positions of the table (see ``index.py``). Moves that
leave the table go to *leaf* nodes ``n .. n+L-1`` whose value is already known:

- a capture that leaves king vs king is a draw;
- a pawn promoting to a knight or bishop is a draw (insufficient material);
- a pawn promoting to a queen or rook enters the solved KQK or KRK table, Black to move, and takes
  that table's value.

Leaves are deduplicated by their ``(wdl, dtm)`` value, so there are only a handful.

Edges are stored as CSR arrays (``succ_ptr``, ``succ``) with the move that produced each edge, in
the canonical coordinates of the parent. ``pred_ptr``/``pred`` hold the reverse edges, with
multiplicity (two moves into the same child give two entries), which the count-down solver needs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import chess
import numpy as np

from .index import TableSpec, canonical, decode, get_spec, legal_mask, make_board

# (table name, slot) -> (wdl, dtm) for the side to move in that table.
ExternalLookup = Callable[[str, int], tuple[int, int]]


@dataclass
class Graph:
    spec: TableSpec
    slots: np.ndarray  # (n,) slot index of each internal node
    node_of_slot: np.ndarray  # (spec.size,) node id or -1
    in_check: np.ndarray  # (n,) bool, side to move is in check
    leaf_wdl: np.ndarray  # (L,) int8
    leaf_dtm: np.ndarray  # (L,) int16, -1 for draws
    succ_ptr: np.ndarray  # (n+1,)
    succ: np.ndarray  # (E,) target node ids (leaves are n + leaf id)
    move_from: np.ndarray  # (E,) int8
    move_to: np.ndarray  # (E,) int8
    move_promo: np.ndarray  # (E,) int8, 0 for none
    pred_ptr: np.ndarray = field(init=False)  # (n+L+1,)
    pred: np.ndarray = field(init=False)  # (E,) source node ids
    build_seconds: float = 0.0

    def __post_init__(self) -> None:
        total = self.n + self.num_leaves
        owner = np.repeat(np.arange(self.n), np.diff(self.succ_ptr))
        order = np.argsort(self.succ, kind="stable")
        self.pred = owner[order]
        counts = np.bincount(self.succ, minlength=total)
        self.pred_ptr = np.concatenate([[0], np.cumsum(counts)])

    @property
    def n(self) -> int:
        return len(self.slots)

    @property
    def num_leaves(self) -> int:
        return len(self.leaf_wdl)

    @property
    def num_edges(self) -> int:
        return len(self.succ)

    @property
    def outdegree(self) -> np.ndarray:
        return np.diff(self.succ_ptr)

    def edge_owner(self) -> np.ndarray:
        return np.repeat(np.arange(self.n), self.outdegree)


def gather(ptr: np.ndarray, data: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Concatenate ``data[ptr[v]:ptr[v+1]]`` for every ``v`` in ``nodes`` (vectorised)."""
    nodes = np.asarray(nodes, dtype=np.int64)
    if nodes.size == 0:
        return data[:0]
    starts = ptr[nodes]
    lens = ptr[nodes + 1] - starts
    total = int(lens.sum())
    if total == 0:
        return data[:0]
    offsets = np.repeat(starts - (np.cumsum(lens) - lens), lens)
    return data[np.arange(total) + offsets]


def build_graph(table: str | TableSpec, external: ExternalLookup | None = None) -> Graph:
    """Enumerate every legal position of ``table`` and its legal moves."""
    spec = get_spec(table)
    started = time.perf_counter()
    mask = legal_mask(spec)
    slots = np.flatnonzero(mask)
    node_of_slot = np.full(spec.size, -1, dtype=np.int64)
    node_of_slot[slots] = np.arange(len(slots))
    stm_a, wk_a, p_a, bk_a = decode(spec, slots)

    leaves: dict[tuple[int, int], int] = {}

    def leaf(wdl: int, dtm: int) -> int:
        return leaves.setdefault((wdl, dtm), len(leaves))

    draw_leaf = leaf(0, -1)

    # Internal successors are collected as raw squares and canonicalised in one batch.
    deg = np.zeros(len(slots), dtype=np.int64)
    in_check = np.zeros(len(slots), dtype=bool)
    e_from, e_to, e_promo = [], [], []
    e_leaf = []  # leaf id, or -1 for an internal successor
    r_stm, r_wk, r_p, r_bk = [], [], [], []

    for node in range(len(slots)):
        stm, wk, p, bk = int(stm_a[node]), int(wk_a[node]), int(p_a[node]), int(bk_a[node])
        board = make_board(spec, stm, wk, p, bk)
        in_check[node] = board.is_check()
        for move in board.legal_moves:
            fr, to, promo = move.from_square, move.to_square, move.promotion or 0
            deg[node] += 1
            e_from.append(fr)
            e_to.append(to)
            e_promo.append(promo)
            nwk, np_, nbk = wk, p, bk
            if stm == 0:
                if fr == wk:
                    nwk = to
                else:
                    np_ = to
                if promo:
                    if promo in (chess.KNIGHT, chess.BISHOP):
                        e_leaf.append(draw_leaf)
                    else:
                        if external is None:
                            raise ValueError("promotion needs solved KQK/KRK tables")
                        name = "KQK" if promo == chess.QUEEN else "KRK"
                        target = get_spec(name)
                        slot, _ = canonical(target, 1, nwk, np_, nbk)
                        e_leaf.append(leaf(*external(name, int(slot))))
                    r_stm.append(0), r_wk.append(0), r_p.append(0), r_bk.append(0)
                    continue
            else:
                nbk = to
                if to == p:  # Black king takes the only white piece: K vs K.
                    e_leaf.append(draw_leaf)
                    r_stm.append(0), r_wk.append(0), r_p.append(0), r_bk.append(0)
                    continue
            e_leaf.append(-1)
            r_stm.append(1 - stm), r_wk.append(nwk), r_p.append(np_), r_bk.append(nbk)

    e_leaf = np.asarray(e_leaf, dtype=np.int64)
    internal = e_leaf < 0
    succ = np.empty(len(e_leaf), dtype=np.int64)
    canon, _ = canonical(
        spec,
        np.asarray(r_stm)[internal],
        np.asarray(r_wk)[internal],
        np.asarray(r_p)[internal],
        np.asarray(r_bk)[internal],
    )
    targets = node_of_slot[canon]
    if np.any(canon < 0) or np.any(targets < 0):
        raise AssertionError(f"{spec.name}: a legal move led to an unindexed position")
    succ[internal] = targets
    succ[~internal] = len(slots) + e_leaf[~internal]

    leaf_items = sorted(leaves.items(), key=lambda kv: kv[1])
    return Graph(
        spec=spec,
        slots=slots,
        node_of_slot=node_of_slot,
        in_check=in_check,
        leaf_wdl=np.array([k[0] for k, _ in leaf_items], dtype=np.int8),
        leaf_dtm=np.array([k[1] for k, _ in leaf_items], dtype=np.int16),
        succ_ptr=np.concatenate([[0], np.cumsum(deg)]),
        succ=succ,
        move_from=np.asarray(e_from, dtype=np.int8),
        move_to=np.asarray(e_to, dtype=np.int8),
        move_promo=np.asarray(e_promo, dtype=np.int8),
        build_seconds=time.perf_counter() - started,
    )
