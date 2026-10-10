"""Independent checks of a solved table.

1. :func:`check_syzygy` compares WDL with Syzygy on every legal position.
2. :func:`check_stockfish_dtm` spot-checks DTM with a Stockfish mate search on random positions.
3. :func:`check_bellman` checks that every position satisfies the negamax Bellman equation on our
   own graph, and that the stored best move leads to a successor exactly one ply closer to mate.
4. :func:`check_value_iteration` runs plain synchronous value iteration (a second, independent
   algorithm) to its fixed point and compares WDL and DTM exactly.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass

import chess
import chess.engine
import chess.syzygy
import numpy as np

from .graph import Graph
from .index import index_to_board
from .solve import LOSS, WIN, edge_scores
from .tables import Table
from .vi import value_iteration, values_to_wdl_dtm


@dataclass
class CheckResult:
    name: str
    checked: int
    agree: int
    details: dict

    @property
    def ok(self) -> bool:
        return self.checked > 0 and self.agree == self.checked

    def as_dict(self) -> dict:
        return {**asdict(self), "ok": self.ok}


def check_syzygy(table: Table, tablebase_dir: str) -> CheckResult:
    """WDL agreement with Syzygy on every legal position (Syzygy ±1 = 50-move draws → 0)."""
    mismatches = []
    slots = np.flatnonzero(table.legal)
    with chess.syzygy.open_tablebase(tablebase_dir) as tb:
        for slot in slots:
            board = index_to_board(int(slot), table.spec)
            raw = tb.probe_wdl(board)
            theirs = 1 if raw == 2 else -1 if raw == -2 else 0
            if theirs != int(table.wdl[slot]):
                mismatches.append(board.fen())
    agree = len(slots) - len(mismatches)
    return CheckResult("syzygy_wdl", len(slots), agree, {"mismatches": mismatches[:20]})


def stockfish_path(configured: str = "stockfish") -> str | None:
    for candidate in (configured, "stockfish", "/usr/games/stockfish"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def check_stockfish_dtm(
    table: Table,
    samples: int = 1000,
    seed: int = 0,
    engine_path: str | None = None,
    nodes: int = 300_000,
) -> CheckResult:
    """Stockfish mate distance on random decisive positions vs our DTM.

    Stockfish has no tablebases here, so it may miss a long mate within its node budget
    (``not_found``) or report a longer one (``longer``); neither contradicts the table. A
    *shorter* mate, or a mate in a position we call a draw, would be a real error.
    """
    path = engine_path or stockfish_path()
    if path is None:
        raise FileNotFoundError("stockfish binary not found")
    rng = np.random.default_rng(seed)
    slots = rng.choice(np.flatnonzero(table.legal), size=samples, replace=False)
    counts = {"exact": 0, "longer": 0, "not_found": 0, "shorter": 0, "draw_ok": 0, "draw_bad": 0}
    errors = []
    with chess.engine.SimpleEngine.popen_uci(path) as engine:
        for slot in slots:
            board = index_to_board(int(slot), table.spec)
            if board.is_game_over():
                counts[
                    "exact" if board.is_checkmate() == (table.wdl[slot] == LOSS) else "shorter"
                ] += 1
                continue
            wdl, dtm = table.lookup(int(slot))
            # A fresh ``game`` object clears the hash between positions.
            info = engine.analyse(board, chess.engine.Limit(nodes=nodes), game=object())
            mate = info["score"].relative.mate()
            if wdl == 0:
                key = "draw_ok" if mate is None else "draw_bad"
            elif mate is None:
                key = "not_found"
            else:
                theirs = 2 * mate - 1 if mate > 0 else -2 * mate
                key = "exact" if theirs == dtm else "longer" if theirs > dtm else "shorter"
                if np.sign(mate) != wdl:
                    key = "shorter"
            counts[key] += 1
            if key in ("shorter", "draw_bad"):
                errors.append(board.fen())
    agree = samples - counts["shorter"] - counts["draw_bad"]
    return CheckResult("stockfish_dtm", samples, agree, {**counts, "errors": errors[:20]})


def check_bellman(table: Table, graph: Graph) -> CheckResult:
    """Every node equals the negamax backup of its successors; best move is DTM-consistent."""
    n = graph.n
    wdl = np.concatenate([table.wdl[graph.slots], graph.leaf_wdl])
    dtm = np.concatenate([table.dtm[graph.slots], graph.leaf_dtm])
    scores = edge_scores(graph, wdl, dtm)
    has_moves = graph.outdegree > 0
    best = np.full(n, np.iinfo(np.int64).min)
    best[has_moves] = np.maximum.reduceat(scores, graph.succ_ptr[:-1][has_moves])
    # Expected (wdl, dtm) from the best backed-up score.
    exp_wdl = np.where(best > 5_000, WIN, np.where(best < -5_000, LOSS, 0))
    exp_dtm = np.where(exp_wdl == WIN, 10_001 - best, np.where(exp_wdl == LOSS, 10_001 + best, -1))
    exp_wdl[~has_moves] = np.where(graph.in_check[~has_moves], LOSS, 0)
    exp_dtm[~has_moves] = np.where(graph.in_check[~has_moves], 0, -1)
    node_ok = (exp_wdl == wdl[:n]) & (exp_dtm == dtm[:n])

    # The stored best move must be a legal edge whose successor is one ply closer to mate.
    owner = graph.edge_owner()
    slots = graph.slots[owner]
    match = (
        (graph.move_from == table.best_from[slots])
        & (graph.move_to == table.best_to[slots])
        & (graph.move_promo == table.best_promo[slots])
    )
    best_edge = np.full(n, -1)
    best_edge[owner[match]] = np.flatnonzero(match)
    child = graph.succ[best_edge[has_moves]]
    move_ok = np.ones(n, dtype=bool)
    nodes = np.flatnonzero(has_moves)
    decisive = wdl[nodes] != 0
    move_ok[nodes] = np.where(
        decisive,
        (wdl[child] == -wdl[nodes]) & (dtm[child] == dtm[nodes] - 1),
        wdl[child] == 0,
    )
    move_ok[nodes[best_edge[nodes] < 0]] = False
    ok = node_ok & move_ok
    return CheckResult(
        "bellman",
        n,
        int(ok.sum()),
        {"backup_errors": int((~node_ok).sum()), "best_move_errors": int((~move_ok).sum())},
    )


def check_value_iteration(table: Table, graph: Graph, gamma: float = 0.9) -> CheckResult:
    """Plain synchronous value iteration must reproduce WDL and DTM exactly."""
    result = value_iteration(graph, gamma)
    wdl, dtm = values_to_wdl_dtm(result.values, gamma)
    ok = (wdl == table.wdl[graph.slots]) & (dtm == table.dtm[graph.slots])
    return CheckResult(
        "value_iteration",
        graph.n,
        int(ok.sum()),
        {"sweeps": result.sweeps, "backups": result.backups, "seconds": round(result.seconds, 3)},
    )
