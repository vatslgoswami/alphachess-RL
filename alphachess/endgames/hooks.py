"""Hooks that plug the exact tables into Agriya's search and evaluation harness (T7).

- :class:`DPEvaluator` is an MCTS ``Evaluator``: for covered positions it puts all prior mass on
  the table's best move and returns the exact value (WDL, or ``±γ^dtm`` if ``gamma`` is set).
- :class:`TablebasePlayer` is a ``Player`` that plays the table move when covered and otherwise
  delegates to a fallback player. Phase 2 uses the same idea to play 5-piece endgames exactly.
- :func:`exact_reference_analyser` is a ``ReferenceAnalyser`` for ``evaluate_positions``.
- :func:`build_suite` / :func:`load_suite` make and read the frozen endgame test suite.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import chess
import numpy as np

from alphachess.common.config import load_config, repo_path
from alphachess.eval.player import MoveDecision, Player
from alphachess.mcts import Evaluator, uniform_evaluator

from . import probe
from .index import SYM, index_to_board
from .solve import LOSS, WIN
from .tables import load_or_solve

WIN_SCORE = 10_000


class DPEvaluator:
    """Exact evaluator for KQK/KRK/KPK positions; other positions go to ``fallback``."""

    def __init__(self, gamma: float | None = None, fallback: Evaluator = uniform_evaluator):
        self.gamma = gamma
        self.fallback = fallback

    def value(self, board: chess.Board) -> float | None:
        wdl = probe.probe_wdl(board)
        if wdl is None:
            return None
        if self.gamma is None or wdl == 0:
            return float(wdl)
        dtm = probe.probe_dtm(board)
        return float(wdl) * self.gamma ** abs(dtm) if dtm is not None else float(wdl)

    def __call__(
        self, boards: list[chess.Board]
    ) -> tuple[list[dict[chess.Move, float]], np.ndarray]:
        priors: list[dict[chess.Move, float] | None] = [None] * len(boards)
        values = np.zeros(len(boards), dtype=np.float32)
        missing = []
        for i, board in enumerate(boards):
            value = self.value(board)
            move = probe.best_move(board) if value is not None else None
            if value is None or (move is None and not board.is_game_over()):
                missing.append(i)
                continue
            priors[i] = {move: 1.0} if move is not None else {}
            values[i] = value
        if missing:
            fb_priors, fb_values = self.fallback([boards[i] for i in missing])
            for j, i in enumerate(missing):
                priors[i] = fb_priors[j]
                values[i] = fb_values[j]
        return priors, values


class TablebasePlayer:
    """Plays the exact table move when the position is covered, else asks ``fallback``."""

    def __init__(self, fallback: Player, name: str | None = None) -> None:
        self.fallback = fallback
        self.name = name or f"tablebase+{fallback.name}"

    def select_move(self, board: chess.Board) -> MoveDecision:
        move = probe.best_move(board)
        if move is None:
            return self.fallback.select_move(board)
        return MoveDecision(move=move, value=float(probe.probe_wdl(board) or 0))


def move_scores(board: chess.Board) -> dict[chess.Move, int]:
    """Exact score of every legal move for the side to move.

    Win = ``10000 - plies to mate``, draw = 0, loss = ``-10000 + plies to mate``, with plies
    counted from ``board`` (so a mating move scores ``10000 - 1``).
    """
    scores = {}
    for move in board.legal_moves:
        board.push(move)
        try:
            wdl = probe.probe_wdl(board)
            dtm = probe.probe_dtm(board)
        finally:
            board.pop()
        if wdl is None or dtm is None:
            raise ValueError(f"position not covered by the exact tables: {board.fen()}")
        plies = abs(dtm) + 1
        scores[move] = WIN_SCORE - plies if wdl == -1 else -WIN_SCORE + plies if wdl == 1 else 0
    return scores


def exact_reference_analyser(board: chess.Board) -> tuple[chess.Move, dict[chess.Move, int]]:
    """``(best_move, {move: score})`` for ``alphachess.eval.metrics.evaluate_positions``."""
    scores = move_scores(board)
    if not scores:
        raise ValueError("no legal moves")
    best = max(scores.values())
    table_move = probe.best_move(board)
    if table_move in scores and scores[table_move] == best:
        return table_move, scores
    return max(scores, key=lambda m: (scores[m], m.uci())), scores


# --- frozen test suite --------------------------------------------------------------------------


def default_suite_path() -> Path:
    return repo_path(load_config()["paths"]["endgame_dir"]) / "suite.fen"


def _randomise(board: chess.Board, rng: np.random.Generator, pawn: bool) -> chess.Board:
    """Apply a random board symmetry (and colour flip), so the suite isn't all canonical."""
    syms = [0, 1] if pawn else list(range(8))
    k = int(rng.choice(syms))
    out = chess.Board(None)
    for sq, piece in board.piece_map().items():
        out.set_piece_at(int(SYM[k][sq]), piece)
    out.turn = board.turn
    return out.mirror() if rng.random() < 0.5 else out


def build_suite(per_table: int = 100, seed: int = 0) -> dict[str, list[str]]:
    """``per_table`` positions per table, stratified by DTM, both sides to move, no terminals.

    Decisive positions are spread evenly over DTM quantile bins; KPK also gets 20% draws.
    """
    rng = np.random.default_rng(seed)
    suite = {}
    for name in ("KQK", "KRK", "KPK"):
        table = load_or_solve(name)
        slots = np.flatnonzero(table.legal)
        boards = []
        quota = {"decisive": per_table - (per_table // 5 if name == "KPK" else 0)}
        quota["draw"] = per_table - quota["decisive"]
        decisive = slots[(table.wdl[slots] == WIN) | (table.wdl[slots] == LOSS)]
        decisive = decisive[table.dtm[decisive] > 0]
        bins = np.array_split(
            decisive[np.argsort(table.dtm[decisive], kind="stable")], quota["decisive"]
        )
        picks = [int(rng.choice(b)) for b in bins if len(b)]
        draws = slots[table.wdl[slots] == 0]
        draws = np.array(
            [
                s
                for s in rng.permutation(draws)[: 5 * quota["draw"] + 50]
                if not index_to_board(int(s), name).is_game_over()
            ]
        )
        picks += [int(s) for s in draws[: quota["draw"]]]
        for slot in picks:
            board = _randomise(index_to_board(slot, name), rng, name == "KPK")
            boards.append(board.fen())
        suite[name] = boards
    return suite


def write_suite(suite: dict[str, list[str]], path: str | Path | None = None) -> Path:
    path = Path(path) if path is not None else default_suite_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Frozen endgame test suite: KQK, KRK, KPK, stratified by depth to mate.",
        "# Generated by scripts/run_endgame_mcts.py --rebuild-suite (seed 0). One FEN per line.",
    ]
    for name, fens in suite.items():
        lines += ["", f"# {name}", *fens]
    path.write_text("\n".join(lines) + "\n")
    return path


def load_suite(path: str | Path | None = None) -> dict[str, list[chess.Board]]:
    """Read the suite, grouped by the ``# KQK`` / ``# KRK`` / ``# KPK`` section headers."""
    path = Path(path) if path is not None else default_suite_path()
    groups: dict[str, list[chess.Board]] = {}
    current = None
    for line in path.read_text().splitlines():
        value = line.strip()
        if value.startswith("# K") and len(value.split()) == 2:
            current = value[2:]
            groups[current] = []
        elif value and not value.startswith("#") and current is not None:
            groups[current].append(chess.Board(value))
    return groups


class RecordingPlayer:
    """Wraps a player and records its decisions (to score them a second way)."""

    def __init__(self, player: Player) -> None:
        self.player = player
        self.name = player.name
        self.moves: list[tuple[str, chess.Move | None]] = []

    def select_move(self, board: chess.Board) -> MoveDecision:
        decision = self.player.select_move(board)
        self.moves.append((board.fen(), decision.move))
        return decision


def optimal_rates(
    moves: list[tuple[str, chess.Move | None]], analyse: Callable = exact_reference_analyser
) -> dict[str, float]:
    """Share of moves that are DTM-optimal (any of the tied best moves) and result-keeping."""
    optimal = keeps = 0
    for fen, move in moves:
        _, scores = analyse(chess.Board(fen))
        best = max(scores.values())
        chosen = scores.get(move, min(scores.values()))
        optimal += chosen == best
        keeps += np.sign(chosen) == np.sign(best)
    n = max(len(moves), 1)
    return {"optimal_move_rate": optimal / n, "keeps_result_rate": keeps / n}
