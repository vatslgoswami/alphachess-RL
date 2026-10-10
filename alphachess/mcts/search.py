"""MCTS. Spec: docs/INTERFACES.md §8."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import chess
import numpy as np

# evaluator(boards) -> (priors per board as {move: p}, values array, side to move's perspective)
Evaluator = Callable[[list[chess.Board]], tuple[list[dict[chess.Move, float]], np.ndarray]]


@dataclass
class SearchResult:
    visits: dict[chess.Move, int] = field(default_factory=dict)
    q: float = 0.0
    best_move: chess.Move | None = None


def uniform_evaluator(boards: list[chess.Board]):
    """Uniform priors and zero value. Lets MCTS be tested before the network exists."""
    priors = []
    for b in boards:
        moves = list(b.legal_moves)
        priors.append({m: 1.0 / len(moves) for m in moves} if moves else {})
    return priors, np.zeros(len(boards), dtype=np.float32)


def search(
    board: chess.Board,
    evaluator: Evaluator,
    num_simulations: int,
    add_noise: bool = False,
) -> SearchResult:
    raise NotImplementedError
