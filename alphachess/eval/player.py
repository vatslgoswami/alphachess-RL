"""Common player interface used by UCI and the evaluation arena."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

import chess

from alphachess.mcts import Evaluator, gumbel_search, search


@dataclass(frozen=True)
class MoveDecision:
    move: chess.Move | None
    value: float = 0.0
    simulations: int = 0
    elapsed_seconds: float = 0.0


class Player(Protocol):
    name: str

    def select_move(self, board: chess.Board) -> MoveDecision: ...


class PolicyPlayer:
    """Zero-search baseline that selects the largest legal evaluator prior."""

    def __init__(self, evaluator: Evaluator, name: str = "policy") -> None:
        self.evaluator = evaluator
        self.name = name

    def select_move(self, board: chess.Board) -> MoveDecision:
        if board.is_game_over(claim_draw=True):
            return MoveDecision(None)
        priors_batch, values = self.evaluator([board.copy(stack=True)])
        legal = set(board.legal_moves)
        priors = {move: float(value) for move, value in priors_batch[0].items() if move in legal}
        if not priors:
            move = max(legal, key=lambda candidate: candidate.uci())
        else:
            move = max(priors, key=lambda candidate: (priors[candidate], candidate.uci()))
        return MoveDecision(move=move, value=float(values[0]))


class MCTSPlayer:
    def __init__(
        self,
        evaluator: Evaluator,
        simulations: int,
        *,
        c_puct: float = 1.5,
        batch_size: int = 1,
        seed: int | None = None,
        use_gumbel: bool = False,
        name: str | None = None,
    ) -> None:
        if simulations < 1:
            raise ValueError("MCTSPlayer requires at least one simulation")
        self.evaluator = evaluator
        self.simulations = simulations
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.seed = seed
        self.use_gumbel = use_gumbel
        self.name = name or ("gumbel" if use_gumbel else "puct") + f"-{simulations}"

    def select_move(self, board: chess.Board) -> MoveDecision:
        search_fn = gumbel_search if self.use_gumbel else search
        result = search_fn(
            board,
            self.evaluator,
            self.simulations,
            c_puct=self.c_puct,
            batch_size=self.batch_size,
            seed=self.seed,
        )
        return MoveDecision(
            move=result.best_move,
            value=result.q,
            simulations=result.diagnostics.simulations,
            elapsed_seconds=result.diagnostics.elapsed_seconds,
        )


class RandomPlayer:
    """Seeded legal-move baseline for smoke tests."""

    def __init__(self, seed: int = 0, name: str = "random") -> None:
        self.seed = seed
        self.name = name

    def select_move(self, board: chess.Board) -> MoveDecision:
        moves = list(board.legal_moves)
        rng = random.Random(f"{self.seed}:{board.fen()}")
        return MoveDecision(rng.choice(moves) if moves else None)
