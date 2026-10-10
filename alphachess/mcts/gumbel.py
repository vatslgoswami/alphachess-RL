"""A low-budget Gumbel-root variant of PUCT.

This keeps the regular tree policy and perturbs only the root action logits.
It is intentionally smaller than the full Gumbel AlphaZero sequential-halving
algorithm, while providing a reproducible low-simulation exploration variant.
"""

from __future__ import annotations

import math

import chess
import numpy as np

from .search import Evaluator, SearchResult, search


def gumbel_search(
    board: chess.Board,
    evaluator: Evaluator,
    num_simulations: int,
    *,
    seed: int | None = None,
    scale: float = 1.0,
    **search_kwargs,
) -> SearchResult:
    """Run PUCT after adding seeded Gumbel noise to root log-priors."""
    if scale < 0:
        raise ValueError("scale must be non-negative")
    root_fen = board.fen()
    rng = np.random.default_rng(seed)
    root_adjusted = False

    def wrapped(boards: list[chess.Board]):
        nonlocal root_adjusted
        priors_batch, values = evaluator(boards)
        adjusted_batch = []
        for candidate, priors in zip(boards, priors_batch, strict=True):
            if not root_adjusted and candidate.fen() == root_fen and priors:
                moves = list(priors)
                logits = np.array(
                    [math.log(max(float(priors[move]), 1e-12)) for move in moves],
                    dtype=np.float64,
                )
                logits += scale * rng.gumbel(size=len(moves))
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                priors = {
                    move: float(probability)
                    for move, probability in zip(moves, probabilities, strict=True)
                }
                root_adjusted = True
            adjusted_batch.append(priors)
        return adjusted_batch, values

    return search(
        board,
        wrapped,
        num_simulations,
        add_noise=False,
        seed=seed,
        **search_kwargs,
    )
