"""Match scoring, Elo conversion, and paired-opening uncertainty."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RatingSummary:
    wins: int
    draws: int
    losses: int
    score: float
    elo_difference: float
    elo_low: float
    elo_high: float


def score_from_results(results: list[float]) -> float:
    if not results:
        raise ValueError("at least one result is required")
    values = np.asarray(results, dtype=np.float64)
    if not np.all(np.isin(values, [0.0, 0.5, 1.0])):
        raise ValueError("results must be 0, 0.5, or 1")
    return float(values.mean())


def score_to_elo(score: float, *, games: int | None = None) -> float:
    """Convert score to Elo, with Jeffreys smoothing at finite-sample endpoints."""
    if not 0 <= score <= 1:
        raise ValueError("score must be in [0, 1]")
    if games is not None and score in (0.0, 1.0):
        if games < 1:
            raise ValueError("games must be positive")
        score = (score * games + 0.5) / (games + 1.0)
    elif games is None:
        score = min(max(score, 1e-12), 1 - 1e-12)
    return float(400.0 * math.log10(score / (1.0 - score)))


def summarise_results(
    paired_results: list[tuple[float, float]],
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> RatingSummary:
    """Summarise colour-swapped pairs and bootstrap whole pairs for uncertainty."""
    if not paired_results:
        raise ValueError("at least one colour-swapped pair is required")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    flat = [result for pair in paired_results for result in pair]
    score = score_from_results(flat)
    wins = sum(result == 1.0 for result in flat)
    draws = sum(result == 0.5 for result in flat)
    losses = sum(result == 0.0 for result in flat)

    pairs = np.asarray(paired_results, dtype=np.float64)
    if not np.all(np.isin(pairs, [0.0, 0.5, 1.0])):
        raise ValueError("paired results must contain only 0, 0.5, or 1")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(pairs), size=(bootstrap_samples, len(pairs)))
    bootstrap_scores = pairs[indices].mean(axis=(1, 2))
    alpha = (1.0 - confidence) / 2.0
    low_score, high_score = np.quantile(bootstrap_scores, [alpha, 1 - alpha])
    games = len(flat)
    return RatingSummary(
        wins=wins,
        draws=draws,
        losses=losses,
        score=score,
        elo_difference=score_to_elo(score, games=games),
        elo_low=score_to_elo(float(low_score), games=games),
        elo_high=score_to_elo(float(high_score), games=games),
    )
