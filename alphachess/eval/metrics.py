"""Position-level evaluation metrics used alongside match results."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import chess

from .player import Player


@dataclass(frozen=True)
class PositionMetricSummary:
    positions: int
    top_move_agreement: float
    mean_centipawn_loss: float


@dataclass(frozen=True)
class PuzzleMetricSummary:
    attempted: int
    solved: int
    solve_rate: float


ReferenceAnalyser = Callable[[chess.Board], tuple[chess.Move, dict[chess.Move, int]]]


def evaluate_positions(
    player: Player,
    boards: Iterable[chess.Board],
    analyse: ReferenceAnalyser,
) -> PositionMetricSummary:
    agreements = 0
    losses = []
    for board in boards:
        best_move, scores = analyse(board.copy(stack=True))
        decision = player.select_move(board.copy(stack=True))
        if decision.move is None:
            continue
        agreements += decision.move == best_move
        best_score = scores[best_move]
        chosen_score = scores.get(decision.move, min(scores.values()))
        losses.append(max(0, best_score - chosen_score))
    if not losses:
        raise ValueError("no positions produced a move")
    return PositionMetricSummary(
        positions=len(losses),
        top_move_agreement=agreements / len(losses),
        mean_centipawn_loss=sum(losses) / len(losses),
    )


def puzzle_solve_rate(solved: Iterable[bool]) -> float:
    outcomes = list(solved)
    if not outcomes:
        raise ValueError("at least one puzzle result is required")
    return sum(outcomes) / len(outcomes)


def evaluate_lichess_puzzles(
    player: Player,
    rows: Iterable[Mapping[str, str]],
    *,
    limit: int | None = None,
) -> PuzzleMetricSummary:
    """Evaluate complete Lichess puzzle lines.

    Lichess stores the position before the opponent's setup move. The first move
    is applied, then the player must match solution moves at indices 1, 3, ...;
    intervening opponent replies are taken from the recorded line.
    """
    solved = []
    for row in rows:
        if limit is not None and len(solved) >= limit:
            break
        moves = [chess.Move.from_uci(value) for value in row["Moves"].split()]
        if len(moves) < 2:
            continue
        board = chess.Board(row["FEN"])
        try:
            board.push(moves[0])
        except ValueError:
            continue
        success = True
        for index, expected in enumerate(moves[1:], start=1):
            if expected not in board.legal_moves:
                success = False
                break
            if index % 2 == 1:
                decision = player.select_move(board.copy(stack=True))
                if decision.move != expected:
                    success = False
                    break
            board.push(expected)
        solved.append(success)
    if not solved:
        raise ValueError("no valid puzzles were evaluated")
    count = sum(solved)
    return PuzzleMetricSummary(len(solved), count, count / len(solved))
