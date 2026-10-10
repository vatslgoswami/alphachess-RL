"""Fast correctness tests for PUCT and Gumbel-root search."""

import chess
import numpy as np
import pytest

from alphachess.mcts import gumbel_search, search, uniform_evaluator


def preferred_move_evaluator(preferred_uci: str, leaf_value: float = 0.0):
    def evaluator(boards):
        priors = []
        values = []
        for board in boards:
            legal = list(board.legal_moves)
            preferred = chess.Move.from_uci(preferred_uci)
            priors.append({move: (1.0 if move == preferred else 0.0) for move in legal})
            values.append(leaf_value if board.ply() else 0.0)
        return priors, np.asarray(values, dtype=np.float32)

    return evaluator


def test_zero_simulations_uses_root_prior_without_mutating_board():
    board = chess.Board()
    original = board.fen()
    result = search(board, preferred_move_evaluator("e2e4"), 0)
    assert result.best_move == chess.Move.from_uci("e2e4")
    assert result.visits[chess.Move.from_uci("e2e4")] == 0
    assert board.fen() == original


def test_backup_flips_leaf_value_to_root_perspective():
    result = search(chess.Board(), preferred_move_evaluator("e2e4", leaf_value=-0.75), 1)
    assert result.best_move == chess.Move.from_uci("e2e4")
    assert result.q == pytest.approx(0.75)


@pytest.mark.parametrize(
    "fen, expected",
    [
        ("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1", -1.0),
        ("7k/5Q2/5K2/8/8/8/8/8 b - - 0 1", 0.0),
    ],
)
def test_terminal_root_never_calls_evaluator(fen, expected):
    def fail(_boards):
        raise AssertionError("terminal positions must not be evaluated")

    result = search(chess.Board(fen), fail, 10)
    assert result.best_move is None
    assert result.q == expected


def test_search_finds_a_mate_in_one_for_white():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    result = search(board, uniform_evaluator, 80)
    assert result.best_move is not None
    board.push(result.best_move)
    assert board.is_checkmate()


def test_batching_completes_exact_simulation_count_and_reduces_calls():
    result = search(chess.Board(), uniform_evaluator, 16, batch_size=4)
    assert sum(result.visits.values()) == 16
    assert result.diagnostics.simulations == 16
    assert result.diagnostics.evaluator_calls <= 5  # root plus four leaf batches
    assert result.diagnostics.mean_batch_size > 1


def test_invalid_evaluator_batch_is_rejected():
    def broken(_boards):
        return [], np.zeros(0, dtype=np.float32)

    with pytest.raises(ValueError, match="batch size"):
        search(chess.Board(), broken, 1)


def test_dirichlet_noise_and_gumbel_are_seeded():
    board = chess.Board()
    first = search(board, uniform_evaluator, 8, add_noise=True, seed=9)
    second = search(board, uniform_evaluator, 8, add_noise=True, seed=9)
    assert first.visits == second.visits

    gumbel_first = gumbel_search(board, uniform_evaluator, 8, seed=12)
    gumbel_second = gumbel_search(board, uniform_evaluator, 8, seed=12)
    assert gumbel_first.visits == gumbel_second.visits
