"""Board and move encoding. Spec: docs/INTERFACES.md §1-2."""

from __future__ import annotations

import chess
import numpy as np

NUM_MOVES = 4672
NUM_PLANES = 19


def encode_board(board: chess.Board) -> np.ndarray:
    """Return float32 planes of shape (NUM_PLANES, 8, 8), side to move's perspective."""
    raise NotImplementedError


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    raise NotImplementedError


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    raise NotImplementedError


def legal_mask(board: chess.Board) -> np.ndarray:
    """Bool array of shape (NUM_MOVES,), True for legal moves."""
    mask = np.zeros(NUM_MOVES, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_index(move, board)] = True
    return mask
