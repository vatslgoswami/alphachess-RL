"""Tests for board and move encoding (INTERFACES §1-2)."""

from __future__ import annotations

import random

import chess
import numpy as np
import pytest

from alphachess.encoding.board import (
    NUM_MOVES,
    NUM_PLANES,
    encode_board,
    index_to_move,
    legal_mask,
    move_to_index,
)

# Positions chosen to hit every special case: castling both ways for both colours, en passant,
# promotions and underpromotions with captures, Black to move, and a busy middlegame.
SPECIAL_FENS = [
    chess.STARTING_FEN,
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  # all four castles available, White
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",  # same, Black
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",  # en passant exf6
    "rnbqkbnr/pppp1ppp/8/8/3PpP2/8/PPP1P1PP/RNBQKBNR b KQkq f3 0 3",  # Black en passant
    "1r2k3/2P5/8/8/8/8/5p2/4K1R1 w - - 0 1",  # White promotes, incl. capture on b8
    "1r2k3/2P5/8/8/8/8/5p2/4K1R1 b - - 0 1",  # Black promotes, incl. capture on g1
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
]


def random_positions(n: int, seed: int = 0) -> list[chess.Board]:
    """Positions from random games, so the round-trip sees many move kinds."""
    rng = random.Random(seed)
    boards = []
    while len(boards) < n:
        board = chess.Board()
        for _ in range(rng.randint(0, 120)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        boards.append(board)
    return boards


def all_boards() -> list[chess.Board]:
    return [chess.Board(f) for f in SPECIAL_FENS] + random_positions(300)


def test_constants():
    assert NUM_MOVES == 8 * 8 * 73
    assert NUM_PLANES == 19


@pytest.mark.parametrize("board", all_boards(), ids=lambda b: b.fen())
def test_move_round_trip(board):
    """INTERFACES §2 invariant: index_to_move(move_to_index(m, b), b) == m for every legal m."""
    seen = set()
    for move in board.legal_moves:
        index = move_to_index(move, board)
        assert 0 <= index < NUM_MOVES
        assert index not in seen, f"two legal moves share index {index}"
        seen.add(index)
        assert index_to_move(index, board) == move


def test_legal_mask_matches_legal_moves():
    for board in all_boards():
        mask = legal_mask(board)
        assert mask.dtype == bool and mask.shape == (NUM_MOVES,)
        assert mask.sum() == board.legal_moves.count()


def test_known_indices():
    board = chess.Board()
    # e2e4: from e2 (square 12), north (direction 0), distance 2 -> type 1.
    assert move_to_index(chess.Move.from_uci("e2e4"), board) == 12 * 73 + 1
    # g1f3: from g1 (square 6), knight jump (-1, +2) = KNIGHT_DELTAS[7] -> type 56 + 7.
    assert move_to_index(chess.Move.from_uci("g1f3"), board) == 6 * 73 + 56 + 7


def test_black_moves_mirror_white_moves():
    """1. e4 for White and 1... e5 for Black are the same move in the side-to-move frame."""
    white = chess.Board()
    black = chess.Board()
    black.push_uci("e2e4")
    w = move_to_index(chess.Move.from_uci("e2e4"), white)
    b = move_to_index(chess.Move.from_uci("e7e5"), black)
    assert w == b


def test_underpromotions_have_their_own_types():
    board = chess.Board("1r2k3/2P5/8/8/8/8/8/4K3 w - - 0 1")
    queen = move_to_index(chess.Move.from_uci("c7c8q"), board)
    knight = move_to_index(chess.Move.from_uci("c7c8n"), board)
    assert queen % 73 < 56  # queen promotion: queen-like plane
    assert knight % 73 >= 64  # underpromotion plane
    assert index_to_move(queen, board).promotion == chess.QUEEN
    capture_rook = move_to_index(chess.Move.from_uci("c7b8r"), board)
    assert index_to_move(capture_rook, board) == chess.Move.from_uci("c7b8r")


def test_encode_board_shape_and_start_position():
    planes = encode_board(chess.Board())
    assert planes.shape == (NUM_PLANES, 8, 8) and planes.dtype == np.float32
    assert planes[0].sum() == 8 and planes[0][1].sum() == 8  # my pawns on rank 2
    assert planes[6].sum() == 8 and planes[6][6].sum() == 8  # their pawns on rank 7
    assert planes[5][0, 4] == 1  # my king on e1
    assert np.all(planes[13:17] == 1)  # all castling rights
    assert np.all(planes[17] == 1)  # White to move
    assert np.all(planes[18] == 0)


def test_encode_board_is_colour_symmetric():
    """Mirroring the board and swapping colours must give identical planes (except colour)."""
    for board in all_boards():
        mirrored = board.mirror()
        a, b = encode_board(board), encode_board(mirrored)
        assert np.array_equal(np.delete(a, 17, axis=0), np.delete(b, 17, axis=0))
        assert a[17, 0, 0] != b[17, 0, 0]


def test_repetition_and_fifty_move_planes():
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
        board.push_uci(uci)
    planes = encode_board(board)
    assert np.all(planes[12] == 1)  # start position seen before
    assert planes[18, 0, 0] == pytest.approx(4 / 100)


def test_lichess_encoder_now_ready():
    from alphachess.lichess.encode import encoder_ready

    assert encoder_ready(encode_board, move_to_index) is None
