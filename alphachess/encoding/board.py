"""Board and move encoding. Spec: docs/INTERFACES.md §1-2.

Everything is seen from the **side to move**. When Black is to move, the board is mirrored
top-to-bottom (rank r becomes rank 7 - r) and the colours are swapped, so "my pieces" always sit
in planes 0-5 and always move "up" the board. The network then never has to learn a separate
rule for Black.

Board planes (``NUM_PLANES = 19``), each 8x8 and indexed ``[plane, rank, file]`` in the flipped
frame:

=====  ============================================================
0-5    my pawns, knights, bishops, rooks, queens, king
6-11   opponent's pawns, knights, bishops, rooks, queens, king
12     repetition: all ones if this position has occurred before
13-14  my castling rights: king-side, queen-side (all ones / zeros)
15-16  opponent's castling rights: king-side, queen-side
17     colour of the side to move: all ones if White, zeros if Black
18     fifty-move counter: halfmove clock / 100
=====  ============================================================

Move index (``NUM_MOVES = 4672 = 64 * 73``), AlphaZero layout::

    index = from_square * 73 + move_type          # from_square in the flipped frame

``move_type``:

- ``0-55``  queen-like moves: ``direction * 7 + (distance - 1)``, 8 directions x 7 distances.
  Rook, bishop, queen, king, pawn pushes and captures, castling (the king's two-square step)
  and promotions to a queen all live here.
- ``56-63`` knight moves, in the order of ``KNIGHT_DELTAS``.
- ``64-72`` underpromotions: ``64 + 3 * piece + (file_delta + 1)``, with piece 0/1/2 = knight,
  bishop, rook and file_delta -1/0/+1 (capture left, push, capture right).
"""

from __future__ import annotations

import chess
import numpy as np

NUM_MOVES = 4672
NUM_PLANES = 19
NUM_MOVE_TYPES = 73

# (file delta, rank delta) for the 8 queen directions: N, NE, E, SE, S, SW, W, NW.
QUEEN_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1),
)
KNIGHT_DELTAS: tuple[tuple[int, int], ...] = (
    (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2),
)
UNDERPROMOTIONS: tuple[int, ...] = (chess.KNIGHT, chess.BISHOP, chess.ROOK)

_QUEEN_BASE = 0
_KNIGHT_BASE = 56
_UNDERPROMO_BASE = 64

_DIRECTION_INDEX = {d: i for i, d in enumerate(QUEEN_DIRECTIONS)}
_KNIGHT_INDEX = {d: i for i, d in enumerate(KNIGHT_DELTAS)}
_PIECE_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


def _flip(square: chess.Square, turn: chess.Color) -> chess.Square:
    """Map a real square into the side-to-move frame (mirror ranks when Black moves)."""
    return square if turn == chess.WHITE else chess.square_mirror(square)


def encode_board(board: chess.Board) -> np.ndarray:
    """Return float32 planes of shape (NUM_PLANES, 8, 8), side to move's perspective."""
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    me, them = board.turn, not board.turn

    for offset, color in ((0, me), (6, them)):
        for i, piece_type in enumerate(_PIECE_ORDER):
            for square in board.pieces(piece_type, color):
                sq = _flip(square, me)
                planes[offset + i, chess.square_rank(sq), chess.square_file(sq)] = 1.0

    if board.is_repetition(2):
        planes[12] = 1.0
    planes[13] = float(board.has_kingside_castling_rights(me))
    planes[14] = float(board.has_queenside_castling_rights(me))
    planes[15] = float(board.has_kingside_castling_rights(them))
    planes[16] = float(board.has_queenside_castling_rights(them))
    planes[17] = float(me == chess.WHITE)
    planes[18] = min(board.halfmove_clock, 100) / 100.0
    return planes


def _move_type(from_sq: chess.Square, to_sq: chess.Square, promotion: int | None) -> int:
    """Move type (0-72) for a move given in the flipped frame."""
    dx = chess.square_file(to_sq) - chess.square_file(from_sq)
    dy = chess.square_rank(to_sq) - chess.square_rank(from_sq)

    if promotion is not None and promotion != chess.QUEEN:
        if dy != 1 or dx not in (-1, 0, 1):
            raise ValueError(f"bad underpromotion delta ({dx}, {dy})")
        return _UNDERPROMO_BASE + 3 * UNDERPROMOTIONS.index(promotion) + (dx + 1)

    if (dx, dy) in _KNIGHT_INDEX:
        return _KNIGHT_BASE + _KNIGHT_INDEX[(dx, dy)]

    distance = max(abs(dx), abs(dy))
    if distance == 0 or (dx != 0 and dy != 0 and abs(dx) != abs(dy)):
        raise ValueError(f"not a queen-like move ({dx}, {dy})")
    direction = (int(np.sign(dx)), int(np.sign(dy)))
    return _QUEEN_BASE + _DIRECTION_INDEX[direction] * 7 + (distance - 1)


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Index in [0, 4672) of ``move`` in ``board`` (side-to-move frame)."""
    from_sq = _flip(move.from_square, board.turn)
    to_sq = _flip(move.to_square, board.turn)
    return from_sq * NUM_MOVE_TYPES + _move_type(from_sq, to_sq, move.promotion)


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Inverse of :func:`move_to_index`. The result need not be legal in ``board``."""
    if not 0 <= index < NUM_MOVES:
        raise ValueError(f"move index {index} out of range")
    from_sq, move_type = divmod(int(index), NUM_MOVE_TYPES)
    fx, fy = chess.square_file(from_sq), chess.square_rank(from_sq)
    promotion = None

    if move_type >= _UNDERPROMO_BASE:
        piece, dx_plus_one = divmod(move_type - _UNDERPROMO_BASE, 3)
        dx, dy = dx_plus_one - 1, 1
        promotion = UNDERPROMOTIONS[piece]
    elif move_type >= _KNIGHT_BASE:
        dx, dy = KNIGHT_DELTAS[move_type - _KNIGHT_BASE]
    else:
        direction, distance_minus_one = divmod(move_type - _QUEEN_BASE, 7)
        ux, uy = QUEEN_DIRECTIONS[direction]
        dx, dy = ux * (distance_minus_one + 1), uy * (distance_minus_one + 1)

    tx, ty = fx + dx, fy + dy
    if not (0 <= tx < 8 and 0 <= ty < 8):
        raise ValueError(f"move index {index} leaves the board")
    to_sq = chess.square(tx, ty)

    real_from = _flip(from_sq, board.turn)
    real_to = _flip(to_sq, board.turn)
    # Queen promotions use the queen-like planes, so restore the promotion piece here.
    if (
        promotion is None
        and ty == 7
        and move_type < _KNIGHT_BASE
        and board.piece_type_at(real_from) == chess.PAWN
    ):
        promotion = chess.QUEEN
    return chess.Move(real_from, real_to, promotion=promotion)


def legal_mask(board: chess.Board) -> np.ndarray:
    """Bool array of shape (NUM_MOVES,), True for legal moves."""
    mask = np.zeros(NUM_MOVES, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_index(move, board)] = True
    return mask
