"""Position indexing for three-piece endgames (KQK, KRK, KPK).

A position is (side to move, white king, white piece, black king); White is always the strong
side. Tables are flat arrays indexed by a canonical slot number:

- Pawnless tables (KQK, KRK) use all 8 board symmetries. The white king is moved into the
  a1-d1-d4 triangle (10 squares), giving ``2 * 10 * 64 * 64 = 81,920`` slots. Positions with the
  white king on the a1-h8 diagonal have two images in the triangle; we keep the one with the
  smaller slot number, so every position has exactly one canonical slot.
- KPK uses only the left-right mirror (pawns break the vertical symmetry): the pawn is put on
  files a-d and ranks 2-7, giving ``2 * 24 * 64 * 64 = 196,608`` slots.

Slot layout: ``index = stm * half + a * 4096 + b * 64 + c`` where ``stm`` is 0 for White to move
and 1 for Black to move. Pawnless: ``a`` = triangle slot of the white king, ``b`` = piece square,
``c`` = black king square. KPK: ``a`` = pawn slot, ``b`` = white king, ``c`` = black king.

A slot is *legal* when it is canonical and the position is legal: distinct squares, kings not
adjacent, the side not to move not in check, no pawn on rank 1 or 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import chess
import numpy as np


@dataclass(frozen=True)
class TableSpec:
    name: str
    piece: chess.PieceType
    pawn: bool

    @property
    def half(self) -> int:
        return (24 if self.pawn else 10) * 4096

    @property
    def size(self) -> int:
        return 2 * self.half

    @property
    def syzygy_name(self) -> str:
        return f"K{chess.piece_symbol(self.piece).upper()}vK"


TABLES: dict[str, TableSpec] = {
    "KQK": TableSpec("KQK", chess.QUEEN, False),
    "KRK": TableSpec("KRK", chess.ROOK, False),
    "KPK": TableSpec("KPK", chess.PAWN, True),
}


def get_spec(table: str | TableSpec) -> TableSpec:
    return table if isinstance(table, TableSpec) else TABLES[table.upper()]


# --- board symmetries ---------------------------------------------------------------------

_SQ = np.arange(64)
_FILE, _RANK = _SQ % 8, _SQ // 8


def _sym_tables() -> np.ndarray:
    """(8, 64) array: SYM[k, sq] is the image of ``sq`` under symmetry ``k``.

    Bit 0 flips files, bit 1 flips ranks, bit 2 transposes (applied last). Symmetry 1 is the
    left-right mirror, the only one allowed with pawns.
    """
    out = np.empty((8, 64), dtype=np.int64)
    for k in range(8):
        f = 7 - _FILE if k & 1 else _FILE.copy()
        r = 7 - _RANK if k & 2 else _RANK.copy()
        if k & 4:
            f, r = r, f
        out[k] = r * 8 + f
    return out


SYM = _sym_tables()
SYM_INV = np.argsort(SYM, axis=1)  # SYM_INV[k, SYM[k, sq]] == sq

TRIANGLE = [sq for sq in range(64) if _FILE[sq] <= 3 and _RANK[sq] <= _FILE[sq]]
_TRI_SLOT = np.full(64, -1, dtype=np.int64)
_TRI_SLOT[TRIANGLE] = np.arange(len(TRIANGLE))
assert len(TRIANGLE) == 10

PAWN_SQUARES = [r * 8 + f for r in range(1, 7) for f in range(4)]
_PAWN_SLOT = np.full(64, -1, dtype=np.int64)
_PAWN_SLOT[PAWN_SQUARES] = np.arange(len(PAWN_SQUARES))


def _slot(spec: TableSpec, stm, wk, p, bk):
    """Slot number for already-oriented squares (arrays or ints); -1 if not representable."""
    stm, wk, p, bk = (np.asarray(x, dtype=np.int64) for x in (stm, wk, p, bk))
    if spec.pawn:
        a = _PAWN_SLOT[p]
        return np.where(a >= 0, stm * spec.half + a * 4096 + wk * 64 + bk, -1)
    a = _TRI_SLOT[wk]
    return np.where(a >= 0, stm * spec.half + a * 4096 + p * 64 + bk, -1)


def canonical(spec: TableSpec, stm, wk, p, bk) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised canonicalisation. Returns ``(index, sym)``; index -1 if not representable.

    ``sym`` is the symmetry that maps the given squares onto the canonical position, so a move
    in canonical coordinates maps back with ``SYM_INV[sym]``.
    """
    stm, wk, p, bk = np.broadcast_arrays(*(np.asarray(x, dtype=np.int64) for x in (stm, wk, p, bk)))
    syms = (0, 1) if spec.pawn else range(8)
    best = np.full(stm.shape, np.iinfo(np.int64).max)
    best_sym = np.zeros(stm.shape, dtype=np.int64)
    for k in syms:
        idx = _slot(spec, stm, SYM[k][wk], SYM[k][p], SYM[k][bk])
        better = (idx >= 0) & (idx < best)
        best = np.where(better, idx, best)
        best_sym = np.where(better, k, best_sym)
    best = np.where(best == np.iinfo(np.int64).max, -1, best)
    return best, best_sym


def decode(spec: TableSpec, index) -> tuple[np.ndarray, ...]:
    """Inverse of the slot layout: ``(stm, wk, p, bk)`` arrays for slot numbers."""
    index = np.asarray(index, dtype=np.int64)
    stm, rest = np.divmod(index, spec.half)
    a, rest = np.divmod(rest, 4096)
    b, c = np.divmod(rest, 64)
    if spec.pawn:
        return stm, b, np.asarray(PAWN_SQUARES)[a], c
    return stm, np.asarray(TRIANGLE)[a], b, c


# --- board <-> index ------------------------------------------------------------------------


def board_squares(board: chess.Board, spec: TableSpec) -> tuple[int, int, int, int]:
    """``(stm, wk, piece, bk)`` for a board with exactly the table's material, White strong."""
    pieces = board.piece_map()
    if len(pieces) != 3:
        raise ValueError(f"{spec.name} needs exactly three pieces, got {len(pieces)}")
    extra = [sq for sq, pc in pieces.items() if pc.piece_type != chess.KING]
    if len(extra) != 1 or pieces[extra[0]] != chess.Piece(spec.piece, chess.WHITE):
        raise ValueError(f"board is not {spec.name} with White as the strong side")
    wk, bk = board.king(chess.WHITE), board.king(chess.BLACK)
    return (0 if board.turn == chess.WHITE else 1), wk, extra[0], bk


def board_to_index(board: chess.Board, table: str | TableSpec) -> int:
    spec = get_spec(table)
    idx, _ = canonical(spec, *board_squares(board, spec))
    if int(idx) < 0:
        raise ValueError(f"position cannot be indexed in {spec.name}: {board.fen()}")
    return int(idx)


def board_to_index_sym(board: chess.Board, table: str | TableSpec) -> tuple[int, int]:
    """Like :func:`board_to_index`, also returning the symmetry used."""
    spec = get_spec(table)
    idx, sym = canonical(spec, *board_squares(board, spec))
    if int(idx) < 0:
        raise ValueError(f"position cannot be indexed in {spec.name}: {board.fen()}")
    return int(idx), int(sym)


def make_board(spec: TableSpec, stm: int, wk: int, p: int, bk: int) -> chess.Board:
    board = chess.Board(None)
    board.set_piece_at(int(wk), chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(int(p), chess.Piece(spec.piece, chess.WHITE))
    board.set_piece_at(int(bk), chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE if int(stm) == 0 else chess.BLACK
    return board


def index_to_board(index: int, table: str | TableSpec) -> chess.Board:
    spec = get_spec(table)
    if not 0 <= index < spec.size:
        raise IndexError(index)
    return make_board(spec, *(int(x) for x in decode(spec, index)))


@cache
def legal_mask(table: str | TableSpec) -> np.ndarray:
    """Bool array over all slots: True where the slot is canonical and the position legal."""
    spec = get_spec(table)
    idx = np.arange(spec.size)
    stm, wk, p, bk = decode(spec, idx)
    canon, _ = canonical(spec, stm, wk, p, bk)
    candidates = (canon == idx) & (wk != p) & (wk != bk) & (p != bk)
    # Kings adjacent (Chebyshev distance 1) is illegal.
    kd = np.maximum(np.abs(wk % 8 - bk % 8), np.abs(wk // 8 - bk // 8))
    candidates &= kd > 1
    mask = np.zeros(spec.size, dtype=bool)
    for i in np.flatnonzero(candidates):
        board = make_board(spec, stm[i], wk[i], p[i], bk[i])
        # Pawn ranks are already restricted by the slot layout; only check remains.
        if not board.was_into_check():
            mask[i] = True
    mask.setflags(write=False)
    return mask


def is_legal_index(index: int, table: str | TableSpec) -> bool:
    spec = get_spec(table)
    return 0 <= index < spec.size and bool(legal_mask(spec)[index])


def map_move(move: chess.Move, sym: int, inverse: bool = True) -> chess.Move:
    """Map a move between canonical and board coordinates."""
    table = SYM_INV if inverse else SYM
    return chess.Move(
        int(table[sym][move.from_square]), int(table[sym][move.to_square]), move.promotion
    )
