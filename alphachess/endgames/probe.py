"""Tablebase probe used by self-play adjudication and evaluation. Spec: docs/INTERFACES.md §7.

Lookup order:

1. Our exact DP tables when the material is KQK, KRK or KPK, for either colour. A position where
   Black is the strong side is mirrored (colours and ranks swapped) first; values are from the
   side to move, so they need no sign change, and the best move is mirrored back.
2. Syzygy, when ``paths.tablebase_dir`` holds the right files (WDL and best move; no DTM).
3. Otherwise ``None``.

Tables load lazily and are cached. Our tables ignore the 50-move rule; Syzygy WDL accounts for it
(cursed wins / blessed losses are reported here as draws).
"""

from __future__ import annotations

from functools import cache, lru_cache
from pathlib import Path

import chess
import chess.syzygy

from alphachess.common.config import load_config, repo_path

from .index import TABLES, board_to_index_sym, map_move
from .tables import Table, load_or_solve

_MATERIAL = {chess.QUEEN: "KQK", chess.ROOK: "KRK", chess.PAWN: "KPK"}


@cache
def get_table(name: str) -> Table:
    """Load (or solve once and save) one of our tables."""
    return load_or_solve(name)


def _own_table(board: chess.Board) -> tuple[str, chess.Board, bool] | None:
    """``(table name, board with White strong, mirrored?)`` if our tables cover ``board``."""
    pieces = board.piece_map()
    if len(pieces) != 3:
        return None
    extra = [pc for pc in pieces.values() if pc.piece_type != chess.KING]
    if len(extra) != 1 or extra[0].piece_type not in _MATERIAL:
        return None
    name = _MATERIAL[extra[0].piece_type]
    if name not in TABLES:
        return None
    if extra[0].color == chess.WHITE:
        return name, board, False
    return name, board.mirror(), True


def _own_lookup(board: chess.Board) -> tuple[Table, int, int, bool] | None:
    found = _own_table(board)
    if found is None:
        return None
    name, oriented, mirrored = found
    try:
        slot, sym = board_to_index_sym(oriented, name)
    except ValueError:
        return None
    table = get_table(name)
    if not table.legal[slot]:
        return None
    return table, slot, sym, mirrored


@lru_cache(maxsize=1)
def _syzygy() -> chess.syzygy.Tablebase | None:
    directory = Path(repo_path(load_config()["paths"]["tablebase_dir"]))
    if not directory.is_dir() or not any(directory.glob("*.rtbw")):
        return None
    return chess.syzygy.open_tablebase(str(directory))


def _syzygy_wdl(board: chess.Board) -> int | None:
    tb = _syzygy()
    if tb is None:
        return None
    try:
        wdl = tb.probe_wdl(board)
    except (KeyError, chess.syzygy.MissingTableError, IndexError):
        return None
    return 1 if wdl == 2 else -1 if wdl == -2 else 0


def probe_wdl(board: chess.Board) -> int | None:
    """+1 / 0 / -1 for the side to move, or None if this material isn't covered."""
    if board.is_checkmate():
        return -1
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    hit = _own_lookup(board)
    if hit is not None:
        table, slot, _, _ = hit
        return int(table.wdl[slot])
    return _syzygy_wdl(board)


def probe_dtm(board: chess.Board) -> int | None:
    """Signed plies to mate (>0 winning for side to move, <0 losing, 0 for a draw or when mated).

    Only our own tables have DTM; Syzygy positions return None.
    """
    if board.is_checkmate():
        return 0
    hit = _own_lookup(board)
    if hit is None:
        return 0 if board.is_stalemate() or board.is_insufficient_material() else None
    table, slot, _, _ = hit
    wdl, dtm = table.lookup(slot)
    return 0 if wdl == 0 else wdl * dtm


def best_move(board: chess.Board) -> chess.Move | None:
    """The optimal move (fastest win, longest loss, any drawing move), or None if not covered."""
    if board.is_game_over():
        return None
    hit = _own_lookup(board)
    if hit is not None:
        table, slot, sym, mirrored = hit
        move = table.best_move(slot)
        if move is None:
            return None
        move = map_move(move, sym)
        if mirrored:
            move = chess.Move(
                chess.square_mirror(move.from_square),
                chess.square_mirror(move.to_square),
                move.promotion,
            )
        return move
    return _syzygy_best_move(board)


def _syzygy_best_move(board: chess.Board) -> chess.Move | None:
    tb = _syzygy()
    if tb is None or _syzygy_wdl(board) is None:
        return None
    best, best_key = None, None
    for move in board.legal_moves:
        zeroing = board.is_zeroing(move)
        board.push(move)
        try:
            if board.is_checkmate():
                key = (3, 0)
            else:
                wdl = -tb.probe_wdl(board)
                dtz = tb.probe_dtz(board)
                # Win: prefer zeroing moves, then the smallest DTZ; loss: the largest DTZ.
                key = (wdl, 1 if zeroing and wdl > 0 else 0, -abs(dtz) if wdl > 0 else abs(dtz))
        except (KeyError, chess.syzygy.MissingTableError, IndexError):
            key = None
        finally:
            board.pop()
        if key is None:
            return None
        if best_key is None or key > best_key:
            best, best_key = move, key
    return best
