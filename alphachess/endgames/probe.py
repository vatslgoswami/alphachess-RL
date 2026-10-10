"""Tablebase probe used by self-play adjudication and evaluation. Spec: docs/INTERFACES.md §7."""

from __future__ import annotations

import chess


def probe_wdl(board: chess.Board) -> int | None:
    """+1 / 0 / -1 for the side to move, or None if this material isn't covered."""
    raise NotImplementedError


def probe_dtm(board: chess.Board) -> int | None:
    """Signed plies to mate (>0 winning for side to move), or None if not covered."""
    raise NotImplementedError


def best_move(board: chess.Board) -> chess.Move | None:
    raise NotImplementedError
