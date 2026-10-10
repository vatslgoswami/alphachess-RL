"""Load deterministic opening suites."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn


@dataclass(frozen=True)
class Opening:
    opening_id: str
    fen: str

    def board(self) -> chess.Board:
        return chess.Board(self.fen)


def load_fens(path: str | Path) -> list[Opening]:
    openings = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        board = chess.Board(value)
        openings.append(Opening(f"fen-{line_number}", board.fen()))
    if not openings:
        raise ValueError(f"opening file contains no positions: {path}")
    return openings

def load_pgn(path: str | Path, *, plies: int | None = None) -> list[Opening]:
    openings = []
    with Path(path).open(encoding="utf-8") as handle:
        game_number = 0
        while game := chess.pgn.read_game(handle):
            game_number += 1
            board = game.board()
            for index, move in enumerate(game.mainline_moves()):
                if plies is not None and index >= plies:
                    break
                board.push(move)
            openings.append(Opening(f"pgn-{game_number}", board.fen()))
    if not openings:
        raise ValueError(f"opening file contains no games: {path}")
    return openings
