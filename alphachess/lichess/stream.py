"""Stream games from the Lichess open database without keeping the file on disk (T8).

``open_pgn(source)`` accepts a ``https://`` URL, a local ``.pgn.zst`` file or a plain ``.pgn`` file
and returns a text stream. ``.zst`` input is decompressed on the fly with ``zstandard``, so the
monthly file (tens of GB compressed) is never written to disk.

``iter_games(stream, keep_headers)`` parses games with ``chess.pgn.read_game``. Games whose headers
fail ``keep_headers`` are skipped before their moves are parsed, which is most of the speed-up:
only games that pass the Elo and time-control filters pay for move parsing.
"""

from __future__ import annotations

import io
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import chess.pgn

DATABASE_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"


def month_url(month: str) -> str:
    """URL of one month's standard rated games, e.g. ``month_url("2024-01")``."""
    return DATABASE_URL.format(month=month)


@contextmanager
def open_pgn(source: str | Path):
    """Yield a text stream of PGN from a URL, a ``.pgn.zst`` file or a ``.pgn`` file."""
    source = str(source)
    is_url = source.startswith(("http://", "https://"))
    if not source.endswith(".zst"):
        if is_url:
            with urllib.request.urlopen(source) as response:
                yield io.TextIOWrapper(response, encoding="utf-8", errors="replace")
        else:
            with open(source, encoding="utf-8", errors="replace") as handle:
                yield handle
        return

    import zstandard

    raw = urllib.request.urlopen(source) if is_url else open(source, "rb")  # noqa: SIM115
    try:
        reader = zstandard.ZstdDecompressor().stream_reader(raw, read_size=1 << 20)
        yield io.TextIOWrapper(
            io.BufferedReader(reader, 1 << 20), encoding="utf-8", errors="replace"
        )
    finally:
        raw.close()


class _FilteringBuilder(chess.pgn.GameBuilder):
    """GameBuilder that skips the move text of games whose headers fail a predicate."""

    def __init__(self, keep: Callable[[chess.pgn.Headers], bool]) -> None:
        super().__init__()
        self.keep = keep
        self.skipped = False

    def begin_game(self) -> None:
        super().begin_game()
        self.skipped = False

    def end_headers(self):
        if not self.keep(self.game.headers):
            self.skipped = True
            return chess.pgn.SKIP
        return None


def iter_games(
    stream,
    keep_headers: Callable[[chess.pgn.Headers], bool] | None = None,
    stats: dict | None = None,
) -> Iterator[chess.pgn.Game]:
    """Yield parsed games that pass ``keep_headers``. ``stats['games_seen']`` counts all games."""
    keep = keep_headers or (lambda _headers: True)
    stats = stats if stats is not None else {}
    stats.setdefault("games_seen", 0)
    while True:
        builder = _FilteringBuilder(keep)
        game = chess.pgn.read_game(stream, Visitor=lambda b=builder: b)
        if game is None:
            return
        stats["games_seen"] += 1
        if builder.skipped:
            continue
        yield game
