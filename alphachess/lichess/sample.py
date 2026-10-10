"""Sample training positions from Lichess games and write them as Parquet shards (T8).

One row per sampled position:

| column | type | meaning |
|---|---|---|
| ``fen`` | str | position before the human move |
| ``move_uci`` | str | the move played in the game (policy target) |
| ``eval_cp`` | int or null | engine eval in centipawns, **side to move's perspective** |
| ``mate`` | int or null | signed mate distance in moves, side to move's perspective (>0: mates) |
| ``elo_white`` / ``elo_black`` | int | ratings |
| ``result`` | str | ``1-0`` / ``0-1`` / ``1/2-1/2`` (PGN, White's perspective) |
| ``ply`` | int | half-moves played before ``fen`` |
| ``game_id`` | str | Lichess game id (from the ``Site`` header) |
| ``eval_source`` | str | ``lichess`` (the game's ``[%eval]`` comment) or ``stockfish`` |

Lichess stores ``[%eval]`` from White's point of view on the node *after* each move, i.e. the
evaluation of the position that move reached. We convert it to the side to move of the sampled
position, matching ``docs/INTERFACES.md`` §4. Splits are by game: a stable hash of ``game_id``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np

COLUMNS = [
    "fen",
    "move_uci",
    "eval_cp",
    "mate",
    "elo_white",
    "elo_black",
    "result",
    "ply",
    "game_id",
    "eval_source",
]
SPLITS = ("train", "val", "test")


@dataclass
class SampleConfig:
    min_elo: int = 1800
    min_base_time: int = 180
    positions_per_game: int = 8
    skip_first_plies: int = 8
    require_eval: bool = True
    val_frac: float = 0.05
    test_frac: float = 0.05

    @classmethod
    def from_config(cls, cfg: dict) -> SampleConfig:
        lc = cfg["lichess"]
        return cls(
            min_elo=lc["min_elo"],
            min_base_time=lc["min_base_time"],
            positions_per_game=lc["positions_per_game"],
            skip_first_plies=lc["skip_first_plies"],
            require_eval=lc.get("require_eval", True),
            val_frac=lc.get("val_frac", 0.05),
            test_frac=lc.get("test_frac", 0.05),
        )


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "", "?") else None
    except ValueError:
        return None


def base_time(headers: chess.pgn.Headers) -> int | None:
    """Base time in seconds from ``TimeControl`` (``"180+2"`` -> 180); None if untimed."""
    tc = headers.get("TimeControl", "-")
    return _int(tc.split("+")[0]) if tc not in ("-", "?") else None


def keep_headers(headers: chess.pgn.Headers, cfg: SampleConfig) -> bool:
    """Rated standard games, both Elos >= ``min_elo``, base time >= ``min_base_time``."""
    if "rated" not in headers.get("Event", "").lower():
        return False
    if headers.get("Variant", "Standard") != "Standard":
        return False
    if headers.get("Result") not in ("1-0", "0-1", "1/2-1/2"):
        return False
    elos = _int(headers.get("WhiteElo")), _int(headers.get("BlackElo"))
    if any(e is None or e < cfg.min_elo for e in elos):
        return False
    base = base_time(headers)
    return base is not None and base >= cfg.min_base_time


def game_id(headers: chess.pgn.Headers) -> str:
    site = headers.get("Site", "")
    return site.rstrip("/").rsplit("/", 1)[-1] if site else headers.get("GameId", "")


def split_of(gid: str, cfg: SampleConfig) -> str:
    """Deterministic split by game: the same game always lands in the same split."""
    bucket = int(hashlib.md5(gid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < cfg.test_frac:
        return "test"
    if bucket < cfg.test_frac + cfg.val_frac:
        return "val"
    return "train"


def _eval_for(board: chess.Board, score: chess.engine.PovScore | None) -> tuple[int | None, ...]:
    if score is None:
        return None, None
    pov = score.pov(board.turn)
    return pov.score(), pov.mate()


def sample_game(game: chess.pgn.Game, cfg: SampleConfig) -> list[dict]:
    """Up to ``positions_per_game`` rows from one game (seeded by its id, so reproducible)."""
    headers = game.headers
    gid = game_id(headers)
    board = game.board()
    candidates = []
    prev_eval = None  # eval of the current position = comment on the move that reached it
    for ply, node in enumerate(game.mainline()):
        move = node.move
        if ply >= cfg.skip_first_plies:
            cp, mate = _eval_for(board, prev_eval)
            if cp is not None or mate is not None or not cfg.require_eval:
                candidates.append((ply, board.fen(), move.uci(), cp, mate))
        board.push(move)
        prev_eval = node.eval()
    if not candidates:
        return []
    seed = int(hashlib.md5(gid.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    k = min(cfg.positions_per_game, len(candidates))
    picks = sorted(rng.choice(len(candidates), size=k, replace=False))
    rows = []
    for i in picks:
        ply, fen, move, cp, mate = candidates[i]
        rows.append(
            {
                "fen": fen,
                "move_uci": move,
                "eval_cp": cp,
                "mate": mate,
                "elo_white": _int(headers.get("WhiteElo")),
                "elo_black": _int(headers.get("BlackElo")),
                "result": headers.get("Result"),
                "ply": ply,
                "game_id": gid,
                "eval_source": "lichess" if cp is not None or mate is not None else "none",
            }
        )
    return rows


def has_evals(game: chess.pgn.Game) -> bool:
    """Lichess annotates every move of an analysed game; the first move is enough to check."""
    first = game.next()
    return first is not None and first.eval() is not None


class StockfishAnnotator:
    """Optional fallback: fill missing evals with a fixed-depth Stockfish search."""

    def __init__(self, path: str, depth: int) -> None:
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.depth = depth

    def __call__(self, row: dict) -> dict:
        board = chess.Board(row["fen"])
        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        score = info["score"].pov(board.turn)
        return {**row, "eval_cp": score.score(), "mate": score.mate(), "eval_source": "stockfish"}

    def close(self) -> None:
        self.engine.quit()


class ShardWriter:
    """Buffers rows per split and writes ``<split>-NNNNN.parquet`` files of ``shard_size`` rows."""

    def __init__(self, out_dir: str | Path, shard_size: int) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.buffers: dict[str, list[dict]] = {s: [] for s in SPLITS}
        self.counts = Counter()
        self.shards: list[str] = []

    def add(self, split: str, row: dict) -> None:
        self.buffers[split].append(row)
        self.counts[split] += 1
        if len(self.buffers[split]) >= self.shard_size:
            self._flush(split)

    def _flush(self, split: str) -> None:
        rows = self.buffers[split]
        if not rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        index = sum(name.startswith(split + "-") for name in self.shards)
        name = f"{split}-{index:05d}.parquet"
        table = pa.Table.from_pylist(rows, schema=SCHEMA())
        pq.write_table(table, self.out_dir / name)
        self.shards.append(name)
        self.buffers[split] = []

    def close(self) -> None:
        for split in SPLITS:
            self._flush(split)


def SCHEMA():
    import pyarrow as pa

    return pa.schema(
        [
            ("fen", pa.string()),
            ("move_uci", pa.string()),
            ("eval_cp", pa.int32()),
            ("mate", pa.int32()),
            ("elo_white", pa.int32()),
            ("elo_black", pa.int32()),
            ("result", pa.string()),
            ("ply", pa.int32()),
            ("game_id", pa.string()),
            ("eval_source", pa.string()),
        ]
    )


@dataclass
class BuildStats:
    games_seen: int = 0
    games_kept_headers: int = 0
    games_with_evals: int = 0
    games_used: int = 0
    positions: int = 0
    positions_by_split: dict = field(default_factory=dict)
    seconds: float = 0.0
    eval_cp_histogram: dict = field(default_factory=dict)
    mate_positions: int = 0

    def as_dict(self) -> dict:
        out = dict(self.__dict__)
        out["games_per_second"] = round(self.games_seen / self.seconds, 1) if self.seconds else 0
        out["positions_per_second"] = round(self.positions / self.seconds, 1) if self.seconds else 0
        return out


def build_dataset(
    games: Iterable[chess.pgn.Game],
    writer: ShardWriter,
    cfg: SampleConfig,
    target_positions: int,
    stream_stats: dict | None = None,
    annotate: Callable[[dict], dict] | None = None,
    log_every: int = 50_000,
    log: Callable[[str], None] = print,
) -> BuildStats:
    """Sample from ``games`` (already header-filtered) into ``writer`` until the target."""
    stats = BuildStats()
    started = time.perf_counter()
    bins = np.array([-1000, -500, -200, -100, -50, 0, 50, 100, 200, 500, 1000])
    hist = np.zeros(len(bins) + 1, dtype=np.int64)
    for game in games:
        stats.games_kept_headers += 1
        evals = has_evals(game)
        stats.games_with_evals += evals
        if cfg.require_eval and not evals:
            continue
        rows = sample_game(game, cfg)
        if not rows:
            continue
        stats.games_used += 1
        split = split_of(rows[0]["game_id"], cfg)
        for row in rows:
            if row["eval_cp"] is None and row["mate"] is None and annotate is not None:
                row = annotate(row)
            writer.add(split, row)
            stats.positions += 1
            if row["eval_cp"] is not None:
                hist[np.searchsorted(bins, row["eval_cp"], side="right")] += 1
            elif row["mate"] is not None:
                stats.mate_positions += 1
        if stats.positions // log_every != (stats.positions - len(rows)) // log_every:
            elapsed = time.perf_counter() - started
            seen = (stream_stats or {}).get("games_seen", 0)
            log(
                f"{stats.positions:,} positions from {stats.games_used:,} games "
                f"({seen:,} scanned) in {elapsed:.0f} s"
            )
        if stats.positions >= target_positions:
            break
    writer.close()
    stats.seconds = time.perf_counter() - started
    stats.games_seen = (stream_stats or {}).get("games_seen", stats.games_kept_headers)
    stats.positions_by_split = dict(writer.counts)
    labels = [f"<{bins[0]}"] + [f"[{a},{b})" for a, b in itertools.pairwise(bins)]
    stats.eval_cp_histogram = dict(zip(labels + [f">={bins[-1]}"], hist.tolist(), strict=True))
    return stats


def write_summary(stats: BuildStats, out_dir: str | Path, extra: dict | None = None) -> Path:
    path = Path(out_dir) / "summary.json"
    path.write_text(json.dumps({**stats.as_dict(), **(extra or {})}, indent=2) + "\n")
    return path
