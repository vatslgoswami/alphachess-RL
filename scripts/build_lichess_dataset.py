"""Stream a month of Lichess games and write sampled positions as Parquet shards (T8).

    python -m scripts.build_lichess_dataset --month 2024-01 --target 2000000
    python -m scripts.build_lichess_dataset --month 2024-01 --target 5000          # smoke run
    python -m scripts.build_lichess_dataset --source games.pgn.zst --target 10000  # local file

Shards go to ``<paths.lichess_dir>/raw/<month>/{train,val,test}-NNNNN.parquet`` with a
``summary.json`` of counts, throughput and the eval distribution. Settings come from the
``lichess:`` section of config.yaml; flags override them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alphachess.common.config import load_config, repo_path
from alphachess.lichess.sample import (
    SampleConfig,
    ShardWriter,
    StockfishAnnotator,
    build_dataset,
    keep_headers,
    write_summary,
)
from alphachess.lichess.stream import iter_games, month_url, open_pgn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--month", default=None, help="YYYY-MM (default: lichess.month)")
    parser.add_argument("--source", default=None, help="local .pgn/.pgn.zst or URL instead")
    parser.add_argument("--target", type=int, default=None, help="default: target_positions")
    parser.add_argument("--max-games", type=int, default=None, help="stop after N games kept")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument(
        "--allow-no-eval",
        action="store_true",
        help="also use games without [%%eval] (pair with --stockfish-fallback for value targets)",
    )
    parser.add_argument("--stockfish-fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    lc = cfg["lichess"]
    month = args.month or lc["month"]
    source = args.source or month_url(month)
    target = args.target or lc["target_positions"]
    sample_cfg = SampleConfig.from_config(cfg)
    if args.allow_no_eval:
        sample_cfg.require_eval = False
    name = month if args.source is None else Path(args.source).name.split(".")[0]
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else repo_path(cfg["paths"]["lichess_dir"]) / "raw" / name
    )
    writer = ShardWriter(out_dir, args.shard_size or lc["shard_size"])
    annotate = None
    if args.stockfish_fallback:
        annotate = StockfishAnnotator(cfg["eval"]["stockfish_path"], lc["stockfish_depth"])

    print(f"source: {source}\ntarget: {target:,} positions -> {out_dir}")
    stream_stats: dict = {}
    try:
        with open_pgn(source) as stream:
            games = iter_games(stream, lambda h: keep_headers(h, sample_cfg), stream_stats)
            if args.max_games:
                games = (g for i, g in enumerate(games) if i < args.max_games)
            stats = build_dataset(games, writer, sample_cfg, target, stream_stats, annotate)
    finally:
        if annotate is not None:
            annotate.close()
    extra = {
        "source": source,
        "config": sample_cfg.__dict__,
        "target": target,
        "shards": writer.shards,
        "extrapolated_hours_for_2M": round(
            2_000_000 / max(stats.positions, 1) * stats.seconds / 3600, 2
        ),
    }
    path = write_summary(stats, out_dir, extra)
    print(json.dumps({**stats.as_dict(), **extra}, indent=2))
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
