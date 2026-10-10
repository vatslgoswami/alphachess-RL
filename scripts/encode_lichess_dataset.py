"""Encode raw Lichess Parquet shards into ``.npz`` training shards (T9, INTERFACES §5).

    python -m scripts.encode_lichess_dataset --month 2024-01

Reads ``<paths.lichess_dir>/raw/<month>/*.parquet`` and writes
``<paths.lichess_dir>/encoded/<month>/*.npz``. Exits with a clear message while
``alphachess.encoding`` is still a stub.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alphachess.common.config import load_config, repo_path
from alphachess.encoding.board import encode_board, move_to_index
from alphachess.lichess.encode import encode_parquet, encoder_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--month", default=None, help="YYYY-MM (default: lichess.month)")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    problem = encoder_ready(encode_board, move_to_index)
    if problem:
        raise SystemExit(problem)
    month = args.month or cfg["lichess"]["month"]
    base = repo_path(cfg["paths"]["lichess_dir"])
    raw = Path(args.raw_dir) if args.raw_dir else base / "raw" / month
    out = Path(args.out_dir) if args.out_dir else base / "encoded" / month
    shards = sorted(raw.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"no Parquet shards in {raw}; run scripts/build_lichess_dataset.py first")
    total = 0
    for shard in shards:
        n = encode_parquet(
            shard,
            out / f"{shard.stem}.npz",
            encode_board,
            move_to_index,
            cfg["lichess"]["value_scale"],
        )
        total += n
        print(f"{shard.name}: {n:,} positions -> {out / (shard.stem + '.npz')}")
    print(f"done: {total:,} positions in {len(shards)} shards")


if __name__ == "__main__":
    main()
