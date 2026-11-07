"""Build the Lichess position dataset from the Hugging Face copy, in parallel (T8 alternative).

    python -m scripts.build_lichess_hf --month 2024-01 --target 4000000 --workers 24
    python -m scripts.build_lichess_hf --month 2024-01 --target 0       # every eval game

Same output as ``scripts/build_lichess_dataset.py`` (``data/lichess/raw/<month>/`` Parquet shards
with ``fen, move_uci, eval_cp, mate, ...`` and a ``summary.json``), so the encoder and trainer
are unchanged. Use it where database.lichess.org is unreachable (e.g. blocked by a firewall) or
to go faster: the month's files are downloaded and sampled by ``--workers`` processes at once.
With a positive ``--target`` it stops once that many positions are written.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from alphachess.common.config import load_config, repo_path
from alphachess.lichess.hf import fetch_and_process, list_month_files
from alphachess.lichess.sample import (
    BuildStats,
    SampleConfig,
    ShardWriter,
    split_of,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--month", required=True, help="YYYY-MM")
    parser.add_argument("--target", type=int, default=None,
                        help="positions to write; 0 = all (default: lichess.target_positions)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--cache-dir", default=None, help="where files are downloaded briefly")
    parser.add_argument("--max-files", type=int, default=None, help="only the first N files")
    parser.add_argument("--insecure-download", action="store_true",
                        help="skip TLS certificate checks for huggingface.co (firewalls)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    lc = cfg["lichess"]
    target = lc["target_positions"] if args.target is None else args.target
    sample_cfg = SampleConfig.from_config(cfg)
    base = repo_path(cfg["paths"]["lichess_dir"])
    out_dir = Path(args.out_dir) if args.out_dir else base / "raw" / args.month
    cache_dir = Path(args.cache_dir) if args.cache_dir else base / "hf_cache" / args.month

    files = list_month_files(args.month, args.insecure_download)
    # Shuffled (fixed seed) so a run stopped early still covers the whole month, not its first days.
    random.Random(0).shuffle(files)
    if args.max_files:
        files = files[: args.max_files]
    goal = f"target {target:,} positions" if target else "all eval games"
    stop_file = out_dir / "STOP"
    stop_file.unlink(missing_ok=True)
    print(f"{args.month}: {len(files)} Parquet files, {args.workers} workers, {goal}\n"
          f"to stop early and keep what is done: touch {stop_file}", flush=True)

    writer = ShardWriter(out_dir, lc["shard_size"])
    stats = BuildStats()
    started = time.perf_counter()
    jobs = [(f, str(cache_dir), asdict(sample_cfg), args.insecure_download) for f in files]
    done_files = 0
    stopped = ""
    pool = mp.get_context("spawn").Pool(args.workers)
    try:
        for result in pool.imap_unordered(fetch_and_process, jobs):
            done_files += 1
            stats.games_seen += result["games_seen"]
            stats.games_kept_headers += result["games_kept_headers"]
            stats.games_with_evals += result["games_with_evals"]
            for game_rows in result["rows"]:
                stats.games_used += 1
                split = split_of(game_rows[0]["game_id"], sample_cfg)
                for row in game_rows:
                    writer.add(split, row)
                    stats.positions += 1
                    if row["eval_cp"] is None and row["mate"] is not None:
                        stats.mate_positions += 1
                if target and stats.positions >= target:
                    break
            elapsed = time.perf_counter() - started
            eta = elapsed / done_files * (len(files) - done_files)
            print(f"[{done_files}/{len(files)} files] {stats.positions:,} positions from "
                  f"{stats.games_used:,} games ({stats.games_seen:,} scanned) in {elapsed:.0f} s"
                  f", all files in ~{eta / 60:.0f} more min", flush=True)
            if target and stats.positions >= target:
                stopped = "target reached"
                break
            if stop_file.exists():
                stopped = "STOP file"
                break
    except KeyboardInterrupt:
        stopped = "interrupted"
    finally:
        pool.terminate()
        pool.join()
    if stopped:
        print(f"stopped early ({stopped}) after {done_files}/{len(files)} files", flush=True)
    writer.close()
    shutil.rmtree(cache_dir, ignore_errors=True)  # files left by workers stopped early
    stats.seconds = time.perf_counter() - started
    stats.positions_by_split = dict(writer.counts)
    extra = {
        "source": f"huggingface:{args.month}",
        "files_processed": done_files,
        "stopped_early": stopped or None,
        "files_total": len(files),
        "config": sample_cfg.__dict__,
        "target": target,
        "shards": writer.shards,
    }
    path = write_summary(stats, out_dir, extra)
    print(json.dumps({**stats.as_dict(), **extra}, indent=2))
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
