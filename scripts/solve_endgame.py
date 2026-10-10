"""Solve KQK / KRK / KPK exactly by retrograde analysis and save the tables.

    python -m scripts.solve_endgame --table KQK
    python -m scripts.solve_endgame --table all      # also writes the summary CSV and histogram

Tables go to ``paths.endgame_dir`` (git-ignored). The summary CSV and DTM histogram go to
``reports/midsem/``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.ticker import MaxNLocator

from alphachess.endgames import plotting
from alphachess.endgames.index import TABLES
from alphachess.endgames.solve import WIN
from alphachess.endgames.tables import endgame_dir, load_table, save_table, solve_table

ORDER = ["KQK", "KRK", "KPK"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", default="all", choices=[*ORDER, "all"])
    parser.add_argument("--out-dir", default=None, help="defaults to paths.endgame_dir")
    parser.add_argument("--report-dir", default="reports/midsem")
    return parser.parse_args()


def summary_rows(directory: Path) -> list[dict]:
    rows = []
    for name in ORDER:
        try:
            table = load_table(name, directory)
        except FileNotFoundError:
            continue
        s, meta = table.meta["summary"], table.meta
        for side in ("white_to_move", "black_to_move"):
            part = s[side]
            n = part["positions"]
            rows.append(
                {
                    "table": name,
                    "side_to_move": side.split("_")[0],
                    "positions": n,
                    "win_pct": round(100 * part["win"] / n, 2),
                    "draw_pct": round(100 * part["draw"] / n, 2),
                    "loss_pct": round(100 * part["loss"] / n, 2),
                    "max_dtm_plies": part["max_dtm_plies"],
                    "build_seconds": meta["build_seconds"],
                    "solve_seconds": meta["solve_seconds"],
                    "slots": TABLES[name].size,
                    "edges": meta["edges"],
                }
            )
    return rows


def plot_histograms(directory: Path, path: Path) -> None:
    tables = []
    for name in ORDER:
        try:
            tables.append(load_table(name, directory))
        except FileNotFoundError:
            pass
    fig, axes = plotting.setup(figsize=(4.0 * len(tables), 3.6), ncols=len(tables), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, table in zip(axes, tables, strict=True):
        half = table.spec.half
        legal = table.legal[:half]
        wins = table.dtm[:half][legal & (table.wdl[:half] == WIN)]
        moves = (wins + 1) // 2
        bins = np.arange(1, moves.max() + 2) - 0.5
        ax.hist(moves, bins=bins, color=plotting.SERIES[0], rwidth=0.85)
        draws = int((table.wdl[:half][legal] == 0).sum())
        ax.set_title(
            f"{table.spec.name}, White to move\n{legal.sum():,} positions, {draws:,} draws, "
            f"longest mate in {moves.max()}",
            fontsize=10,
        )
        ax.set_xlabel("moves to mate (optimal play)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("positions (symmetry-reduced)")
    plotting.save(fig, path)


def main() -> None:
    args = parse_args()
    directory = Path(args.out_dir) if args.out_dir else endgame_dir()
    names = ORDER if args.table == "all" else [args.table]
    solved = {}
    for name in names:
        table, _ = solve_table(name, solved)
        solved[name] = table
        path = save_table(table, directory)
        s = table.meta["summary"]
        print(
            f"{name}: {s['positions']:,} positions, build {table.meta['build_seconds']} s, "
            f"solve {table.meta['solve_seconds']} s -> {path}"
        )
        print(json.dumps(s, indent=2))

    report = Path(args.report_dir)
    rows = summary_rows(directory)
    if rows:
        (report / "results").mkdir(parents=True, exist_ok=True)
        with (report / "results" / "endgame_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plot_histograms(directory, report / "figures" / "dtm_histograms.png")


if __name__ == "__main__":
    main()
