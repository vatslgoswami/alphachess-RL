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
from alphachess.endgames.index import TABLES, decode
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


def kpk_by_pawn_square(directory: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """White-to-move win % per pawn square (ranks 2-7 x files a-d; e-h are mirror images), and
    per pawn file when the black king stands on the pawn's file in front of it."""
    try:
        table = load_table("KPK", directory)
    except FileNotFoundError:
        return None
    half = table.spec.half
    _, _, pawn, bk = decode(table.spec, np.arange(half))
    legal = table.legal[:half]
    win = (table.wdl[:half] == WIN) & legal
    by_square = 100 * win.reshape(24, 4096).sum(1) / legal.reshape(24, 4096).sum(1)
    front = legal & (bk % 8 == pawn % 8) & (bk // 8 > pawn // 8)
    by_file = np.array(
        [
            100 * win[front & (pawn % 8 == f)].sum() / (front & (pawn % 8 == f)).sum()
            for f in range(4)
        ]
    )
    return by_square.reshape(6, 4), by_file


def plot_kpk(pct: np.ndarray, front: np.ndarray, path: Path) -> None:
    full = np.concatenate([pct, pct[:, ::-1]], axis=1)  # mirror to files a-h
    fig, (ax, bx) = plotting.setup(
        figsize=(10.0, 4.2), ncols=2, gridspec_kw={"width_ratios": [1.35, 1]}
    )
    image = ax.imshow(full, origin="lower", cmap="Blues", vmin=0, vmax=100)
    for r in range(6):
        for f in range(8):
            ax.text(
                f,
                r,
                f"{full[r, f]:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if full[r, f] > 60 else plotting.TEXT,
            )
    ax.set_xticks(range(8), list("abcdefgh"))
    ax.set_yticks(range(6), [str(r) for r in range(2, 8)])
    ax.grid(False)
    ax.set_xlabel("pawn file")
    ax.set_ylabel("pawn rank")
    ax.set_title("All positions: % won, by pawn square")
    fig.colorbar(image, ax=ax, shrink=0.8, label="% won")

    files = ["a / h", "b / g", "c / f", "d / e"]
    bars = bx.bar(files, front, color=plotting.SERIES[0], width=0.6)
    bx.bar_label(
        bars,
        labels=[f"{v:.1f}%" for v in front],
        color=plotting.TEXT_SECONDARY,
        fontsize=9,
        padding=2,
    )
    bx.set_ylim(0, max(front) * 1.25)
    bx.grid(axis="x", visible=False)
    bx.set_xlabel("pawn file")
    bx.set_ylabel("% won")
    bx.set_title("Black king in front of the pawn")
    fig.suptitle("KPK, White to move", color=plotting.TEXT)
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
    kpk = kpk_by_pawn_square(directory)
    if kpk is not None:
        pct, front = kpk
        with (report / "results" / "kpk_win_by_pawn_square.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["rank", "a", "b", "c", "d"])
            for r in range(6):
                writer.writerow([r + 2, *(round(float(x), 1) for x in pct[r])])
            writer.writerow(["king_in_front", *(round(float(x), 1) for x in front)])
        plot_kpk(pct, front, report / "figures" / "kpk_win_by_pawn_square.png")


if __name__ == "__main__":
    main()
