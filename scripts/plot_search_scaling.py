"""Build the midsem Elo-versus-simulations figure from completed run directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output-dir", default="reports/midsem/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for value in args.run_dirs:
        run_dir = Path(value)
        config = json.loads((run_dir / "config.json").read_text())
        summary = json.loads((run_dir / "summary.json").read_text())
        rows.append({"simulations": int(config["simulations"]), **summary})
    rows.sort(key=lambda row: row["simulations"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "search_scaling.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    positions = list(range(len(rows)))
    elo = [row["elo_difference"] for row in rows]
    lower = [row["elo_difference"] - row["elo_low"] for row in rows]
    upper = [row["elo_high"] - row["elo_difference"] for row in rows]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.errorbar(positions, elo, yerr=[lower, upper], marker="o", capsize=4)
    axis.axhline(0, color="0.6", linewidth=1)
    axis.set_xticks(positions, [str(row["simulations"]) for row in rows])
    axis.set_xlabel("MCTS simulations per move (0 = raw policy)")
    axis.set_ylabel("Elo difference vs fixed opponent")
    axis.set_title("Search improves playing strength")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "elo_vs_simulations.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
