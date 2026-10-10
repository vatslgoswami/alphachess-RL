"""Value iteration vs retrograde analysis vs RTDP on KRK (T5).

    python -m scripts.run_rtdp                   # ~5 min
    python -m scripts.run_rtdp --starts 1000 --naive-passes 50

Writes ``reports/midsem/results/rtdp_results.{csv,md}``, ``rtdp_curve.csv`` and the figure
``reports/midsem/figures/rtdp_touched_vs_trials.png``.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from alphachess.common.config import load_config
from alphachess.endgames import plotting
from alphachess.endgames.approx import load_krk
from alphachess.endgames.index import decode
from alphachess.endgames.rtdp import RTDP, BoundedRTDP
from alphachess.endgames.solve import solve
from alphachess.endgames.vi import value_iteration, values_to_wdl_dtm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--starts", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--naive-passes", type=int, default=50)
    parser.add_argument("--max-passes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-dir", default="reports/midsem")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gamma = load_config()["endgames"].get("approx_gamma", 0.9)
    data = load_krk(gamma=gamma, seed=args.seed)
    g = data.graph
    n = g.n
    rows = []

    started = time.perf_counter()
    sol = solve(g)
    retro_seconds = time.perf_counter() - started
    decisive = np.flatnonzero(sol.wdl != 0)
    pred_visits = int(np.diff(g.pred_ptr)[decisive].sum())
    rows.append(
        {
            "method": "retrograde analysis",
            "starts": "all",
            "trials_or_sweeps": f"{sol.levels} levels",
            "backups": pred_visits,
            "positions_touched": n,
            "exact_entries": n,
            "wrong_entries": 0,
            "seconds": round(retro_seconds, 2),
            "converged": True,
        }
    )

    vi = value_iteration(g, gamma)
    wdl, dtm = values_to_wdl_dtm(vi.values, gamma)
    exact = int(((wdl == sol.wdl[:n]) & (dtm == sol.dtm[:n])).sum())
    rows.append(
        {
            "method": "synchronous value iteration",
            "starts": "all",
            "trials_or_sweeps": f"{vi.sweeps} sweeps",
            "backups": vi.backups,
            "positions_touched": n,
            "exact_entries": exact,
            "wrong_entries": n - exact,
            "seconds": round(vi.seconds, 2),
            "converged": exact == n,
        }
    )

    stm = decode(g.spec, g.slots)[0]
    rng = np.random.default_rng(args.seed)
    pool = rng.permutation(np.flatnonzero(stm == 0))
    curves = {}

    naive_starts = pool[: max(args.starts)]
    naive = RTDP(g, data.value, data.best_edge, gamma, init=0.0)
    res = naive.run(naive_starts, data.dtm, max_passes=args.naive_passes)
    rows.append(_rtdp_row("RTDP, single value, V0 = 0 (primer)", len(naive_starts), res))
    curves[f"RTDP V0=0, {len(naive_starts)} starts"] = res.curve
    print(rows[-1])

    for count in args.starts:
        bounded = BoundedRTDP(g, data.value, data.best_edge, gamma)
        res = bounded.run(pool[:count], data.dtm, max_passes=args.max_passes)
        rows.append(_rtdp_row("RTDP with bounds (U0 = +1, L0 = -1)", count, res))
        curves[f"bounded RTDP, {count} starts"] = res.curve
        print(rows[-1])

    report = Path(args.report_dir)
    (report / "results").mkdir(parents=True, exist_ok=True)
    with (report / "results" / "rtdp_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (report / "results" / "rtdp_results.md").open("w") as handle:
        handle.write(
            f"KRK, {n:,} positions, γ = {gamma}. Backups for retrograde analysis count "
            "predecessor edges walked.\n\n"
        )
        handle.write("| " + " | ".join(rows[0]) + " |\n")
        handle.write("|" + "---|" * len(rows[0]) + "\n")
        for row in rows:
            handle.write(
                "| "
                + " | ".join(
                    f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else str(v)
                    for v in row.values()
                )
                + " |\n"
            )
    with (report / "results" / "rtdp_curve.csv").open("w", newline="") as handle:
        fields = ["run", "pass", "trials", "backups", "touched", "starts_solved", "wrong_entries"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run, curve in curves.items():
            for point in curve:
                writer.writerow({"run": run, **point})
    plot(curves, n, vi.backups, report / "figures" / "rtdp_touched_vs_trials.png")


def _rtdp_row(method: str, starts: int, res) -> dict:
    return {
        "method": method,
        "starts": starts,
        "trials_or_sweeps": f"{res.trials:,} trials",
        "backups": res.backups,
        "positions_touched": res.touched,
        "exact_entries": res.touched - res.wrong_entries,
        "wrong_entries": res.wrong_entries,
        "seconds": round(res.seconds, 2),
        "converged": res.converged,
    }


def plot(curves: dict, n: int, vi_backups: int, path: Path) -> None:
    fig, (ax, bx) = plotting.setup(figsize=(11.0, 4.0), ncols=2)
    for i, (name, curve) in enumerate(curves.items()):
        trials = [p["trials"] for p in curve]
        color = plotting.SERIES[i]
        ax.plot(trials, [p["touched"] for p in curve], color=color, label=name)
        ax.plot(
            trials,
            [p["touched"] - p["wrong_entries"] for p in curve],
            color=color,
            linestyle=":",
            linewidth=1.5,
        )
        starts = int(name.split(", ")[1].split()[0])
        bx.plot(trials, [100 * p["starts_solved"] / starts for p in curve], color=color, label=name)
    ax.axhline(n, color=plotting.TEXT_SECONDARY, linewidth=1)
    ax.text(
        0.98,
        n * 0.97,
        f"all {n:,} positions",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="top",
        color=plotting.TEXT_SECONDARY,
        fontsize=8,
    )
    ax.set_xscale("log")
    ax.set_xlabel("trials")
    ax.set_ylabel("positions")
    ax.set_title("positions touched (solid) and exact (dotted)")
    bx.set_xscale("log")
    bx.set_ylim(0, 102)
    bx.set_xlabel("trials")
    bx.set_ylabel("% of start positions")
    bx.set_title("start positions solved (greedy White play is optimal)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8)
    fig.suptitle(f"RTDP on KRK (value iteration: {vi_backups:,} backups)", color=plotting.TEXT)
    plotting.save(fig, path, rect=(0, 0.07, 1, 1))


if __name__ == "__main__":
    main()
