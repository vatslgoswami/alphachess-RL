"""Approximation study on KRK: value models vs the exact table (T6).

    python -m scripts.run_approx                # full run (~10-15 min on 2 CPU threads)
    python -m scripts.run_approx --quick        # tiny smoke run

Outputs: ``reports/midsem/results/approx_metrics.csv`` (overall + per-DTM-bucket metrics),
``approx_fvi_rounds.csv``, and figures ``approx_error_vs_size.png`` / ``approx_error_vs_dtm.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from alphachess.common.config import load_config
from alphachess.endgames import approx as A
from alphachess.endgames import plotting

SIZES = [("linear", None), ("mlp-64", 64), ("mlp-256", 256), ("mlp-1024", 1024)]
METRICS = ["mse", "rel_err", "sign_acc", "optimal_move", "keeps_result", "mate_rate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--fvi-rounds", type=int, default=60)
    parser.add_argument("--fvi-epochs", type=int, default=2)
    parser.add_argument("--fvi-hidden", type=int, default=256)
    parser.add_argument("--mate-starts", type=int, default=500)
    parser.add_argument("--max-moves", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--quick", action="store_true", help="2 epochs, 3 FVI rounds")
    parser.add_argument("--report-dir", default="reports/midsem")
    return parser.parse_args()


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    import torch

    args = parse_args()
    if args.quick:
        args.epochs, args.fvi_rounds, args.fvi_epochs = 2, 3, 1
    torch.set_num_threads(args.threads)
    gamma = load_config()["endgames"].get("approx_gamma", 0.9)
    data = A.load_krk(gamma=gamma, seed=args.seed)
    starts = A.mate_start_positions(data, args.mate_starts, args.seed)
    hanging = data.inputs["features"][:, 8] > 0  # rook attacked and undefended
    print(f"KRK: {len(data.train)} train / {len(data.test)} test positions, gamma={gamma}")

    runs = []
    for inputs in ("onehot", "features"):
        for label, hidden in SIZES:
            model, secs = A.timed(A.train_supervised, data, inputs, hidden, args.epochs, args.seed)
            runs.append((label, inputs, model, secs, "exact labels"))
            print(f"{label:>9} {inputs:>8}: trained in {secs:.0f} s")
    fvi_log: list[dict] = []
    model, secs = A.timed(
        A.train_fitted_vi,
        data,
        "onehot",
        args.fvi_hidden,
        args.fvi_rounds,
        args.fvi_epochs,
        args.seed,
        log=fvi_log,
    )
    runs.append((f"fvi-{args.fvi_hidden}", "onehot", model, secs, "bootstrapped (no labels)"))
    print(f"fitted VI: trained in {secs:.0f} s")

    rows, curves = [], {}
    for label, inputs, model, secs, targets in runs:
        pred = A.predict(model, data.inputs[inputs])
        res = A.evaluate(data, pred, starts, args.max_moves)
        test = data.test
        err2 = (pred[test] - data.value[test]) ** 2
        row = {
            "model": label,
            "inputs": inputs,
            "targets": targets,
            "params": n_params(model),
            "train_seconds": round(secs, 1),
            **{m: round(res[m], 5) for m in METRICS},
            "mse_rook_hanging": round(float(err2[hanging[test]].mean()), 5),
            "mse_rook_safe": round(float(err2[~hanging[test]].mean()), 5),
        }
        for metric in METRICS:
            for bucket, value in res["by_bucket"][metric].items():
                row[f"{metric}[{bucket}]"] = None if np.isnan(value) else round(value, 5)
        rows.append(row)
        curves[(label, inputs)] = res["by_bucket"]
        print(json.dumps({k: row[k] for k in ["model", "inputs", *METRICS]}))

    report = Path(args.report_dir)
    (report / "results").mkdir(parents=True, exist_ok=True)
    with (report / "results" / "approx_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (report / "results" / "approx_fvi_rounds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "test_mse"])
        writer.writeheader()
        writer.writerows(fvi_log)
    plot_size(rows, report / "figures" / "approx_error_vs_size.png", args)
    plot_dtm(curves, report / "figures" / "approx_error_vs_dtm.png")
    meta = {**vars(args), "gamma": gamma, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (report / "results" / "approx_run.json").write_text(json.dumps(meta, indent=2) + "\n")


def plot_size(rows: list[dict], path: Path, args) -> None:
    fig, axes = plotting.setup(figsize=(11.0, 3.8), ncols=3)
    labels = [label for label, _ in SIZES]
    x = np.arange(len(labels))
    fvi = next(r for r in rows if r["model"].startswith("fvi"))
    panels = [
        ("mse", "test MSE of the value (log scale)", True),
        ("optimal_move", "optimal-move rate, 1-move lookahead", False),
        ("mate_rate", f"mate conversion vs perfect defence (≤{args.max_moves} moves)", False),
    ]
    for ax, (metric, title, log) in zip(axes, panels, strict=True):
        for color, inputs, name in (
            (0, "onehot", "one-hot squares"),
            (1, "features", "hand features"),
        ):
            ys = [
                next(r[metric] for r in rows if r["model"] == m and r["inputs"] == inputs)
                for m in labels
            ]
            ax.plot(x, ys, marker="o", color=plotting.SERIES[color], label=name)
        ax.axhline(
            fvi[metric],
            color=plotting.SERIES[2],
            linestyle="--",
            linewidth=1.5,
            label=f"{fvi['model']} (one-hot, no labels)",
        )
        ax.set_xticks(x, labels)
        ax.set_title(title, fontsize=10)
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1.02)
    axes[0].legend(fontsize=8)
    fig.suptitle("KRK: approximation error vs model size", color=plotting.TEXT)
    plotting.save(fig, path)


def plot_dtm(curves: dict, path: Path) -> None:
    buckets = [b[2] for b in A.BUCKETS] + ["draw"]
    fig, axes = plotting.setup(figsize=(14.0, 3.8), ncols=4)
    shown = [
        ("linear", "features"),
        ("mlp-64", "onehot"),
        ("mlp-1024", "onehot"),
        ("mlp-1024", "features"),
    ]
    shown += [key for key in curves if key[0].startswith("fvi")]
    panels = [
        ("mse", "test MSE (log scale)", True),
        ("rel_err", "relative error |V̂−V|/|V| (log scale)", True),
        ("sign_acc", "win/draw/loss accuracy", False),
        ("optimal_move", "optimal-move rate", False),
    ]
    x = np.arange(len(buckets))
    for ax, (metric, title, log) in zip(axes, panels, strict=True):
        for i, key in enumerate(shown):
            ys = [curves[key][metric][b] for b in buckets]
            ax.plot(x, ys, marker="o", color=plotting.SERIES[i], label=f"{key[0]} ({key[1]})")
        ax.set_xticks(x, buckets)
        ax.set_xlabel("exact DTM (plies)")
        ax.set_title(title, fontsize=10)
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1.02)
    axes[0].legend(fontsize=8)
    fig.suptitle("KRK: where approximation fails, by distance to mate", color=plotting.TEXT)
    plotting.save(fig, path)


if __name__ == "__main__":
    main()
