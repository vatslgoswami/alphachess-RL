"""Search improves the policy, with exact ground truth: MCTS vs simulations on the KRK suite (T7).

    python -m scripts.run_endgame_mcts                 # build the suite if missing, run, plot
    python -m scripts.run_endgame_mcts --rebuild-suite

Agriya's ``MCTSPlayer`` with ``uniform_evaluator`` (no knowledge, values 0) is scored with
``evaluate_positions`` and the exact reference analyser. A ``DPEvaluator`` run and the
``TablebasePlayer`` are sanity checks (both should be ~100%).

Writes ``reports/midsem/results/mcts_vs_sims_krk.csv`` and
``reports/midsem/figures/mcts_vs_sims_krk.png``.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from alphachess.endgames import hooks, plotting, probe
from alphachess.eval.metrics import evaluate_positions
from alphachess.eval.player import MCTSPlayer, RandomPlayer
from alphachess.mcts import uniform_evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", default="KRK", choices=["KQK", "KRK", "KPK"])
    parser.add_argument("--simulations", type=int, nargs="+", default=[1, 16, 64, 256, 1024])
    parser.add_argument("--rebuild-suite", action="store_true")
    parser.add_argument("--suite", default=None, help="defaults to data/endgames/suite.fen")
    parser.add_argument("--value-hidden", type=int, default=64, help="0 skips the learned run")
    parser.add_argument("--value-epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-dir", default="reports/midsem")
    return parser.parse_args()


def score(player, boards, label: str, sims: int) -> dict:
    recorder = hooks.RecordingPlayer(player)
    started = time.perf_counter()
    summary = evaluate_positions(recorder, boards, hooks.exact_reference_analyser)
    seconds = time.perf_counter() - started
    rates = hooks.optimal_rates(recorder.moves)
    short = [m for m in recorder.moves if abs(probe.probe_dtm(_board(m[0]))) <= 5]
    long = [m for m in recorder.moves if abs(probe.probe_dtm(_board(m[0]))) > 5]
    return {
        "player": label,
        "simulations": sims,
        "positions": summary.positions,
        "top_move_agreement": round(summary.top_move_agreement, 4),
        "mean_score_loss": round(summary.mean_centipawn_loss, 2),
        "optimal_move_rate": round(rates["optimal_move_rate"], 4),
        "keeps_result_rate": round(rates["keeps_result_rate"], 4),
        "optimal_dtm_le5": round(hooks.optimal_rates(short)["optimal_move_rate"], 4),
        "optimal_dtm_gt5": round(hooks.optimal_rates(long)["optimal_move_rate"], 4),
        "seconds": round(seconds, 1),
    }


def learned_evaluator(hidden: int, epochs: int, seed: int):
    """Train the T6 MLP on one-hot KRK inputs (80% of positions) and wrap it for MCTS."""
    import torch

    from alphachess.endgames import approx

    torch.set_num_threads(2)
    data = approx.load_krk(seed=seed)
    model = approx.train_supervised(data, "onehot", hidden, epochs, seed)
    pred = approx.predict(model, data.inputs["onehot"])
    return approx.SlotValueEvaluator(approx.values_by_slot(data, pred))


def _board(fen: str):
    import chess

    return chess.Board(fen)


def main() -> None:
    args = parse_args()
    path = Path(args.suite) if args.suite else hooks.default_suite_path()
    if args.rebuild_suite or not path.exists():
        hooks.write_suite(hooks.build_suite(seed=args.seed), path)
        print(f"wrote {path}")
    boards = hooks.load_suite(path)[args.table]
    print(f"{args.table}: {len(boards)} suite positions")

    rows = [score(RandomPlayer(seed=args.seed), boards, "random", 0)]
    for sims in args.simulations:
        player = MCTSPlayer(uniform_evaluator, sims, seed=args.seed)
        rows.append(score(player, boards, "MCTS, uniform evaluator", sims))
        print(rows[-1])
    if args.table == "KRK" and args.value_hidden:
        evaluator = learned_evaluator(args.value_hidden, args.value_epochs, args.seed)
        label = f"MCTS, learned value (MLP-{args.value_hidden})"
        for sims in args.simulations:
            rows.append(score(MCTSPlayer(evaluator, sims, seed=args.seed), boards, label, sims))
            print(rows[-1])
    rows.append(
        score(
            MCTSPlayer(hooks.DPEvaluator(gamma=0.9), 16, seed=args.seed),
            boards,
            "MCTS, DPEvaluator (sanity)",
            16,
        )
    )
    rows.append(score(hooks.TablebasePlayer(RandomPlayer()), boards, "TablebasePlayer", 0))
    print(rows[-2], rows[-1], sep="\n")

    report = Path(args.report_dir)
    name = f"mcts_vs_sims_{args.table.lower()}"
    (report / "results").mkdir(parents=True, exist_ok=True)
    with (report / "results" / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.table, len(boards), report / "figures" / f"{name}.png")


def plot(rows: list[dict], table: str, n: int, path: Path) -> None:
    fig, ax = plotting.setup(figsize=(7.4, 4.4))
    families = [
        ("MCTS, uniform evaluator", "uniform evaluator (no knowledge)", 0),
        ("MCTS, learned value", "learned value net (T6 MLP), uniform priors", 1),
    ]
    sims = None
    for prefix, label, color in families:
        part = [r for r in rows if r["player"].startswith(prefix)]
        if not part:
            continue
        sims = np.array([r["simulations"] for r in part])
        ax.plot(
            sims,
            [r["optimal_move_rate"] for r in part],
            marker="o",
            color=plotting.SERIES[color],
            label=label,
        )
    random_row = rows[0]
    ax.axhline(
        random_row["optimal_move_rate"], color=plotting.TEXT_SECONDARY, linewidth=1, linestyle="--"
    )
    ax.text(
        sims[-1],
        random_row["optimal_move_rate"] - 0.05,
        "random mover",
        ha="right",
        color=plotting.TEXT_SECONDARY,
        fontsize=8,
    )
    ax.set_xscale("log", base=4)
    ax.set_xticks(sims, [str(s) for s in sims])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("MCTS simulations per move")
    ax.set_ylabel("DTM-optimal move rate")
    ax.set_title(
        f"{table}: search improves the policy ({n} suite positions, exact reference)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left")
    plotting.save(fig, path)


if __name__ == "__main__":
    main()
