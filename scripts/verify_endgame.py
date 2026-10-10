"""Verify solved tables: Syzygy WDL on every position, Bellman consistency, value iteration,
and a Stockfish DTM spot-check.

    python -m scripts.verify_endgame --table all
    python -m scripts.verify_endgame --table KRK --stockfish-samples 1000

Checks that need missing files or binaries (Syzygy, Stockfish) are reported as skipped. Results
are appended to ``reports/midsem/results/verification.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from alphachess.common.config import load_config, repo_path
from alphachess.endgames import verify
from alphachess.endgames.graph import build_graph
from alphachess.endgames.tables import DEPENDENCIES, load_or_solve

ORDER = ["KQK", "KRK", "KPK"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", default="all", choices=[*ORDER, "all"])
    parser.add_argument("--stockfish-samples", type=int, default=1000)
    parser.add_argument("--stockfish-nodes", type=int, default=300_000)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--output", default="reports/midsem/results/verification.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    gamma = args.gamma or cfg["endgames"].get("approx_gamma", 0.9)
    tb_dir = repo_path(cfg["paths"]["tablebase_dir"])
    rows = []
    for name in ORDER if args.table == "all" else [args.table]:
        table = load_or_solve(name)
        deps = {dep: load_or_solve(dep) for dep in DEPENDENCIES[name]}
        graph = build_graph(name, (lambda d, s, deps=deps: deps[d].lookup(s)) if deps else None)
        checks = [
            verify.check_bellman(table, graph),
            verify.check_value_iteration(table, graph, gamma),
        ]
        if any(Path(tb_dir).glob(f"{table.spec.syzygy_name}.rtbw")):
            checks.append(verify.check_syzygy(table, str(tb_dir)))
        else:
            print(f"{name}: Syzygy files not found in {tb_dir}; skipping the Syzygy check")
        engine = verify.stockfish_path(cfg["eval"].get("stockfish_path", "stockfish"))
        if engine and args.stockfish_samples > 0:
            checks.append(
                verify.check_stockfish_dtm(
                    table, args.stockfish_samples, engine_path=engine, nodes=args.stockfish_nodes
                )
            )
        elif args.stockfish_samples > 0:
            print(f"{name}: Stockfish not found; skipping the DTM spot-check")
        for check in checks:
            print(f"{name} {check.name}: {check.agree}/{check.checked} ok={check.ok}")
            rows.append(
                {
                    "table": name,
                    "check": check.name,
                    "checked": check.checked,
                    "agree": check.agree,
                    "ok": check.ok,
                    "details": json.dumps(check.details),
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if output.exists():
        with output.open() as handle:
            existing = list(csv.DictReader(handle))
    keys = {(r["table"], r["check"]) for r in rows}
    merged = [r for r in existing if (r["table"], r["check"]) not in keys] + rows
    merged.sort(key=lambda r: (ORDER.index(r["table"]), r["check"]))
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(merged)
    failed = [r for r in rows if str(r["ok"]) != "True"]
    if failed:
        raise SystemExit(f"{len(failed)} check(s) failed")


if __name__ == "__main__":
    main()
