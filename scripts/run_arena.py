"""Run a resumable paired-opening match against Stockfish or a random baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from alphachess.common.checkpoint import git_commit
from alphachess.common.config import load_config
from alphachess.eval.arena import ResultWriter, run_paired_match
from alphachess.eval.factory import load_evaluator
from alphachess.eval.openings import Opening, load_fens, load_pgn
from alphachess.eval.player import MCTSPlayer, PolicyPlayer, RandomPlayer
from alphachess.eval.rating import summarise_results
from alphachess.eval.stockfish import StockfishPlayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--evaluator", default="uniform", help="uniform or module:factory")
    parser.add_argument("--simulations", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--c-puct", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gumbel", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--openings")
    parser.add_argument("--opening-format", choices=["fen", "pgn"], default="fen")
    parser.add_argument("--opening-plies", type=int)
    parser.add_argument("--opponent", choices=["random", "stockfish"], default="random")
    parser.add_argument("--stockfish-path")
    parser.add_argument("--stockfish-elo", type=int)
    parser.add_argument("--stockfish-depth", type=int)
    parser.add_argument("--max-plies", type=int)
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def _load_openings(args: argparse.Namespace) -> list[Opening]:
    if not args.openings:
        import chess

        return [Opening("startpos", chess.STARTING_FEN)]
    if args.opening_format == "pgn":
        return load_pgn(args.openings, plies=args.opening_plies)
    return load_fens(args.openings)


def _read_all_pairs(path: Path) -> list[tuple[float, float]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["opening_id"], []).append(float(row["result_for_candidate"]))
    return [tuple(values) for values in grouped.values() if len(values) == 2]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    mcts_config = config["mcts"]
    eval_config = config["eval"]
    args.simulations = (
        args.simulations if args.simulations is not None else mcts_config["simulations"]
    )
    args.batch_size = args.batch_size if args.batch_size is not None else mcts_config["batch_size"]
    args.c_puct = args.c_puct if args.c_puct is not None else mcts_config["c_puct"]
    args.seed = args.seed if args.seed is not None else eval_config.get("seed", 0)
    args.gumbel = args.gumbel if args.gumbel is not None else mcts_config.get("gumbel", False)
    args.stockfish_path = args.stockfish_path or eval_config.get("stockfish_path", "stockfish")
    args.stockfish_depth = args.stockfish_depth or eval_config.get("stockfish_depth", 8)
    args.max_plies = args.max_plies or eval_config.get("max_plies", 300)
    evaluator = load_evaluator(args.evaluator)
    candidate = (
        PolicyPlayer(evaluator)
        if args.simulations == 0
        else MCTSPlayer(
            evaluator,
            args.simulations,
            c_puct=args.c_puct,
            batch_size=args.batch_size,
            seed=args.seed,
            use_gumbel=args.gumbel,
        )
    )
    opponent = (
        RandomPlayer(args.seed)
        if args.opponent == "random"
        else StockfishPlayer(
            args.stockfish_path,
            elo=args.stockfish_elo,
            depth=args.stockfish_depth,
        )
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = vars(args).copy()
    metadata["git_commit"] = git_commit()
    if args.openings:
        metadata["opening_sha256"] = hashlib.sha256(Path(args.openings).read_bytes()).hexdigest()
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    try:
        writer = ResultWriter(run_dir)
        run_paired_match(
            candidate,
            opponent,
            _load_openings(args),
            writer,
            max_plies=args.max_plies,
        )
        pairs = _read_all_pairs(writer.jsonl_path)
        if pairs:
            summary = summarise_results(pairs, seed=args.seed)
            (run_dir / "summary.json").write_text(
                json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n"
            )
    finally:
        close = getattr(opponent, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    main()
