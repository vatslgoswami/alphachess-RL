"""Evaluate top-move agreement/CPL or Lichess puzzle solve rate."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import chess
import chess.engine

from alphachess.common.config import load_config
from alphachess.eval.factory import load_evaluator
from alphachess.eval.metrics import evaluate_lichess_puzzles, evaluate_positions
from alphachess.eval.player import MCTSPlayer, PolicyPlayer
from alphachess.eval.stockfish import StockfishAnalyser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--evaluator", default="uniform", help="uniform or module:factory")
    parser.add_argument("--simulations", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--input", required=True)
    parser.add_argument("--format", choices=["fen", "lichess-puzzles"], required=True)
    parser.add_argument("--stockfish-path")
    parser.add_argument("--stockfish-depth", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.simulations = args.simulations if args.simulations is not None else config["mcts"]["simulations"]
    args.batch_size = args.batch_size if args.batch_size is not None else config["mcts"]["batch_size"]
    args.stockfish_path = args.stockfish_path or config["eval"].get("stockfish_path", "stockfish")
    args.stockfish_depth = args.stockfish_depth or config["eval"].get("stockfish_depth", 12)
    evaluator = load_evaluator(args.evaluator)
    player = (
        PolicyPlayer(evaluator)
        if args.simulations == 0
        else MCTSPlayer(evaluator, args.simulations, batch_size=args.batch_size)
    )
    if args.format == "lichess-puzzles":
        with Path(args.input).open(newline="", encoding="utf-8") as handle:
            summary = evaluate_lichess_puzzles(player, csv.DictReader(handle), limit=args.limit)
    else:
        lines = [line.strip() for line in Path(args.input).read_text().splitlines()]
        boards = [chess.Board(line) for line in lines if line and not line.startswith("#")]
        if args.limit is not None:
            boards = boards[: args.limit]
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
        try:
            summary = evaluate_positions(
                player,
                boards,
                StockfishAnalyser(engine, depth=args.stockfish_depth),
            )
        finally:
            engine.quit()
    Path(args.output).write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
