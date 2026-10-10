"""Launch AlphaChess as a UCI engine."""

from __future__ import annotations

import argparse

from alphachess.common.config import load_config
from alphachess.eval.factory import load_evaluator
from alphachess.eval.uci import UCIEngine, mcts_player_factory, run_uci_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--evaluator", default="uniform", help="uniform or module:factory")
    parser.add_argument("--simulations", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--c-puct", type=float)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_config(args.config)["mcts"]
    simulations = args.simulations if args.simulations is not None else settings["simulations"]
    batch_size = args.batch_size if args.batch_size is not None else settings["batch_size"]
    c_puct = args.c_puct if args.c_puct is not None else settings["c_puct"]
    seed = args.seed if args.seed is not None else settings.get("seed", 0)
    evaluator = load_evaluator(args.evaluator)
    factory = mcts_player_factory(
        evaluator,
        c_puct=c_puct,
        batch_size=batch_size,
    )
    engine = UCIEngine(
        factory(simulations, seed),
        player_factory=factory,
    )
    run_uci_loop(engine)


if __name__ == "__main__":
    main()
