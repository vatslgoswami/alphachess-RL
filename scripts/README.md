# scripts

Command-line entry points, one per task (e.g. `solve_endgame.py`, `build_lichess_dataset.py`, `train_supervised.py`, `run_match.py`). Keep the logic in `alphachess/` and keep scripts thin: parse args, load config, call the library.

Current search/evaluation commands:

- `python -m scripts.run_uci`: expose a configured player as a UCI engine;
- `python -m scripts.run_arena`: run resumable colour-swapped matches;
- `python -m scripts.eval_positions`: calculate Stockfish test-set or Lichess puzzle metrics;
- `python -m scripts.plot_search_scaling`: generate the midsem Elo/simulation table and figure.

See [`docs/SEARCH_EVAL.md`](../docs/SEARCH_EVAL.md) for examples and output formats.

Endgame DP and Lichess data commands (Vatsl):

- `python -m scripts.solve_endgame --table all`: solve KQK/KRK/KPK exactly, write tables and figures;
- `python -m scripts.verify_endgame --table all`: Syzygy, Bellman, value iteration and Stockfish checks;
- `python -m scripts.run_approx`: KRK approximation study;
- `python -m scripts.run_rtdp`: value iteration vs retrograde analysis vs RTDP;
- `python -m scripts.run_endgame_mcts`: MCTS vs simulations on the frozen endgame suite;
- `python -m scripts.build_lichess_dataset --month 2024-01 --target 2000000`: raw Lichess shards;
- `python -m scripts.encode_lichess_dataset --month 2024-01`: encoded training shards.

See [`docs/Vatsl_Phase1_Status.md`](../docs/Vatsl_Phase1_Status.md) and
[`reports/midsem/endgames.md`](../reports/midsem/endgames.md).
