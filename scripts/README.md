# scripts

Command-line entry points, one per task (e.g. `solve_endgame.py`, `build_lichess_dataset.py`, `train_supervised.py`, `run_match.py`). Keep the logic in `alphachess/` and keep scripts thin: parse args, load config, call the library.

Current search/evaluation commands:

- `python -m scripts.run_uci`: expose a configured player as a UCI engine;
- `python -m scripts.run_arena`: run resumable colour-swapped matches;
- `python -m scripts.eval_positions`: calculate Stockfish test-set or Lichess puzzle metrics;
- `python -m scripts.plot_search_scaling`: generate the midsem Elo/simulation table and figure.

See [`docs/SEARCH_EVAL.md`](../docs/SEARCH_EVAL.md) for examples and output formats.
