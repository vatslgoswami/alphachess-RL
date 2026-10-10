# scripts

Command-line entry points, one per task (e.g. `solve_endgame.py`, `build_lichess_dataset.py`, `train_supervised.py`, `run_match.py`). Keep the logic in `alphachess/` and keep scripts thin: parse args, load config, call the library.
