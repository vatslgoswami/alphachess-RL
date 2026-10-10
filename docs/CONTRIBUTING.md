# Contributing

## Branches and PRs

- `main` is always green. Don't push to it directly.
- Branch names: `<name>/<topic>`, e.g. `vatsl/endgame-dp`, `samyak/encoding`, `agriya/mcts`.
- Open a PR into `main`, and get one teammate to glance at it if it touches `docs/INTERFACES.md` or `alphachess/common/`.
- CI (`.github/workflows/tests.yml`) runs `ruff check` and `pytest` on every PR.

## Code

- Put library code in `alphachess/<component>/` and runnable entry points in `scripts/`.
- Read settings from `config.yaml` via `alphachess.common.config.load_config()`. Don't hard-code paths.
- Tests go in `tests/test_<component>.py`. Mark anything over ~10 s with `@pytest.mark.slow`; those tests run only with `pytest -m slow`.
- Must-have tests: encoding round-trips, and MCTS backup signs. Silent bugs hide in those two places.

## Data and runs

- Never commit data, checkpoints or tablebases. `data/` and `runs/` are git-ignored.
- Share large files through Kaggle Datasets (or Drive) and put the link in the relevant README section.
- Log every training/eval run in the shared runs table: config, git commit, Kaggle account, and wall-clock time.

## Notebooks

Strip outputs before committing (`jupyter nbconvert --clear-output --inplace nb.ipynb`). Move anything reusable into `alphachess/`.
