# alpha-chess-rl

A small-scale AlphaZero-style chess agent. We train a policy–value ResNet on human games from Lichess, then improve it with MCTS self-play RL, and measure how much Elo the RL adds.

Team: **Samyak**, **Vatsl**, **Agriya**. RL course project, Monsoon 2026.

## Layout

All code is in one package, `alphachess/`. Each sub-package has one owner, but anyone can contribute.

| Folder | What lives here | Owner |
|---|---|---|
| `alphachess/common/` | Config loading, checkpoint save/load, shared types | everyone |
| `alphachess/encoding/` | Board → planes, move ↔ index (4672), legal-move mask | Samyak |
| `alphachess/model/` | Policy–value ResNet | Samyak |
| `alphachess/train/` | Supervised trainer (Phase 1), RL trainer (Phase 2) | Samyak |
| `alphachess/endgames/` | Exact DP solver (KQK, KRK, KPK), approximation experiments, tablebase probing | Vatsl |
| `alphachess/lichess/` | Lichess data pipeline: sample positions, attach move + Stockfish eval, write shards | Vatsl |
| `alphachess/replay/` | Replay buffer of (s, π, z) (Phase 2) | Vatsl |
| `alphachess/selfplay/` | Self-play workers, resignation, tablebase adjudication (Phase 2) | Vatsl |
| `alphachess/mcts/` | PUCT / Gumbel MCTS with batched network calls | Agriya |
| `alphachess/eval/` | UCI wrapper, match harness vs Stockfish, Elo with error bars | Agriya |
| `scripts/` | Command-line entry points (`python -m scripts.<name>` or `python scripts/<name>.py`) | — |
| `tests/` | `pytest` tests, one file per sub-package | — |
| `notebooks/` | Exploration and plotting notebooks (strip outputs before committing) | — |
| `reports/` | Midsem and end-term write-ups and figures | — |
| `docs/` | Interfaces and contributing guide | — |
| `data/`, `runs/` | Local datasets, tablebases, checkpoints, logs. **Git-ignored.** | — |

The interfaces between components (encoding format, checkpoint format, replay record) are in [docs/INTERFACES.md](docs/INTERFACES.md). Code against those so all three of us can work in parallel.

Agriya's Phase 1 search and evaluation implementation is documented in
[`docs/SEARCH_EVAL.md`](docs/SEARCH_EVAL.md). It includes batched PUCT, the UCI wrapper, paired
matches, Stockfish evaluation, Elo intervals, puzzle/position metrics, and the midsem plotting
workflow. A conceptual walkthrough for teammates and teaching staff is in
[`docs/PHASE1_SEARCH_EVAL_EXPLAINER.md`](docs/PHASE1_SEARCH_EVAL_EXPLAINER.md). The original
implementation plan is in
[`docs/AGRIYA_PHASE1_PLAN.md`](docs/AGRIYA_PHASE1_PLAN.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

PyTorch is an optional extra (`pip install -e ".[dev,torch]"`) because the endgame and data code doesn't need it.

Stockfish is needed for the data pipeline and the evaluation harness: `brew install stockfish` (macOS) or `apt install stockfish` (Linux / Kaggle).

## Config

Shared settings are in [`config.yaml`](config.yaml). Load them with:

```python
from alphachess.common.config import load_config
cfg = load_config()            # config.yaml
cfg = load_config("my.yaml")   # or override file
```

## Workflow

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md). In short: branch per feature, open a PR into `main`, and CI must pass.
