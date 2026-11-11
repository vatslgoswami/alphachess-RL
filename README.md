# AlphaChess

How much does self-play reinforcement learning improve a chess agent that has already learned from human games? AlphaChess is a small-scale AlphaZero-style project built to investigate that question. It is an RL course project by Samyak, Vatsl and Agriya.

## The experiment

The starting point is a policy-value network trained on positions from Lichess games. Given a board, the network predicts both a move and the likely outcome. Monte Carlo tree search (MCTS) uses those predictions to explore possible continuations before choosing a move.

We plan to improve the supervised agent through self-play: search generates moves for games against itself, and the resulting positions and outcomes become training examples. We can then compare the supervised and self-play agents in matches and estimate the change in playing strength. Small endgames provide a separate setting for studying value functions and search against exact solutions.

The project is in progress. The repository includes the Lichess data pipeline, supervised training, search and evaluation tools, and endgame solvers. The self-play and replay components are planned; the comparison and its results are not yet available.

## Setup

Python 3.10 or later is required. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For the neural network and supervised trainer, install PyTorch with `pip install -e ".[dev,torch]"`. Stockfish is needed for evaluations against Stockfish; install it separately and make sure `stockfish` is on your PATH.

## Try it

Run a short match between MCTS and a random player:

```bash
python -m scripts.run_arena --evaluator uniform --simulations 16 \
  --opponent random --max-plies 40 --run-dir runs/smoke-sim16
```

This uses a uniform evaluator, so it exercises search without a trained model. The games and summary are saved in `runs/smoke-sim16/`. For UCI usage, trained evaluators and Stockfish matches, see [search and evaluation](docs/SEARCH_EVAL.md).

Run the fast tests with:

```bash
pytest
```

## Project files

- `alphachess/lichess/`, `encoding/`, `model/` and `train/`: prepare positions and train the policy-value network.
- `alphachess/mcts/` and `eval/`: search, play matches and measure performance.
- `alphachess/endgames/`: solve and investigate small endgames.
- `scripts/`: entry points for data preparation, training and experiments.
- `config.yaml`: shared experiment settings; datasets and run outputs live locally in `data/` and `runs/`.

For more detail, see the explainers on [encoding and training](docs/Samyak_Encoding_Network_Training_Explainer.md), [search and evaluation](docs/Agriya_Phase1_Search_Eval_Explainer.md), and [endgame methods](docs/Vatsl_Endgame_DP_Primer.html).
