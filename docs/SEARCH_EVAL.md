# Search and evaluation

Agriya's Phase 1 implementation provides batched PUCT search, a low-budget Gumbel-root variant,
a UCI engine, paired-opening matches, Stockfish integration, Elo intervals, position metrics, and
the midsem search-scaling plot.

The implementation deliberately depends only on the evaluator contract in
[`INTERFACES.md`](INTERFACES.md). The policy-value model can be connected after its interface is
final without changing MCTS, UCI, or the arena.

## Quick start

Install the project and run the fast test suite:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest
```

Check the UCI handshake with the built-in uniform evaluator:

```bash
printf 'uci\nisready\nposition startpos\ngo nodes 16\nquit\n' | \
  python -m scripts.run_uci --evaluator uniform
```

Run a two-game colour-swapped smoke match:

```bash
python -m scripts.run_arena \
  --evaluator uniform \
  --simulations 16 \
  --opponent random \
  --max-plies 40 \
  --run-dir runs/smoke-sim16
```

The smoke run creates:

```text
runs/smoke-sim16/
├── config.json     resolved arguments, Git commit, opening hash when applicable
├── games.jsonl     one append-only record per game
├── games.pgn       replayable games
└── summary.json    W/D/L, score, Elo difference, and paired-bootstrap interval
```

Rerunning the same command skips completed game IDs.

## Modules

```text
alphachess/mcts/
├── node.py       tree statistics in the side-to-move value convention
├── batch.py      in-flight leaf reservation
├── search.py     batched PUCT, terminal handling, root noise, diagnostics
└── gumbel.py     seeded Gumbel perturbation of root policy logits

alphachess/eval/
├── evaluator.py PyTorch adapter with injected encoding functions
├── factory.py   CLI evaluator-factory loading
├── player.py    policy, MCTS, and seeded random players
├── uci.py       UCI protocol loop
├── arena.py     games, colour-swapped pairs, resume-safe output
├── openings.py  FEN and PGN opening suites
├── stockfish.py fixed-strength opponent and reference analyser
├── rating.py    Elo conversion and paired bootstrap
└── metrics.py   top-move agreement, centipawn loss, and Lichess puzzles
```

## Evaluator contract

An evaluator receives a list of boards and returns legal priors plus values:

```python
def evaluator(boards):
    # priors[i] maps legal chess.Move objects to probabilities.
    # values[i] is in [-1, 1] from boards[i].turn's perspective.
    return priors, values
```

To connect the supervised network without changing the command-line tools, expose a zero-argument
factory:

```python
# my_integration.py
def make_evaluator():
    model = ...
    return TorchEvaluator(model, encode_board, move_to_index, device="cuda")
```

Then pass `--evaluator my_integration:make_evaluator` to `run_uci`, `run_arena`, or
`eval_positions`. Until the shared model boundary lands, `--evaluator uniform` keeps all search
and evaluation paths runnable.

## MCTS behavior

- Values are stored from the side to move at each node and negated once per backed-up edge.
- Root expansion is not counted as a simulation.
- Evaluator priors are restricted to legal moves and normalized; invalid/zero priors fall back to
  uniform legal priors.
- Batched search reserves in-flight paths, evaluates each unique leaf once, and still backs up
  every requested simulation.
- `0` simulations is handled by `PolicyPlayer`; positive budgets use `MCTSPlayer`.
- Evaluation uses no Dirichlet noise. Root noise is available for later self-play.
- `gumbel_search` is the Phase 1 low-budget root-perturbation variant, not the full sequential-
  halving Gumbel AlphaZero algorithm.

## UCI

The wrapper supports `uci`, `isready`, `setoption`, `ucinewgame`, `position startpos`, `position
fen`, `go`, `go nodes`, `stop`, and `quit`. `go nodes N` maps to `N` completed MCTS simulations.
The reproducible evaluation protocol uses node budgets rather than wall-clock limits.

Example tournament engine command:

```bash
python -m scripts.run_uci \
  --evaluator my_integration:make_evaluator \
  --simulations 64 \
  --batch-size 16
```

Only UCI messages are written to stdout. A terminal result is returned as `bestmove 0000`.

## Paired matches and Stockfish

FEN opening files contain one complete FEN per non-comment line. PGN suites are also supported.
Each opening is played twice with colours swapped.

```bash
python -m scripts.run_arena \
  --evaluator my_integration:make_evaluator \
  --simulations 64 \
  --openings data/openings/midsem.fen \
  --opening-format fen \
  --opponent stockfish \
  --stockfish-path /path/to/stockfish \
  --stockfish-elo 1500 \
  --run-dir runs/sim64-sf1500
```

The Stockfish adapter fixes its thread and hash settings and checks that the installed build
supports `UCI_Elo` when a limited rating is requested. Rating intervals resample whole opening
pairs, since games in a colour-swapped pair are not independent.

## Position and puzzle metrics

For top-move agreement and centipawn loss, provide one FEN per line:

```bash
python -m scripts.eval_positions \
  --format fen \
  --input data/eval/positions.fen \
  --evaluator my_integration:make_evaluator \
  --stockfish-path /path/to/stockfish \
  --output runs/position-metrics.json
```

For the official Lichess puzzle CSV, the evaluator checks the complete recorded solution line:

```bash
python -m scripts.eval_positions \
  --format lichess-puzzles \
  --input data/lichess/lichess_db_puzzle.csv \
  --limit 1000 \
  --evaluator my_integration:make_evaluator \
  --output runs/puzzle-metrics.json
```

## Midsem search-scaling experiment

Run the same checkpoint, opening suite, Stockfish settings, and seed at budgets `0, 1, 16, 64,
256`. Use a separate run directory for each budget. Then generate the CSV and figure:

```bash
python -m scripts.plot_search_scaling \
  runs/sim0-sf1500 \
  runs/sim1-sf1500 \
  runs/sim16-sf1500 \
  runs/sim64-sf1500 \
  runs/sim256-sf1500 \
  --output-dir reports/midsem/figures
```

This writes `search_scaling.csv` and `elo_vs_simulations.png`. Pilot with 20–40 games per point,
freeze the configuration, and then target the configured 200 games per point.

## Known boundary

The network and encoding implementations are not part of Agriya's track and are still being
developed. `TorchEvaluator` and `module:factory` loading are the intentional integration seam.
Once that boundary is confirmed, add the project-specific factory and its real-checkpoint slow
test; no search, UCI, arena, rating, or plotting API should need to change.
