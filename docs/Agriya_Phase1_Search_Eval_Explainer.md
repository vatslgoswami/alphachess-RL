# Phase 1 search and evaluation — explainer

This document explains Agriya's Phase 1 contribution in plain language: what was added, why each
piece exists, how information moves through the system, and how the work supports the midsem
claim that search improves a fixed chess policy.

For exact command options and operational details, see [`SEARCH_EVAL.md`](SEARCH_EVAL.md). For the
shared component contracts, see [`INTERFACES.md`](INTERFACES.md).

## The result in one paragraph

The pushed code turns any compatible policy-value evaluator into a chess-playing agent. The
evaluator proposes promising legal moves and estimates who is winning. Monte Carlo Tree Search
(MCTS) looks ahead from those proposals, the player layer chooses a move, the UCI wrapper exposes
the player to standard chess software, and the evaluation harness measures it in colour-swapped
games against fixed opponents. The same harness also computes Elo uncertainty, compares moves to
Stockfish, evaluates Lichess puzzles, and generates the planned Elo-versus-search-budget figure.

The real supervised network is not hard-coded into this contribution. It plugs in through a
small evaluator contract once the teammate-owned encoder and checkpoint are ready.

## What question does this code answer?

Phase 1 needs to isolate the value of search:

> If the neural-network checkpoint stays fixed, does giving the agent more MCTS simulations make
> it stronger?

The planned comparison is:

| Search budget | Meaning |
|---:|---|
| 0 | Choose directly from the network policy. |
| 1 | Perform one completed MCTS simulation. |
| 16 | Low-compute search. |
| 64 | Default project search budget. |
| 256 | Larger search budget for the scaling curve. |

Everything except the search budget must remain fixed: checkpoint, openings, colours, opponent,
Stockfish settings, random seed, adjudication rules, and hardware class.

## How the pieces fit together

```text
Chess position
      │
      ▼
Board encoder + policy-value network       teammate-owned boundary
      │
      ▼
Evaluator: legal move priors + value       stable integration contract
      │
      ├──────────────► PolicyPlayer        0-simulation baseline
      │
      ▼
Batched PUCT / Gumbel-root MCTS            1/16/64/256 simulations
      │
      ▼
MoveDecision
      │
      ├──────────────► UCI engine ───────► chess tournament tools
      │
      ├──────────────► Arena ────────────► PGNs + per-game JSON
      │                                      │
      │                                      ▼
      │                                 Elo + uncertainty
      │
      └──────────────► Position metrics ─► agreement, CPL, puzzles
                                             │
                                             ▼
                                     midsem tables and plots
```

The UCI engine and the Python arena use the same player objects. That prevents the live/demo
engine from behaving differently from the engine being measured.

## 1. The evaluator contract

MCTS does not know how the neural network is implemented. It calls an evaluator with a batch of
boards:

```python
priors, values = evaluator(boards)
```

For board `i`:

- `priors[i]` maps legal `chess.Move` objects to probabilities;
- `values[i]` is in `[-1, 1]`;
- the value is always from the side-to-move's perspective;
- `+1` means the side to move is winning, `0` means a draw, and `-1` means losing.

[`TorchEvaluator`](../alphachess/eval/evaluator.py) adapts a PyTorch policy-value model to this
contract. The model, board encoder, and move-index function are injected instead of imported from
unfinished teammate modules.

Command-line programs load the real integration through `module:factory`:

```python
# Example only; add this after the shared model boundary is final.
def make_evaluator():
    model = load_the_team_checkpoint()
    return TorchEvaluator(model, encode_board, move_to_index, device="cuda")
```

```bash
python -m scripts.run_uci --evaluator my_integration:make_evaluator
```

Until that factory exists, the uniform evaluator makes every downstream component runnable and
testable without pretending that it is a trained chess player.

## 2. How PUCT search works

The core loop is implemented in [`alphachess/mcts/search.py`](../alphachess/mcts/search.py).

Each simulation performs four steps:

1. **Selection:** walk down the current tree using a PUCT score.
2. **Expansion:** ask the evaluator for priors and a value at an unseen position.
3. **Evaluation:** use a rule-based terminal result or the evaluator's value.
4. **Backup:** propagate the result to the root, negating it after every move.

The selection score balances exploitation and exploration:

```text
score(move) = Q(move) + c_puct × P(move) × sqrt(parent visits + 1)
                                      / (1 + move visits)
```

- `Q` is the backed-up value of the move.
- `P` is the policy prior supplied by the evaluator.
- `c_puct` controls how strongly the search explores less-visited moves.

The implementation stores a child value from the child position's side-to-move perspective. A
parent therefore reads that value as `-child.Q`. This sign convention is subtle: getting it wrong
would make the engine prefer moves that are good for its opponent.

### Terminal positions

Terminal values come from chess rules, not the neural network:

- checkmate is `-1` for the checkmated side to move;
- stalemate and insufficient material are `0`;
- claimable repetition and fifty-move draws are `0`.

A terminal board is never sent to the evaluator.

### Legal priors

Only legal moves may enter the tree. Evaluator priors are filtered and normalized. Non-finite,
negative, missing, or zero-total priors fall back to a uniform distribution over legal moves,
preventing a malformed policy from producing an illegal search tree.

### What counts as a simulation?

The initial evaluation that expands the root is not counted. One simulation means one selected
leaf and one completed backup. The visit counts returned at the root therefore add up exactly to
the requested simulation budget.

## 3. Why the evaluator is batched

A GPU is inefficient when it receives one chess position at a time. The search therefore selects
several pending leaves and evaluates them together.

[`batch.py`](../alphachess/mcts/batch.py) reserves in-flight paths while a batch is being formed.
The temporary reservations discourage every selection from following the same branch. If the
same leaf is still selected more than once, it is evaluated once but backed up once per requested
simulation, preserving the exact simulation count.

Search diagnostics record:

- completed simulations;
- evaluator calls;
- evaluated positions;
- nodes expanded;
- mean evaluator batch size;
- elapsed time and simulations per second.

This lets us distinguish a weak agent from a batching or throughput problem.

## 4. Root exploration modes

Two root-exploration mechanisms are present:

- **Dirichlet noise:** available for later self-play so games do not repeatedly follow the same
  opening. It is disabled during evaluation.
- **Gumbel-root search:** perturbs root policy logits with seeded Gumbel samples for low-budget
  exploration.

The current Gumbel implementation is a reproducible root-perturbation variant. It is not the full
sequential-halving Gumbel AlphaZero algorithm. This limitation is explicit so the report does not
overstate the method used.

## 5. Player layer

[`player.py`](../alphachess/eval/player.py) gives every engine the same interface:

```python
decision = player.select_move(board)
```

A decision contains the move, estimated value, completed simulations, and elapsed search time.

The available players are:

- `PolicyPlayer`: uses the evaluator policy directly; this is the 0-simulation baseline.
- `MCTSPlayer`: uses PUCT or Gumbel-root search at a fixed budget.
- `RandomPlayer`: a seeded legal-move baseline for smoke testing.

The random player is deterministic for a seed and position, making interrupted smoke runs
repeatable.

## 6. UCI engine

[`uci.py`](../alphachess/eval/uci.py) exposes a player using the Universal Chess Interface. This
allows standard chess tournament programs to launch the agent as a subprocess.

Supported commands include:

- `uci` and `isready` for the handshake;
- `setoption` for simulations and seed;
- `ucinewgame`;
- `position startpos` and `position fen`, including subsequent moves;
- `go` and `go nodes N`;
- `stop` and `quit`.

`go nodes N` maps directly to `N` completed simulations. Fixed node budgets are the scientific
evaluation protocol because wall-clock budgets vary with hardware load.

Example smoke handshake:

```bash
printf 'uci\nisready\nposition startpos\ngo nodes 16\nquit\n' | \
  python -m scripts.run_uci --evaluator uniform
```

## 7. Match arena

[`arena.py`](../alphachess/eval/arena.py) plays reproducible games between any two players.

For every opening, the candidate plays twice:

1. candidate as White;
2. candidate as Black.

This colour swap reduces first-move and opening bias. The arena records illegal moves or crashes
as forfeits rather than silently dropping the game.

Every run produces:

```text
runs/<run-name>/
├── config.json     resolved settings, Git commit, opening hash
├── games.jsonl     one machine-readable row per game
├── games.pgn       games for human inspection
└── summary.json    W/D/L, score, Elo, and interval
```

The JSONL file is append-only. If a run is interrupted, restarting it skips completed game IDs
and plays only missing games.

## 8. Opening suites

[`openings.py`](../alphachess/eval/openings.py) accepts:

- one complete FEN per non-comment line; or
- PGN games, optionally truncated after a fixed number of plies.

[`data/openings/smoke.fen`](../data/openings/smoke.fen) contains four small integration-test
positions. It is not the final experimental suite. The team must still select and freeze
`midsem.fen`, then use its hash in every comparison.

## 9. Stockfish integration

[`stockfish.py`](../alphachess/eval/stockfish.py) serves two purposes:

1. a fixed opponent for matches;
2. a reference engine for top-move and centipawn-loss metrics.

The opponent pins threads and hash size, supports fixed search limits, and checks that the chosen
Stockfish build exposes `UCI_Elo` when limited strength is requested. The pushed implementation
was smoke-tested against Stockfish 19.

## 10. Elo and error bars

[`rating.py`](../alphachess/eval/rating.py) converts game results into:

- wins, draws, and losses;
- score, where a win is `1`, draw `0.5`, and loss `0`;
- Elo difference;
- a confidence interval.

The Elo conversion is:

```text
Elo difference = 400 × log10(score / (1 - score))
```

All-win and all-loss samples would mathematically produce infinite Elo, so finite-sample endpoint
smoothing is applied only in those cases.

Confidence intervals use bootstrap resampling of complete colour-swapped opening pairs. The two
games in a pair share an opening and should not be treated as independent samples.

## 11. Position and puzzle metrics

[`metrics.py`](../alphachess/eval/metrics.py) provides supporting evidence beyond match Elo:

- **Top-move agreement:** how often the candidate chooses Stockfish's best move.
- **Mean centipawn loss:** how much evaluation the selected move loses compared with the best
  move.
- **Lichess puzzle solve rate:** whether the candidate follows the complete recorded solution,
  not merely the first answer.

These are diagnostic metrics. Match results remain the primary strength measurement.

## 12. Experiment and plotting workflow

The intended midsem workflow is:

```text
one frozen checkpoint
      │
      ├── budget 0   ─► paired games ─► summary
      ├── budget 1   ─► paired games ─► summary
      ├── budget 16  ─► paired games ─► summary
      ├── budget 64  ─► paired games ─► summary
      └── budget 256 ─► paired games ─► summary
                                      │
                                      ▼
                         plot_search_scaling.py
                                      │
                         CSV + Elo/error-bar figure
```

The runnable programs are:

| Command | Purpose |
|---|---|
| `python -m scripts.run_uci` | Launch the agent as a UCI engine. |
| `python -m scripts.run_arena` | Run or resume colour-swapped matches. |
| `python -m scripts.eval_positions` | Evaluate FEN positions or Lichess puzzles. |
| `python -m scripts.plot_search_scaling` | Build the final CSV and search-scaling figure. |

The plot script treats the budgets as ordered categories so the zero-simulation baseline can be
shown beside positive budgets without using an invalid logarithm of zero.

## 13. What was verified

The pushed implementation was checked with:

- 32 passing automated tests;
- repository-wide lint checks;
- Git whitespace checks;
- a real subprocess UCI handshake;
- exact visit accounting at budgets `0, 1, 16, 64, 256`;
- real colour-swapped games against Stockfish 19;
- a real Stockfish position-metrics run;
- arena output and restart behavior;
- CSV and PNG plot generation.

Important tests include mate-in-one search, checkmate/stalemate handling, sign-correct backup,
batch accounting, malformed evaluator output, illegal-move forfeits, Elo endpoints, paired
bootstrap reproducibility, full puzzle lines, and Stockfish option configuration.

## 14. What the smoke results do not prove

The committed smoke suite uses an untrained uniform evaluator, tiny search budgets, and short game
caps. Those runs verify engineering integration only. They do **not** demonstrate that MCTS
improves the supervised network or establish a meaningful Elo value.

The research result still requires:

1. the final board and move encoding;
2. a supervised checkpoint;
3. the project-specific evaluator factory;
4. a frozen opening suite;
5. agreed Stockfish Elo settings;
6. pilot runs followed by the final games-per-point run.

When those inputs arrive, the remaining work is to connect the factory and execute the existing
commands. The search, UCI, arena, rating, metrics, and plotting interfaces should remain stable.

## 15. Suggested demonstration order

For a teaching-staff demo, present the contribution in this order:

1. Show one position and the network's raw policy choice.
2. Run MCTS on the same position and show the changed visit distribution or move.
3. Explain the side-to-move value convention and sign flip.
4. Show the UCI engine returning a legal `bestmove`.
5. Show a paired PGN and its JSON result record.
6. Show the Elo-versus-simulations plot with confidence intervals.
7. Close with the Phase 2 reuse: self-play will store the same MCTS visit distribution as the
   improved policy target, and the same arena will rate every later checkpoint.
