# Agriya — Phase 1 plan: search and evaluation

Status: planning document; no implementation is included here.

## 1. Phase 1 outcome

Agriya's Phase 1 work should answer one experimental question:

> With the network checkpoint held fixed, how much does MCTS improve playing strength as
> the search budget increases?

The work has three deliverables:

1. a correct, testable MCTS implementation under `alphachess/mcts/`;
2. a UCI entry point under `alphachess/eval/` so standard chess tools can run the agent; and
3. a reproducible evaluation harness that produces Elo-versus-simulations results, confidence
   intervals, game records, and supporting test-set metrics.

This is Agriya's contribution to the wider midsem story. The team's exact endgame DP and
approximation experiments are owned by Vatsl; the encoder, network, and supervised checkpoint
are owned by Samyak. Agriya's code consumes their published interfaces without duplicating them.

## 2. Repository fit

The existing repository already fixes the package layout, configuration, and shared interface.
The implementation should extend that structure as follows:

```text
alphachess/
├── mcts/
│   ├── __init__.py          public search exports
│   ├── node.py              node/edge statistics and selection helpers
│   ├── search.py            PUCT search and SearchResult (existing stub)
│   ├── batch.py             leaf collection and batched evaluator calls
│   └── gumbel.py            low-budget variant; add only after PUCT is validated
└── eval/
    ├── __init__.py          public evaluation exports (existing stub)
    ├── evaluator.py         checkpoint-to-MCTS evaluator adapter
    ├── player.py            raw-policy and MCTS player abstractions
    ├── uci.py               stdin/stdout UCI engine loop
    ├── arena.py             paired games, colour swap, adjudication, PGN output
    ├── openings.py          deterministic opening-suite loading and validation
    ├── rating.py            scores, Elo differences, and uncertainty
    └── metrics.py           top-move agreement, centipawn loss, puzzle results

scripts/
├── run_uci.py               launch the engine for fastchess/cutechess
├── run_arena.py             run/resume a match experiment
├── eval_positions.py        test-set and puzzle evaluation
└── plot_search_scaling.py   create the midsem figure and summary table

tests/
├── test_mcts.py
├── test_uci.py
├── test_arena.py
└── test_rating.py

reports/midsem/
├── figures/                 generated plots
└── tables/                  generated CSV summaries

data/openings/               local opening suites; large files remain untracked
runs/<run_name>/             config snapshot, metadata, JSONL results, PGNs, logs
```

Keep reusable logic in `alphachess/`; scripts should only parse arguments, load configuration,
and call library functions. Do not create parallel board, move, model, or checkpoint formats.

## 3. System boundary and data flow

```text
checkpoint + config
        │
        ▼
Network evaluator ── legal priors + value ──► PUCT search ──► chosen move
        ▲                                          │
        │                                          ▼
 encoded boards                         visits, root Q, diagnostics
                                                   │
                     ┌─────────────────────────────┴────────────────────┐
                     ▼                                                  ▼
                UCI process                                      Python arena
                     │                                                  │
                     └──── fastchess/cutechess + Stockfish ─────────────┘
                                                                        │
                                                         JSONL + PGN + run metadata
                                                                        │
                                                                        ▼
                                                            ratings, CIs, plots
```

There should be only one move-selection path. Both the UCI wrapper and the Python arena call the
same player/search API so tournament results measure the same engine that is demonstrated live.

## 4. Contracts to freeze before implementation

The draft contracts in `docs/INTERFACES.md` remain authoritative. Agriya should get explicit
confirmation from Samyak on board/move encoding before integrating a real checkpoint.

### MCTS input and output

Retain the existing public call:

```python
search(board, evaluator, num_simulations, add_noise=False) -> SearchResult
```

Required semantics:

- `board` is not mutated;
- `evaluator(list[Board])` returns legal-move priors and values in the side-to-move convention;
- `SearchResult.visits` contains root legal moves only;
- `SearchResult.q` is the root value from the root player's perspective;
- `SearchResult.best_move` is deterministic when noise is off and a seed/tie rule is fixed;
- zero simulations means raw-policy selection in the player layer, not a malformed tree search;
- one simulation and terminal-root positions have explicitly tested behavior.

The network adapter must mask illegal logits and renormalize them. If all legal logits become
non-finite or sum to zero, it should fail loudly in development and have a documented uniform
fallback for evaluation runs.

### Value convention

Every stored node value is from the side to move at that node. A checkmated node evaluates to
`-1`; draws evaluate to `0`. Backup negates the value once per traversed ply. This convention is
the highest-risk correctness point and must be covered by small forced-mate tests.

### Configuration

Continue using the existing `mcts` and `eval` sections in `config.yaml`. Add keys rather than
renaming existing ones. Likely additions are:

- MCTS seed, temperature/tie rule, virtual-loss or in-flight visit setting, and Gumbel toggle;
- engine checkpoint path, device, and UCI defaults;
- Stockfish binary, opening-suite path, move/game limits, concurrency, and output directory;
- evaluation seed and a clear pilot/final game count.

Each run must save the resolved configuration and Git commit next to its results.

## 5. Work packages

### WP1 — Deterministic PUCT core

Implement the smallest correct search first:

1. terminal detection and side-to-move values;
2. expansion from evaluator priors;
3. PUCT selection using `Q + U`;
4. negamax backup with a sign flip per edge;
5. visit-count policy and best-move selection;
6. optional root Dirichlet noise, applied once at the root only.

Do not add transposition tables in Phase 1. Chess repetition state and the half-move clock make
state identity easy to get wrong, while the midsem result does not require tree reuse across
moves. Reusing the chosen child as the next root can be revisited after the basic search is
validated.

Acceptance gate:

- finds mate in one for either colour;
- avoids a move that allows forced mate in one;
- backup signs match hand-calculated two- and three-ply trees;
- all returned moves are legal;
- visit counts equal the requested completed simulation count under the documented convention;
- fixed seed plus noise off gives identical results.

### WP2 — Batched evaluator path

Keep the public search result unchanged while collecting several pending leaves before one
network call. Use temporary in-flight visits or virtual loss so a batch does not repeatedly select
the same path. First prove that batch size 1 matches the deterministic PUCT reference; then test
larger batches.

Record search diagnostics needed for performance work: completed simulations, evaluator calls,
mean batch size, elapsed time, nodes expanded, and simulations/second. The Phase 1 target is
correct batching and measurable throughput, not maximum optimization.

Acceptance gate:

- batch size 1 agrees with the reference search;
- a batched call never contains illegal or duplicate accidental work caused by missing in-flight
  accounting;
- requested simulations are neither lost nor over-counted;
- benchmark shows evaluator calls decrease as batching increases.

### WP3 — Gumbel low-budget search

Treat Gumbel search as a second implementation milestone, not part of the initial PUCT core. Add
it only after the PUCT tests and end-to-end arena are green. Compare it at 16 and 32 simulations
under the same checkpoint, openings, and seeds. If it is not stable by the experiment freeze,
report PUCT only and list Gumbel as post-midsem work rather than weakening the primary result.

### WP4 — Network evaluator and player layer

Build an adapter that:

- loads the repository checkpoint format;
- encodes a list of positions using `alphachess.encoding`;
- runs one inference batch in evaluation/no-gradient mode;
- masks illegal policy logits and returns normalized move priors;
- returns values with shape `(B,)` in the shared side-to-move convention.

Expose two comparable players:

- `PolicyPlayer`: zero-search baseline using masked network policy;
- `MCTSPlayer`: the same checkpoint plus a specified simulation budget.

This makes the `0, 1, 16, 64, 256` experiment precise: `0` is the policy baseline and every
positive number is completed MCTS simulations.

### WP5 — UCI wrapper

Implement the minimum robust UCI subset needed by fastchess/cutechess:

- `uci`, `isready`, `ucinewgame`, `position startpos ...`, `position fen ...`, `go`, `stop`, and
  `quit`;
- options for checkpoint, simulations, device, and seed;
- `bestmove <uci_move>` for every valid non-terminal request.

Map `go nodes N` directly to simulations where possible. For `go movetime`, use a documented
budget policy and stop cooperatively; fixed-node/simulation matches remain the scientific
protocol because they are more reproducible than wall-clock limits.

Protocol output must stay on stdout and contain only valid UCI messages. Diagnostics go to
stderr or run logs. Add transcript tests that launch the engine as a subprocess and verify the
handshake, FEN reconstruction, legal `bestmove`, new-game reset, and clean shutdown.

### WP6 — Match harness and rating

Build a small Python arena first for smoke tests, then use the UCI wrapper with fastchess or
cutechess for final tournament runs. The harness should:

- use a versioned, fixed opening suite;
- play every opening twice with colours swapped;
- pin checkpoint, Stockfish version/settings, simulations, seeds, and adjudication rules;
- forfeit illegal moves or crashes and record the failure reason;
- save one append-only result row per game and PGNs, allowing interrupted runs to resume;
- separate pilot runs from final results.

For each data point report wins, draws, losses, score, Elo difference, and a 95% interval. Estimate
uncertainty by resampling colour-swapped opening pairs, not individual games, because paired games
share an opening. Document smoothing at all-win/all-loss endpoints so Elo never becomes infinite.

Use two evaluation modes:

1. **controlled ablation:** policy-only and each MCTS budget play the same opponents/openings;
2. **sanity match:** adjacent budgets play each other to confirm the ordering without relying
   only on Stockfish's limited-strength calibration.

### WP7 — Supporting position metrics

These metrics are supporting evidence, not substitutes for games:

- top-move agreement on a frozen position set;
- centipawn loss computed by one pinned Stockfish configuration;
- Lichess puzzle success with an explicit success rule (first move only or full line).

Freeze the position IDs before running comparisons. Cache Stockfish reference analysis so all
search budgets are scored against identical targets.

## 6. Test plan

| Layer | Fast tests | Slow/integration tests |
|---|---|---|
| Tree logic | hand-built priors/values, terminal states, sign backup, visit totals | tactical FEN suite at several budgets |
| Batching | batch-1 equivalence, pending-leaf accounting | throughput benchmark on fixed positions |
| Evaluator | legal masking, normalization, shapes, NaN handling | load a real supervised checkpoint |
| UCI | command transcript and legal best move | short external-engine match |
| Arena | colour pairing, adjudication, resume/idempotence | fixed-seed mini gauntlet |
| Rating | known W/D/L cases, endpoint handling | bootstrap coverage simulation |
| Metrics | fixed toy positions and cache reuse | frozen test set/puzzle subset |

Normal CI should stay under roughly ten seconds. Mark checkpoint, Stockfish, tournament, and
throughput tests with `@pytest.mark.slow` as required by the repository conventions.

## 7. Experiment protocol for the midsem result

### Frozen factors

- one supervised checkpoint and exact Git commit;
- budgets `0, 1, 16, 64, 256`;
- same opening pairs and random seeds for every budget;
- same Stockfish binary, `UCI_LimitStrength`/Elo settings, threads, and hash;
- same hardware class, concurrency, move cap, and adjudication rules;
- root noise off and deterministic move choice for evaluation.

### Run stages

1. **Smoke:** 2–4 games per condition; detect crashes and illegal moves.
2. **Pilot:** 20–40 games per condition; estimate runtime, draw rate, and useful Stockfish levels.
3. **Freeze:** lock config, checkpoint hash, opening-suite hash, and output schema.
4. **Final:** target the project-plan value of 200 games per point, in colour-swapped pairs.
5. **Analyse:** generate the plot and table only from the frozen final run.

Primary figure: Elo estimate versus simulations on a log-like categorical x-axis, with 95%
intervals. Include a companion table with W/D/L, score, games, wall-clock time, and simulations
per second. Do not claim monotonic improvement where intervals overlap; state the measured effect
and uncertainty.

## 8. Suggested sequence and milestones

The dates should be filled in once the midsem date is known.

| Order | Milestone | Evidence of completion | Dependency |
|---:|---|---|---|
| 1 | Freeze contracts | team acknowledgement of encoding/value/checkpoint semantics | Samyak |
| 2 | PUCT reference | unit tests and tactical micro-suite pass | none; uniform evaluator works |
| 3 | Batched search | batch-1 equivalence and benchmark report | milestone 2 |
| 4 | Python arena | paired deterministic games and resumable results | milestone 2 |
| 5 | Network integration | raw policy and MCTS both play legal games | Samyak checkpoint/encoder |
| 6 | UCI engine | transcript tests and a short fastchess/cutechess match | milestones 4–5 |
| 7 | Pilot evaluation | runtime/draw-rate report; final opponent levels chosen | Stockfish + openings |
| 8 | Gumbel decision | validated 16/32-simulation result or explicit deferral | core pipeline green |
| 9 | Frozen final run | complete JSONL, PGNs, metadata, no missing pairs | milestone 7 |
| 10 | Midsem assets | plot, table, demo command, method/limitations text | milestone 9 |

The critical path is PUCT → arena → real checkpoint → UCI → pilot → final matches. Rating and
plotting can be developed against synthetic game records in parallel with search integration.

## 9. Midsem presentation package

Agriya should hand over:

- a one-command UCI launch and one-command reproducible evaluation run;
- a live or recorded example showing raw policy versus MCTS on the same position;
- the Elo-versus-simulations figure with uncertainty;
- a W/D/L and throughput table;
- one search-tree diagnostic or tactical example explaining why search changed the move;
- a concise methods section covering PUCT, side-to-move backup, pairing, and rating uncertainty;
- a limitations slide: small sample size, Stockfish calibration, checkpoint dependence, and
  compute limits;
- a post-midsem integration slide showing that the same MCTS API will feed self-play visit targets
  and the same harness will rate later RL checkpoints.

## 10. Risks and fallback decisions

| Risk | Early signal | Phase 1 response |
|---|---|---|
| Backup sign bug | black/white mate tests disagree | block integration until hand-tree tests pass |
| Network not ready | no checkpoint by network-integration milestone | finish with uniform/scripted evaluator and wire in checkpoint later |
| Python search too slow | low simulations/second, idle accelerator | batch first, profile, then reduce final games/budgets transparently |
| UCI tool setup is fragile | crashes in smoke match | retain Python arena as a diagnostic path; fix UCI before final ratings |
| Many draws | wide intervals and weak separation | tune opponent levels in pilot; never change them after freeze |
| 200 games/point is infeasible | pilot runtime exceeds compute budget | reduce all points equally and report wider intervals |
| Gumbel delays core work | PUCT/UCI/harness not yet end-to-end | defer Gumbel to post-midsem |

## 11. Definition of done

Agriya's Phase 1 is complete when:

- PUCT search passes sign, terminal, legality, reproducibility, and batching tests;
- the supervised checkpoint can play through the same search API at every planned budget;
- a standard UCI tournament tool can complete paired games without protocol failures;
- runs are resumable and preserve config, commit, seeds, PGNs, and per-game outcomes;
- the final script produces Elo estimates and honest 95% intervals from frozen data;
- the `0, 1, 16, 64, 256` comparison and its limitations are ready for the midsem report; and
- the interfaces required later by self-play and checkpoint gauntlets remain stable.

## 12. Decisions to take to the next team meeting

1. Confirm the exact midsem date and available compute budget.
2. Confirm Samyak's encoding planes, move mapping, checkpoint readiness date, and evaluator device.
3. Choose fastchess or cutechess as the final tournament runner and pin its version.
4. Choose/version the opening suite and decide whether it may be committed or must stay in data.
5. Confirm Stockfish levels and whether 200 games per point is affordable after the pilot.
6. Decide whether Gumbel is required for midsem or explicitly a stretch goal.
7. Decide the exact puzzle-success rule and frozen test-set source.
