# Vatsl Phase 1: status

Branch `vatsl/phase1-endgames`, PR
[vatslgoswami/alpha-chess-rl#1](https://github.com/vatslgoswami/alpha-chess-rl/pull/1) into
`agriya/phase1-search-eval`. Plan: [Vatsl Phase 1 plan](https://claude.ai/code/artifact/bc74afe0-6348-4fff-b27c-5b3fd57e5c3b).
Write-up with all results: [`reports/midsem/endgames.md`](../reports/midsem/endgames.md).

All eight slices of the plan have landed. Two things are left for Vatsl: the full 2M-position
Lichess run (slice 6), and running the encoding once Samyak's encoder exists (slice 7).

## Headline numbers

| Result | Number |
|---|---|
| KQK / KRK / KPK legal positions (symmetry-reduced) | 46,137 / 50,015 / 165,676 |
| Longest forced mate, White to move | KQK 19 plies (mate in 10), KRK 31 (mate in 16), KPK 55 (mate in 28) |
| WDL agreement with Syzygy, every legal position | 100% for all three tables |
| Solve time (graph build + retrograde) | KQK 2.9 + 0.09 s, KRK 2.7 + 0.10 s, KPK 10.9 + 0.18 s |
| KPK, White to move | 76.5% wins, 23.5% draws; rook pawn with the defender in front: 0% won |
| KRK value nets: mate conversion vs perfect defence | MLP-1024 (one-hot) 94.8%; hand-feature MLPs ≤ 3.2%; fitted VI (no labels) 32.8% |
| KRK: VI vs retrograde vs RTDP backups | 1.65M / 0.52M / 0.61–0.68M (bounded RTDP; plain RTDP with V₀ = 0 does not converge) |
| MCTS on the KRK suite, DTM-optimal moves | uniform evaluator flat at 33% (1–1,024 sims); learned value 33% → 70% |
| Lichess smoke run (2024-01) | 5,000 positions in 21 s; 1,394 games/s scanned; ≈ 2.4 h for 2M positions |

## What landed

### Endgames (`alphachess/endgames/`)

| Module | What it does |
|---|---|
| `index.py` | Position ↔ slot index with symmetry reduction (8 symmetries pawnless, mirror for KPK); `board_to_index`, `index_to_board`, `is_legal_index` |
| `graph.py` | Successor graph built once with python-chess, stored as CSR arrays (successors and predecessors) |
| `solve.py` | Retrograde analysis (count-down BFS by DTM level), best moves, γ-values |
| `tables.py` | Solve, save and load `data/endgames/<name>.npz` (`legal`, `wdl`, `dtm`, `best_from/to/promo`, metadata) |
| `probe.py` | `probe_wdl`, `probe_dtm`, `best_move` (INTERFACES §7): our tables for KQK/KRK/KPK (either colour), Syzygy otherwise, else `None`; lazy and cached |
| `verify.py` | Syzygy WDL check, Bellman consistency, independent value iteration, Stockfish DTM spot-check |
| `vi.py` | Synchronous value iteration on γ-values |
| `rtdp.py` | Plain RTDP (one value, dictionary store, least-tried ties) and RTDP with upper/lower bounds |
| `approx.py` | KRK approximation study: inputs, models, fitted value iteration, metrics, learned-value MCTS evaluator |
| `hooks.py` | `DPEvaluator`, `TablebasePlayer`, `exact_reference_analyser`, frozen-suite builder/loader |
| `plotting.py` | Shared figure style |

### Lichess (`alphachess/lichess/`)

| Module | What it does |
|---|---|
| `stream.py` | Streams `.pgn.zst` over HTTP (zstandard), or a local `.pgn`/`.pgn.zst`; skips games that fail the header filter before parsing their moves |
| `sample.py` | Filters (rated, Elo ≥ 1800 both, base ≥ 180 s, has `[%eval]`), samples 8 positions/game after ply 8, evals in the **side to move's** perspective, split by game hash, Parquet shards, `summary.json` |
| `encode.py` | Raw rows → INTERFACES §5 arrays (`planes`, `pi_idx`, `pi_p`, `z = tanh(cp/400)`, mate → ±1); encoder injected |

### Scripts (`python -m scripts.<name>`)

| Script | Output |
|---|---|
| `solve_endgame --table all` | `data/endgames/{KQK,KRK,KPK}.npz`, `reports/midsem/results/endgame_summary.csv`, `kpk_win_by_pawn_square.csv`, figures `dtm_histograms.png`, `kpk_win_by_pawn_square.png` |
| `verify_endgame --table all` | `reports/midsem/results/verification.csv` |
| `run_approx` | `results/approx_metrics.csv`, `approx_fvi_rounds.csv`, figures `approx_error_vs_size.png`, `approx_error_vs_dtm.png` |
| `run_rtdp` | `results/rtdp_results.{csv,md}`, `rtdp_curve.csv`, figure `rtdp_touched_vs_trials.png` |
| `run_endgame_mcts` | `data/endgames/suite.fen` (if missing), `results/mcts_vs_sims_krk.csv`, figure `mcts_vs_sims_krk.png` |
| `build_lichess_dataset` | `data/lichess/raw/<month>/{train,val,test}-NNNNN.parquet` + `summary.json` |
| `encode_lichess_dataset` | `data/lichess/encoded/<month>/*.npz` (exits with a clear message until the encoder lands) |

### Tests

`tests/test_endgames.py` (indexing round-trips and symmetries, mate-in-10/16, probe cases incl.
stalemate, hanging rook and colour flip, Bellman and VI checks, Syzygy on every KQK position, RTDP,
hooks, suite, approximation metrics; KPK textbook positions and the Stockfish check are `slow`) and
`tests/test_lichess.py` (PGN fixture built in the test: filters, sampling, eval perspective, splits,
Parquet, zstd, encoding with a fake encoder). `pytest -q`: 78 passed (~25 s);
`pytest -m slow`: 11 more.

### Data in the repo

- `data/syzygy/K{Q,R,P}vK.rtb{w,z}`: the 3-piece Syzygy tables (~33 KB in total), committed.
- `data/endgames/suite.fen`: the frozen 300-position endgame suite (100 per table), committed.
- `data/endgames/*.npz` and `data/lichess/` stay git-ignored. Regenerate with the commands below.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,torch,data]" && sudo apt-get install -y stockfish
python -m scripts.solve_endgame --table all          # ~20 s
python -m scripts.verify_endgame --table all         # ~30 min (Stockfish part); --stockfish-samples 0: ~1 min
python -m scripts.run_approx                         # ~6 min
python -m scripts.run_rtdp                           # ~5 min
python -m scripts.run_endgame_mcts                   # ~3 min
python -m scripts.build_lichess_dataset --month 2024-01 --target 5000   # smoke run, ~30 s
pytest -q && pytest -q -m slow
```

## What was verified, and how

- **WDL against Syzygy on every legal position** of KQK, KRK and KPK: 100% agreement
  (`results/verification.csv`). The tables were downloaded from `tablebase.lichess.ovh`.
- **DTM**:
  - every position satisfies the negamax Bellman equation on our own graph, and every stored best
    move leads to a successor exactly one ply closer to mate;
  - independent synchronous value iteration reproduces WDL and DTM exactly;
  - Stockfish 16 on 1,000 random positions per table (300k nodes, no tablebases) never found a
    shorter mate than ours, nor a mate in a position we call drawn. Without tablebases it often
    reports a longer mate or none, so this check is one-sided.
- **Known values**: longest mates of 10 / 16 / 28 moves, and the KPK textbook positions.

## Left for Vatsl

1. **Full Lichess run** (slice 6), on your machine:
   `python -m scripts.build_lichess_dataset --month 2024-01 --target 2000000`. That's about 2.4 h at
   the smoke-run rate, which is only an estimate: throughput depends on your connection and on how
   many games carry evals. Pick the month before running (open question in the plan).
2. **Encoding** (slice 7) once Samyak's encoder is merged:
   `python -m scripts.encode_lichess_dataset --month 2024-01`, then publish the shards as a Kaggle
   Dataset.
3. Decide with Samyak whether Lichess's own evals (depth varies by game) are good enough, or
   whether to re-annotate at a fixed depth (`--allow-no-eval --stockfish-fallback`; much slower).

## Blocked on teammates

- **T9 encoding waits on Samyak's encoder** (`alphachess/encoding/board.py`, INTERFACES §1–2). The
  code and tests are done against a fake encoder.

## Notes for the team

- **Eval perspective.** The raw Parquet shards store `eval_cp` and `mate` from the **side to
  move's** perspective (INTERFACES §4), not White's as in the PGN. `result` keeps the PGN string.
- **Top-move agreement.** `evaluate_positions` counts agreement with a single best move. Most
  KRK positions have several equally fast mates, so `hooks.optimal_rates` (any DTM-optimal move)
  is the fairer metric for endgames. Both are reported.
- **Uniform MCTS on long mates.** With `uniform_evaluator`, KRK mates are beyond the search
  horizon, every leaf is 0, and `MCTSPlayer`'s move falls to its `uci()` tie-break. That is
  expected behaviour, not a bug in Agriya's code. A learned value at the leaves is what makes
  search pay off.
- **One PR.** The plan suggested follow-up PRs for later slices; the task asked for all slices on
  one branch and one PR, so everything is in PR #1.

## Handover to Samyak

Hi Samyak, a 2-minute summary of what's yours next.

1. **Board and move encoder** (`alphachess/encoding/`, INTERFACES §1–2). The Lichess encoding (T9)
   is waiting on it. Once it lands, run `python -m scripts.encode_lichess_dataset --month 2024-01`
   to turn the raw Parquet shards into §5 `.npz` shards. Raw shard columns: `fen, move_uci,
   eval_cp, mate, elo_white, elo_black, result, ply, game_id, eval_source`, with evals already
   from the side to move's perspective.
2. **Supervised training code**, plus a **trial run on the Lichess data before the break**, to
   prove the training loop works end to end (load a shard, train, checkpoint, report policy
   accuracy and value error on the val split). The long training run happens after the break.
3. **The midsem presentation, in LaTeX (Beamer).** It must show:
   - **everything we've done so far, on all three tracks**: exact endgame DP (Vatsl), MCTS and
     the evaluation harness (Agriya), encoding, network and training (you);
   - **graphs and diagrams**. Ready-made figures are in `reports/midsem/figures/`:
     `dtm_histograms.png`, `kpk_win_by_pawn_square.png`, `approx_error_vs_size.png`,
     `approx_error_vs_dtm.png`, `rtdp_touched_vs_trials.png` and `mcts_vs_sims_krk.png`, plus
     Agriya's Elo-vs-simulations plot from `scripts/plot_search_scaling.py`. Add an architecture
     diagram of the pipeline (network → MCTS → self-play → replay buffer → trainer, with the
     tablebase cut-off) and one of the network;
   - **the methodology going forward**: supervised training on Lichess, then MCTS self-play RL
     (approximate policy iteration: search improves the policy, training stores the improvement),
     the evaluation plan (Elo vs Stockfish levels, Elo vs training iterations), and the ablations.
4. **An explainer doc for your track** (encoding, network, supervised and RL training), with the
   theory and worked examples. Everyone writes one: see `docs/Vatsl_Endgame_DP_Primer.html` and
   `docs/Agriya_Phase1_Search_Eval_Explainer.md`.

**Scope decisions to plan around:**

- Self-play is restricted to **French Defense** openings.
- Games stop at **5 or fewer pieces** and take the exact Syzygy result. The bot also plays those
  endgames straight from the tables.
- So RL data is French-only and tablebase-cut. Weakness in other openings and in endgames the
  network never trains on is expected, not a bug. The supervised Lichess stage still covers all
  of chess, so the starting network is general.

What you can use now: `alphachess.endgames.probe` (`probe_wdl`, `probe_dtm`, `best_move`) for exact
endgame values, and `DPEvaluator` / `TablebasePlayer` in `alphachess/endgames/hooks.py`.
