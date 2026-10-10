# Opening suites

- `smoke.fen` is a tiny committed suite for integration checks only.
- `midsem.fen` is the frozen experimental suite named by `config.yaml`; add it locally after the
  team selects and versions the source. It remains ignored until the team confirms redistribution
  and repository-size constraints.

FEN suites contain one complete FEN per non-comment line. Every position is played twice with
colours swapped by `scripts.run_arena`.
