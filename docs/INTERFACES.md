# Interfaces

These are the contracts between components. The stubs in `alphachess/` follow them. If you need to change one, open a PR that touches this file and tag the other owners.

Status: **draft**. The encoding owner (Samyak) should confirm §1 and §2 in week 1.

## 1. Board encoding (`alphachess.encoding`) — Samyak

```python
encode_board(board: chess.Board) -> np.ndarray      # float32, shape (C, 8, 8)
NUM_PLANES: int                                     # C
```

- Always from the **side to move's** perspective: flip the board when black is to move, so "my pieces" are always planes 0–5.
- Suggested planes (C = 19 with `history: 1`): 6 own pieces, 6 opponent pieces, 1 repetition count, 4 castling rights, 1 side-to-move colour, 1 fifty-move counter / 100.

## 2. Move encoding — Samyak

```python
NUM_MOVES = 4672                                    # 8 * 8 * 73
move_to_index(move: chess.Move, board: chess.Board) -> int
index_to_move(index: int, board: chess.Board) -> chess.Move
legal_mask(board: chess.Board) -> np.ndarray        # bool, shape (4672,)
```

- AlphaZero layout: from-square (64) × 73 move types (56 queen-like, 8 knight, 9 underpromotions). Queen promotions use the queen-like planes.
- Uses the same perspective flip as §1.
- Invariant (unit test): for every legal move m in a position, `index_to_move(move_to_index(m, b), b) == m`.

## 3. Network (`alphachess.model`) — Samyak

```python
net = PolicyValueNet(blocks, channels)              # torch.nn.Module
policy_logits, value = net(x)                       # x: (B, C, 8, 8)
# policy_logits: (B, 4672) raw logits (mask illegal moves outside the net)
# value: (B,) in [-1, 1], from the side to move's perspective
```

## 4. Value convention (everyone)

- **All values are from the side to move's perspective**, in [-1, 1]: +1 = side to move wins, 0 = draw, −1 = side to move loses.
- Game outcome z for a stored position = result from that position's side to move.
- Endgame DP tables use the same sign convention (see §7).

## 5. Training example / replay record (`alphachess.replay`) — Vatsl

One example is a `Sample` (defined in `alphachess/common/types.py`):

| Field | dtype / shape | Meaning |
|---|---|---|
| `planes` | `float32 (C, 8, 8)` or `uint8` | `encode_board(s)` |
| `pi_idx` | `int16 (k,)` | move indices with non-zero target probability |
| `pi_p` | `float32 (k,)` | target probabilities, sum to 1 |
| `z` | `float32` | value target in [-1, 1] |

- Supervised (Lichess) data: `pi` is one-hot on the human move (`k = 1`), and `z = tanh(cp / 400)` from the Stockfish eval (side to move's perspective; mate scores → ±1).
- Self-play data: `pi` is the MCTS visit distribution, and `z` is the game result (optionally blended with search Q later).
- On disk: shards of `.npz` with stacked arrays `planes (N,C,8,8)`, `pi_idx (N,K)` padded with −1, `pi_p (N,K)` padded with 0, and `z (N,)`.

## 6. Checkpoint (`alphachess.common.checkpoint`) — everyone

`torch.save` of a dict:

```python
{
  "model": state_dict, "optimizer": state_dict | None,
  "step": int, "config": dict, "git_commit": str,
  "created": iso8601 str, "notes": str,
}
```

File name: `runs/<run_name>/ckpt_<step:08d>.pt`. The replay buffer is saved alongside it as `replay_<step:08d>.npz`.

## 7. Endgame tables and tablebase probe (`alphachess.endgames`) — Vatsl

```python
probe_wdl(board) -> int | None   # +1 / 0 / -1 for the side to move, None if not covered
probe_dtm(board) -> int | None   # plies to mate under optimal play (signed: >0 winning), None if not covered
best_move(board) -> chess.Move | None
```

Backed by our own exact DP tables when the material is solved (KQK, KRK, KPK), and by Syzygy (WDL/DTZ only) otherwise when `paths.tablebase_dir` exists. Self-play adjudication (Phase 2) calls `probe_wdl`.

## 8. MCTS (`alphachess.mcts`) — Agriya

```python
search(board, evaluator, num_simulations, add_noise=False) -> SearchResult
# SearchResult.visits: dict[chess.Move, int]; .q: float (root value, side to move); .best_move
# evaluator(boards: list[chess.Board]) -> (priors: list[dict[Move, float]], values: np.ndarray)
```

The `evaluator` indirection lets MCTS run with a uniform or random evaluator before the network exists, and lets it batch calls.
