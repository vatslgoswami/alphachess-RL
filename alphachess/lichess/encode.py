"""Convert raw Lichess Parquet shards into training shards (T9). Format: docs/INTERFACES.md §5.

Each output ``.npz`` holds stacked arrays:

- ``planes``  ``float32 (N, C, 8, 8)``: ``encode_board(board)``;
- ``pi_idx``  ``int16 (N, 1)``: ``move_to_index(human move, board)`` (one-hot policy, ``k = 1``);
- ``pi_p``    ``float32 (N, 1)``: all ones;
- ``z``       ``float32 (N,)``: ``tanh(eval_cp / value_scale)``, or ±1 for a forced mate, from the
  side to move's perspective (the raw shards already store evals that way).

The encoder and move index are parameters, so this runs (and is tested) against a fake encoder
until Samyak's ``alphachess.encoding`` lands.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import chess
import numpy as np

EncodeBoard = Callable[[chess.Board], np.ndarray]
MoveToIndex = Callable[[chess.Move, chess.Board], int]


def value_target(eval_cp: int | None, mate: int | None, value_scale: float) -> float:
    """``z`` for the side to move: ±1 for a forced mate, else ``tanh(cp / value_scale)``."""
    if mate is not None:
        return 1.0 if mate > 0 else -1.0
    if eval_cp is None:
        raise ValueError("row has neither eval_cp nor mate")
    return float(np.tanh(eval_cp / value_scale))


def encode_rows(
    rows: Iterable[dict],
    encode_board: EncodeBoard,
    move_to_index: MoveToIndex,
    value_scale: float = 400.0,
) -> dict[str, np.ndarray]:
    """Encode raw rows (``fen``, ``move_uci``, ``eval_cp``, ``mate``) into §5 arrays."""
    planes, idx, z = [], [], []
    for row in rows:
        board = chess.Board(row["fen"])
        move = chess.Move.from_uci(row["move_uci"])
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {row['move_uci']} in {row['fen']}")
        planes.append(np.asarray(encode_board(board), dtype=np.float32))
        idx.append(move_to_index(move, board))
        z.append(value_target(row.get("eval_cp"), row.get("mate"), value_scale))
    n = len(planes)
    return {
        "planes": np.stack(planes) if n else np.zeros((0, 0, 8, 8), dtype=np.float32),
        "pi_idx": np.asarray(idx, dtype=np.int16).reshape(n, 1),
        "pi_p": np.ones((n, 1), dtype=np.float32),
        "z": np.asarray(z, dtype=np.float32),
    }


def encoder_ready(encode_board: EncodeBoard, move_to_index: MoveToIndex) -> str | None:
    """None if the encoder works, else a message explaining what is missing."""
    board = chess.Board()
    try:
        encode_board(board)
        move_to_index(chess.Move.from_uci("e2e4"), board)
    except NotImplementedError:
        return (
            "The board/move encoder (alphachess.encoding.encode_board / move_to_index) is not "
            "implemented yet. T9 waits on Samyak's encoder (docs/INTERFACES.md §1-2); rerun "
            "scripts/encode_lichess_dataset.py once it has landed."
        )
    return None


def encode_parquet(
    src: str | Path,
    dst: str | Path,
    encode_board: EncodeBoard,
    move_to_index: MoveToIndex,
    value_scale: float = 400.0,
    batch_rows: int = 10_000,
) -> int:
    """Encode one Parquet shard into one compressed ``.npz`` shard. Returns the row count."""
    import pyarrow.parquet as pq

    parts = []
    for batch in pq.ParquetFile(src).iter_batches(batch_size=batch_rows):
        parts.append(encode_rows(batch.to_pylist(), encode_board, move_to_index, value_scale))
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else {}
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **arrays)
    return len(arrays.get("z", []))
