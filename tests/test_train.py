"""End-to-end test of the supervised trainer on synthetic §5 shards."""

from __future__ import annotations

import json
import random

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from alphachess.common.checkpoint import load_checkpoint
from alphachess.encoding.board import encode_board, move_to_index
from alphachess.lichess.encode import encode_rows
from alphachess.model.net import PolicyValueNet
from alphachess.train.supervised import (
    TrainConfig,
    dense_policy_target,
    iterate_batches,
    train,
)


def make_rows(n: int, seed: int) -> list[dict]:
    """Rows shaped like the raw Lichess Parquet: random positions, random legal move."""
    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        board = chess.Board()
        for _ in range(rng.randint(0, 40)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        moves = list(board.legal_moves)
        if not moves:
            continue
        rows.append(
            {
                "fen": board.fen(),
                "move_uci": rng.choice(moves).uci(),
                "eval_cp": rng.randint(-600, 600),
                "mate": None,
            }
        )
    return rows


def write_shard(path, rows):
    arrays = encode_rows(rows, encode_board, move_to_index, 400)
    np.savez_compressed(path, **arrays)


def test_dense_policy_target_handles_padding():
    idx = torch.tensor([[5, -1], [7, 9]])
    p = torch.tensor([[1.0, 0.0], [0.25, 0.75]])
    target = dense_policy_target(idx, p)
    assert target.shape == (2, 4672)
    assert target[0, 5] == 1 and target[0].sum() == 1
    assert target[1, 7] == 0.25 and target[1, 9] == 0.75


def test_iterate_batches_covers_every_row(tmp_path):
    write_shard(tmp_path / "train-00000.npz", make_rows(37, 0))
    write_shard(tmp_path / "train-00001.npz", make_rows(20, 1))
    paths = sorted(str(p) for p in tmp_path.glob("*.npz"))
    batches = list(iterate_batches(paths, 16, np.random.default_rng(0)))
    assert sum(len(b["z"]) for b in batches) == 57


def test_training_run_end_to_end(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    write_shard(data / "train-00000.npz", make_rows(64, 0))
    write_shard(data / "val-00000.npz", make_rows(32, 1))
    cfg = TrainConfig(
        run_dir=str(tmp_path / "run"),
        train_shards=[str(data / "train-00000.npz")],
        val_shards=[str(data / "val-00000.npz")],
        blocks=1,
        channels=16,
        batch_size=16,
        lr=3e-3,
        epochs=15,
        eval_every=20,
        device="cpu",
    )
    summary = train(cfg)
    run = tmp_path / "run"
    assert (run / "summary.json").exists() and (run / "metrics.jsonl").exists()
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    train_losses = [r["loss"] for r in records if r["split"] == "train"]
    assert train_losses[-1] < train_losses[0]  # it learns the (memorisable) training set

    ckpt = load_checkpoint(summary["checkpoint"])
    assert set(ckpt) >= {"model", "optimizer", "step", "config", "git_commit", "created", "notes"}
    net = PolicyValueNet(blocks=1, channels=16)
    net.load_state_dict(ckpt["model"])
    assert 0.0 <= summary["val_final"]["policy_top1"] <= 1.0
