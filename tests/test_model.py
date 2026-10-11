"""Tests for the policy-value network (INTERFACES §3)."""

from __future__ import annotations

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from alphachess.common.config import load_config
from alphachess.encoding.board import (
    NUM_MOVES,
    NUM_PLANES,
    encode_board,
    move_to_index,
)
from alphachess.model.net import PolicyValueNet, build_model, count_parameters


def test_output_shapes_and_value_range():
    net = PolicyValueNet(blocks=2, channels=16).eval()
    x = torch.randn(5, NUM_PLANES, 8, 8)
    logits, value = net(x)
    assert logits.shape == (5, NUM_MOVES)
    assert value.shape == (5,)
    assert torch.all(value.abs() <= 1)


def test_build_from_config():
    net = build_model(load_config())
    assert net.blocks == 6 and net.channels == 64
    assert count_parameters(net) > 100_000


def test_policy_layout_matches_move_index():
    """Flat logit s*73 + t must be the policy plane t at square s, i.e. move_to_index order."""
    net = PolicyValueNet(blocks=1, channels=8).eval()
    x = torch.randn(1, NUM_PLANES, 8, 8)
    with torch.no_grad():
        h = net.tower(net.stem(x))
        planes = net.policy_head(h)[0]  # (73, 8, 8)
        logits, _ = net(x)
    board = chess.Board()
    move = chess.Move.from_uci("g1f3")
    index = move_to_index(move, board)
    square, move_type = divmod(index, 73)
    rank, file = divmod(square, 8)
    assert torch.isclose(logits[0, index], planes[move_type, rank, file])


def test_works_with_mcts_torch_evaluator():
    from alphachess.eval.evaluator import TorchEvaluator

    net = PolicyValueNet(blocks=1, channels=8)
    evaluator = TorchEvaluator(net, encode_board, move_to_index)
    priors, values = evaluator([chess.Board(), chess.Board("8/8/8/8/8/2k5/8/K6Q b - - 0 1")])
    assert len(priors[0]) == 20
    assert abs(sum(priors[0].values()) - 1) < 1e-6
    assert values.shape == (2,) and np.all(np.abs(values) <= 1)
