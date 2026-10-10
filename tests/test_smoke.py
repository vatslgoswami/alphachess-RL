"""Smoke tests: every package imports and the shared config loads."""

import importlib

import pytest

from alphachess.common.config import load_config

PACKAGES = [
    "alphachess.common.config",
    "alphachess.common.types",
    "alphachess.common.checkpoint",
    "alphachess.encoding.board",
    "alphachess.model",
    "alphachess.train",
    "alphachess.endgames.probe",
    "alphachess.lichess",
    "alphachess.replay",
    "alphachess.selfplay",
    "alphachess.mcts.search",
    "alphachess.eval",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_imports(name):
    importlib.import_module(name)


def test_config_loads():
    cfg = load_config()
    assert cfg["encoding"]["num_moves"] == 4672
    for section in ["paths", "endgames", "lichess", "mcts", "train", "eval"]:
        assert section in cfg


def test_uniform_evaluator():
    import chess

    from alphachess.mcts.search import uniform_evaluator

    priors, values = uniform_evaluator([chess.Board()])
    assert len(priors[0]) == 20
    assert abs(sum(priors[0].values()) - 1) < 1e-9
    assert values.shape == (1,)
