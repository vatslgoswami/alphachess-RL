"""Turn a saved checkpoint into an MCTS evaluator (INTERFACES §8), for Agriya's harness.

Every CLI that takes ``--evaluator`` (``run_uci``, ``run_arena``, ``eval_positions``) accepts::

    ALPHACHESS_CHECKPOINT=runs/sl_trial_5k/ckpt_00000180.pt \\
        python -m scripts.run_arena --evaluator alphachess.model.evaluator:make_evaluator ...

The network size is read from the checkpoint's saved config, so any checkpoint written by
``alphachess.train.supervised`` loads without extra flags.
"""

from __future__ import annotations

import os

from alphachess.common.checkpoint import load_checkpoint
from alphachess.encoding.board import encode_board, move_to_index
from alphachess.model.net import PolicyValueNet


def load_model(path: str, device: str = "cpu") -> PolicyValueNet:
    ckpt = load_checkpoint(path, map_location=device)
    cfg = ckpt.get("config", {})
    size = cfg.get("train") or cfg.get("model") or {}
    model = PolicyValueNet(blocks=size.get("blocks", 6), channels=size.get("channels", 64))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def make_evaluator(path: str | None = None, device: str | None = None):
    """Zero-argument factory for ``--evaluator``; the checkpoint comes from the environment."""
    from alphachess.eval.evaluator import TorchEvaluator
    from alphachess.train.supervised import pick_device

    path = path or os.environ.get("ALPHACHESS_CHECKPOINT")
    if not path:
        raise SystemExit("set ALPHACHESS_CHECKPOINT to a checkpoint path (runs/<run>/ckpt_*.pt)")
    device = device or os.environ.get("ALPHACHESS_DEVICE") or str(pick_device("auto"))
    return TorchEvaluator(load_model(path, device), encode_board, move_to_index, device=device)
