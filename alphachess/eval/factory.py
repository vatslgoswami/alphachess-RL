"""Load evaluator factories without coupling CLI tools to the model implementation."""

from __future__ import annotations

import importlib

from alphachess.mcts import Evaluator, uniform_evaluator


def load_evaluator(factory_spec: str | None) -> Evaluator:
    """Load a zero-argument evaluator factory as ``module:attribute``."""
    if factory_spec in (None, "uniform"):
        return uniform_evaluator
    if ":" not in factory_spec:
        raise ValueError("evaluator factory must be 'uniform' or 'module:attribute'")
    module_name, attribute_name = factory_spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute_name)
    evaluator = factory()
    if not callable(evaluator):
        raise TypeError("evaluator factory did not return a callable")
    return evaluator
