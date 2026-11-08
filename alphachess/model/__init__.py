"""Policy-value ResNet.

Owner: Samyak. Interface: docs/INTERFACES.md.

The network lives in ``alphachess.model.net`` and needs PyTorch (``pip install -e ".[torch]"``).
Names are imported lazily so that ``import alphachess.model`` works without PyTorch (CI installs
only ``.[dev]``).
"""

from __future__ import annotations

__all__ = ["PolicyValueNet", "build_model", "count_parameters"]


def __getattr__(name: str):
    if name in __all__:
        from alphachess.model import net

        return getattr(net, name)
    raise AttributeError(f"module 'alphachess.model' has no attribute {name!r}")
