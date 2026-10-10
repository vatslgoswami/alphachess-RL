"""Tree data structures used by Monte Carlo tree search."""

from __future__ import annotations

from dataclasses import dataclass, field

import chess


@dataclass
class Node:
    """A search node whose value is from the side to move at this node."""

    prior: float = 1.0
    visit_count: int = 0
    value_sum: float = 0.0
    pending_visits: int = 0
    children: dict[chess.Move, Node] = field(default_factory=dict)
    expanded: bool = False

    @property
    def q(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def update(self, value: float) -> None:
        self.visit_count += 1
        self.value_sum += float(value)

    def expand(self, priors: dict[chess.Move, float]) -> None:
        self.children = {move: Node(prior=float(prior)) for move, prior in priors.items()}
        self.expanded = True
