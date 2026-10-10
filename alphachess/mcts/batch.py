"""Helpers for evaluator batching."""

from __future__ import annotations

from dataclasses import dataclass

import chess

from .node import Node


@dataclass
class PendingSimulation:
    """A selected leaf and the node path that reaches it."""

    board: chess.Board
    path: list[Node]
    terminal_value: float | None = None

    @property
    def leaf(self) -> Node:
        return self.path[-1]


def reserve(path: list[Node]) -> None:
    for node in path:
        node.pending_visits += 1


def release(path: list[Node]) -> None:
    for node in path:
        node.pending_visits -= 1
        if node.pending_visits < 0:  # pragma: no cover - defensive invariant
            raise RuntimeError("MCTS pending visit count became negative")
