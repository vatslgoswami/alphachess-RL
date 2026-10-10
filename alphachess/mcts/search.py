"""PUCT Monte Carlo tree search with batched evaluator calls."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import chess
import numpy as np

from .batch import PendingSimulation, release, reserve
from .node import Node

Evaluator = Callable[[list[chess.Board]], tuple[list[dict[chess.Move, float]], np.ndarray]]


@dataclass
class SearchDiagnostics:
    simulations: int = 0
    evaluator_calls: int = 0
    evaluated_positions: int = 0
    nodes_expanded: int = 0
    elapsed_seconds: float = 0.0

    @property
    def mean_batch_size(self) -> float:
        return self.evaluated_positions / self.evaluator_calls if self.evaluator_calls else 0.0

    @property
    def simulations_per_second(self) -> float:
        return self.simulations / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0


@dataclass
class SearchResult:
    visits: dict[chess.Move, int] = field(default_factory=dict)
    q: float = 0.0
    best_move: chess.Move | None = None
    policy: dict[chess.Move, float] = field(default_factory=dict)
    diagnostics: SearchDiagnostics = field(default_factory=SearchDiagnostics)


def uniform_evaluator(
    boards: list[chess.Board],
) -> tuple[list[dict[chess.Move, float]], np.ndarray]:
    """Return uniform legal priors and zero values for each board."""
    priors = []
    for board in boards:
        moves = list(board.legal_moves)
        priors.append({move: 1.0 / len(moves) for move in moves} if moves else {})
    return priors, np.zeros(len(boards), dtype=np.float32)


def terminal_value(board: chess.Board) -> float | None:
    """Return a terminal value for the side to move, or ``None`` if play continues."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def _normalise_priors(
    board: chess.Board, priors: dict[chess.Move, float]
) -> dict[chess.Move, float]:
    legal = list(board.legal_moves)
    if not legal:
        return {}
    cleaned = {}
    for move in legal:
        value = float(priors.get(move, 0.0))
        cleaned[move] = value if math.isfinite(value) and value > 0 else 0.0
    total = sum(cleaned.values())
    if total <= 0:
        probability = 1.0 / len(legal)
        return {move: probability for move in legal}
    return {move: value / total for move, value in cleaned.items()}


def _evaluate_and_expand(
    boards: list[chess.Board],
    nodes: list[Node],
    evaluator: Evaluator,
    diagnostics: SearchDiagnostics,
) -> list[float]:
    if not boards:
        return []
    priors_batch, values = evaluator([board.copy(stack=True) for board in boards])
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(priors_batch) != len(boards) or len(values) != len(boards):
        raise ValueError("evaluator returned a different batch size than it received")
    if not np.all(np.isfinite(values)):
        raise ValueError("evaluator returned a non-finite value")

    diagnostics.evaluator_calls += 1
    diagnostics.evaluated_positions += len(boards)
    for board, node, priors in zip(boards, nodes, priors_batch, strict=True):
        node.expand(_normalise_priors(board, priors))
        diagnostics.nodes_expanded += 1
    return [float(np.clip(value, -1.0, 1.0)) for value in values]


def _select_child(node: Node, c_puct: float) -> tuple[chess.Move, Node]:
    parent_visits = node.visit_count + node.pending_visits
    scale = math.sqrt(parent_visits + 1.0)

    def score(item: tuple[chess.Move, Node]) -> tuple[float, str]:
        move, child = item
        child_visits = child.visit_count + child.pending_visits
        q_parent = -child.q
        exploration = c_puct * child.prior * scale / (1 + child_visits)
        return q_parent + exploration, move.uci()

    return max(node.children.items(), key=score)


def _select_leaf(root: Node, root_board: chess.Board, c_puct: float) -> PendingSimulation:
    board = root_board.copy(stack=True)
    node = root
    path = [root]
    while node.expanded and node.children:
        move, node = _select_child(node, c_puct)
        board.push(move)
        path.append(node)
    value = terminal_value(board)
    reserve(path)
    return PendingSimulation(board=board, path=path, terminal_value=value)


def _backup(path: list[Node], leaf_value: float) -> None:
    value = float(leaf_value)
    for node in reversed(path):
        node.update(value)
        value = -value


def _visit_policy(root: Node) -> dict[chess.Move, float]:
    total = sum(child.visit_count for child in root.children.values())
    if total:
        return {move: child.visit_count / total for move, child in root.children.items()}
    total_prior = sum(child.prior for child in root.children.values())
    if not total_prior:
        return {}
    return {move: child.prior / total_prior for move, child in root.children.items()}


def _add_root_noise(
    root: Node,
    rng: np.random.Generator,
    alpha: float,
    epsilon: float,
) -> None:
    if not root.children:
        return
    if alpha <= 0 or not 0 <= epsilon <= 1:
        raise ValueError("Dirichlet alpha must be positive and epsilon must be in [0, 1]")
    noise = rng.dirichlet(np.full(len(root.children), alpha, dtype=np.float64))
    for child, sample in zip(root.children.values(), noise, strict=True):
        child.prior = (1 - epsilon) * child.prior + epsilon * float(sample)


def search(
    board: chess.Board,
    evaluator: Evaluator,
    num_simulations: int,
    add_noise: bool = False,
    *,
    c_puct: float = 1.5,
    batch_size: int = 1,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    seed: int | None = None,
) -> SearchResult:
    """Run PUCT without mutating ``board``.

    Root expansion is not counted as a simulation. Each completed leaf backup is
    counted once, including repeated selections of the same in-flight leaf.
    """
    if num_simulations < 0:
        raise ValueError("num_simulations must be non-negative")
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    if c_puct < 0:
        raise ValueError("c_puct must be non-negative")

    started = time.perf_counter()
    diagnostics = SearchDiagnostics()
    root = Node()
    root_terminal = terminal_value(board)
    if root_terminal is not None:
        diagnostics.elapsed_seconds = time.perf_counter() - started
        return SearchResult(q=root_terminal, diagnostics=diagnostics)

    root_value = _evaluate_and_expand([board], [root], evaluator, diagnostics)[0]
    if add_noise:
        _add_root_noise(
            root,
            np.random.default_rng(seed),
            alpha=dirichlet_alpha,
            epsilon=dirichlet_eps,
        )

    remaining = num_simulations
    while remaining:
        current_batch = min(batch_size, remaining)
        pending = [_select_leaf(root, board, c_puct) for _ in range(current_batch)]
        unique_nonterminal: dict[int, PendingSimulation] = {}
        for simulation in pending:
            if simulation.terminal_value is None and not simulation.leaf.expanded:
                unique_nonterminal.setdefault(id(simulation.leaf), simulation)

        unique = list(unique_nonterminal.values())
        evaluated_values = _evaluate_and_expand(
            [simulation.board for simulation in unique],
            [simulation.leaf for simulation in unique],
            evaluator,
            diagnostics,
        )
        value_by_leaf = {
            id(simulation.leaf): value
            for simulation, value in zip(unique, evaluated_values, strict=True)
        }

        for simulation in pending:
            try:
                value = simulation.terminal_value
                if value is None:
                    value = value_by_leaf[id(simulation.leaf)]
                _backup(simulation.path, value)
                diagnostics.simulations += 1
            finally:
                release(simulation.path)
        remaining -= current_batch

    policy = _visit_policy(root)
    best_move = max(policy, key=lambda move: (policy[move], move.uci())) if policy else None
    diagnostics.elapsed_seconds = time.perf_counter() - started
    return SearchResult(
        visits={move: child.visit_count for move, child in root.children.items()},
        q=root.q if root.visit_count else root_value,
        best_move=best_move,
        policy=policy,
        diagnostics=diagnostics,
    )
