"""Real-time dynamic programming on an endgame graph (the primer's section 4).

The same γ-backup as value iteration, ``V(s) = max_a -γ V(next)``, but only along trials:

- a trial starts at one of the start positions and plays greedily on the current table, *both*
  sides, backing up each position it visits;
- ties are broken by the move this position has tried least often (then by move order);
- values live in a dictionary, so only positions a trial actually visits get an entry; unvisited
  positions read as the initial value, terminal positions read their value from the rules.

We run passes of one trial per unsolved start position and stop when every start is *solved*:
its stored value equals the exact value, and White's greedy play on the current table, against
perfect defence (the tablebase move for Black), mates in exactly the optimal number of plies. The
exact table is used only for this stopping check and for counting wrong entries, never in a
backup.

Two variants:

- :class:`RTDP`: one value per position, initialised to ``init`` (0 in the primer);
- :class:`BoundedRTDP`: an upper and a lower bound per position, so both players stay optimistic.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .graph import Graph


@dataclass
class RTDPResult:
    trials: int
    passes: int
    backups: int
    touched: int  # distinct positions with a table entry
    wrong_entries: int  # touched positions whose stored value is not exact
    seconds: float
    converged: bool
    curve: list[dict] = field(default_factory=list)  # per pass


def _rule_values(graph: Graph) -> np.ndarray:
    """Values known from the rules: checkmate -1, stalemate 0, bare kings (leaves) 0; else NaN."""
    n = graph.n
    rule = np.full(n + graph.num_leaves, np.nan)
    terminal = graph.outdegree == 0
    rule[:n][terminal] = np.where(graph.in_check[terminal], -1.0, 0.0)
    rule[n:] = 0.0
    return rule


def _plays_out(
    g: Graph,
    rule: np.ndarray,
    defence: np.ndarray,
    start: int,
    exact_dtm: int,
    white_edge: Callable[[int], int],
) -> bool:
    """White plays ``white_edge(node)``, Black the tablebase move: mate in exactly ``exact_dtm``?"""
    node = start
    for ply in range(exact_dtm + 1):
        if not np.isnan(rule[node]):
            return rule[node] == -1.0 and ply == exact_dtm
        edge = white_edge(node) if ply % 2 == 0 else int(defence[node])
        node = int(g.succ[edge])
    return False


class _Base:
    def __init__(
        self,
        graph: Graph,
        exact_values: np.ndarray,
        defence: np.ndarray,
        gamma: float,
        max_trial_plies: int,
    ) -> None:
        self.g = graph
        self.exact = exact_values  # (n + L,) exact γ-values: stopping check only
        self.defence = defence  # (n,) tablebase edge per node: stopping check only
        self.gamma = gamma
        self.max_trial_plies = max_trial_plies
        self.rule = _rule_values(graph)
        self.tries: dict[int, int] = {}  # edge id -> times played
        self.backups = 0

    def _edges(self, node: int) -> range:
        return range(int(self.g.succ_ptr[node]), int(self.g.succ_ptr[node + 1]))

    def _least_tried(self, edges, scores, best: float) -> int:
        tied = [e for e, s in zip(edges, scores, strict=True) if s == best]
        return min(tied, key=lambda e: (self.tries.get(e, 0), e))

    def run(self, starts: np.ndarray, exact_dtm: np.ndarray, max_passes: int = 500) -> RTDPResult:
        started = time.perf_counter()
        curve, converged, trials, passes = [], False, 0, 0
        solved = np.zeros(len(starts), dtype=bool)
        for passes in range(1, max_passes + 1):
            for i, s in enumerate(starts):
                if not solved[i]:
                    self.trial(int(s))
                    trials += 1
            solved = np.array([self.solved(int(s), int(exact_dtm[s])) for s in starts])
            curve.append(
                {
                    "pass": passes,
                    "trials": trials,
                    "backups": self.backups,
                    "touched": self.touched(),
                    "starts_solved": int(solved.sum()),
                    "wrong_entries": self.wrong_entries(),
                }
            )
            if solved.all():
                converged = True
                break
        return RTDPResult(
            trials,
            passes,
            self.backups,
            self.touched(),
            self.wrong_entries(),
            time.perf_counter() - started,
            converged,
            curve,
        )


class RTDP(_Base):
    """Plain RTDP with one value per position."""

    def __init__(
        self,
        graph: Graph,
        exact_values: np.ndarray,
        defence: np.ndarray,
        gamma: float = 0.9,
        max_trial_plies: int = 300,
        init: float = 0.0,
    ) -> None:
        super().__init__(graph, exact_values, defence, gamma, max_trial_plies)
        self.init = init  # value of a position with no entry yet
        self.V: dict[int, float] = {}

    def value(self, node: int) -> float:
        rule = self.rule[node]
        return float(rule) if not np.isnan(rule) else self.V.get(node, self.init)

    def _scores(self, node: int) -> tuple[range, list[float]]:
        edges = self._edges(node)
        return edges, [-self.gamma * self.value(int(self.g.succ[e])) for e in edges]

    def backup(self, node: int) -> int:
        """Back up ``node`` and return the greedy edge (ties: least tried, then first)."""
        edges, scores = self._scores(node)
        best = max(scores)
        self.V[node] = best
        self.backups += 1
        return self._least_tried(edges, scores, best)

    def trial(self, start: int) -> None:
        node = start
        for _ in range(self.max_trial_plies):
            if not np.isnan(self.rule[node]):
                return
            edge = self.backup(node)
            self.tries[edge] = self.tries.get(edge, 0) + 1
            node = int(self.g.succ[edge])

    def _greedy_edge(self, node: int) -> int:
        edges, scores = self._scores(node)
        return edges[int(np.argmax(scores))]

    def solved(self, start: int, exact_dtm: int) -> bool:
        if abs(self.V.get(start, self.init) - self.exact[start]) > 1e-12:
            return False
        return _plays_out(self.g, self.rule, self.defence, start, exact_dtm, self._greedy_edge)

    def touched(self) -> int:
        return len(self.V)

    def wrong_entries(self) -> int:
        if not self.V:
            return 0
        nodes = np.fromiter(self.V.keys(), dtype=np.int64)
        values = np.fromiter(self.V.values(), dtype=np.float64)
        return int(np.sum(np.abs(values - self.exact[nodes]) > 1e-12))


class BoundedRTDP(_Base):
    """RTDP with an upper and a lower bound per position (in the spirit of Bounded RTDP).

    One shared value can't be optimistic for both players: in KRK every White-to-move position is
    a win, so an initial 0 is pessimistic for White, who stops exploring once one winning line is
    known (see :class:`RTDP`). Here each visited position stores ``U >= V* >= L`` for its side to
    move, starting from ``U = +1`` and ``L = -1``:

        U(s) = max_a -γ L(next),    L(s) = max_a -γ U(next).

    Each mover plays the move with the best *optimistic* value (``argmax -γ L(next)``, ties by
    least tried), so both sides keep exploring. A trial ends at a terminal position, at a position
    whose bounds have met, or after ``max_trial_plies``; its path is then backed up in reverse
    order.

    ``L(s)`` is the value the side to move can *guarantee* by playing ``argmax -γ U(next)``. A
    start is solved once ``L(start)`` reaches the exact value: White's safe greedy play is then
    optimal against any defence. That needs a proof of White's win (one White move per position,
    every Black reply), not a proof that no faster mate exists (which would need ``U = L``).
    """

    def __init__(
        self,
        graph: Graph,
        exact_values: np.ndarray,
        defence: np.ndarray,
        gamma: float = 0.9,
        max_trial_plies: int = 300,
    ) -> None:
        super().__init__(graph, exact_values, defence, gamma, max_trial_plies)
        self.U: dict[int, float] = {}
        self.L: dict[int, float] = {}

    def bounds(self, node: int) -> tuple[float, float]:
        rule = self.rule[node]
        if not np.isnan(rule):
            return float(rule), float(rule)
        return self.U.get(node, 1.0), self.L.get(node, -1.0)

    def backup(self, node: int) -> int:
        edges = self._edges(node)
        child = [self.bounds(int(self.g.succ[e])) for e in edges]
        optimistic = [-self.gamma * low for _, low in child]
        best = max(optimistic)
        self.U[node] = best
        self.L[node] = max(-self.gamma * u for u, _ in child)
        self.backups += 1
        return self._least_tried(edges, optimistic, best)

    def trial(self, start: int) -> None:
        node, path = start, []
        for _ in range(self.max_trial_plies):
            if not np.isnan(self.rule[node]):
                break
            edge = self.backup(node)
            path.append(node)
            if self.U[node] == self.L[node]:
                break
            self.tries[edge] = self.tries.get(edge, 0) + 1
            node = int(self.g.succ[edge])
        for visited in reversed(path):
            self.backup(visited)

    def _safe_edge(self, node: int) -> int:
        """The move with the best *guaranteed* value, ``argmax -γ U(next)``."""
        edges = self._edges(node)
        scores = [-self.gamma * self.bounds(int(self.g.succ[e]))[0] for e in edges]
        return edges[int(np.argmax(scores))]

    def solved(self, start: int, exact_dtm: int) -> bool:
        """White's guaranteed value is exact, and safe play vs perfect defence realises it.

        If ``L(start) = V*``, White's safe move leads to a child with ``U = V*``, and after the
        perfect Black reply the lower bound is exact again, so the play-out passes by induction.
        """
        if abs(self.bounds(start)[1] - self.exact[start]) > 1e-12:
            return False
        return _plays_out(self.g, self.rule, self.defence, start, exact_dtm, self._safe_edge)

    def touched(self) -> int:
        return len(self.U)

    def wrong_entries(self) -> int:
        """Touched positions whose guaranteed value ``L`` is not the exact value."""
        if not self.L:
            return 0
        nodes = np.fromiter(self.L.keys(), dtype=np.int64)
        low = np.fromiter(self.L.values(), dtype=np.float64)
        return int(np.sum(np.abs(low - self.exact[nodes]) > 1e-12))
