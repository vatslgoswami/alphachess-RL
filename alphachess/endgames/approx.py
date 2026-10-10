"""How well does function approximation recover the exact KRK value? (T6)

Target: the exact γ-value ``±γ^dtm`` (0 for draws) of every legal KRK position, both sides to
move, split 80/20 at random into train and test positions.

Inputs:
  ``onehot``   3 x 64 square one-hots (white king, rook, black king) + side to move (193 dims);
  ``features`` hand features: king-king distances, black king's distance to the edge and to the
               nearest corner, rook-king distances, rook attacked / defended, rook on the black
               king's line, side to move.

Models: a linear model, 2-layer MLPs with 64 / 256 / 1024 hidden units trained on exact labels, and
one *fitted value iteration* learner: an MLP trained for K rounds on bootstrapped targets
``max_a -γ V̂(next)`` computed from its own predictions. It never sees an exact label, only the
rules (checkmate = -1, stalemate or bare kings = 0).

Metrics (overall and per DTM bucket, on test positions):
  ``mse``            value error against the exact γ-value;
  ``rel_err``        mean ``|V̂ - V| / |V|`` on decisive positions (scale-free: γ^dtm shrinks
                     long mates, so their absolute error looks small);
  ``sign_acc``       win/draw/loss read from the predicted value;
  ``optimal_move``   one-move lookahead ``argmax_a -V̂(next)`` picks a DTM-optimal move;
  ``keeps_result``   the lookahead move keeps the exact game result (keeps the win / the draw);
  ``mate_rate``      from won White-to-move test positions, White plays the lookahead move and
                     Black plays the tablebase's defence; mated within ``max_moves`` moves?

Terminal successors are scored by the rules in the lookahead, never by the network.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .graph import Graph, build_graph
from .index import decode
from .solve import LOSS, WIN, best_edges, edge_scores, gamma_values
from .tables import Table, load_or_solve

# DTM buckets in plies (both sides): draws are their own bucket.
BUCKETS = [(0, 7, "0-7"), (8, 15, "8-15"), (16, 23, "16-23"), (24, 32, "24-32")]


@dataclass
class KRKData:
    graph: Graph
    wdl: np.ndarray  # (n + L,) exact
    dtm: np.ndarray  # (n + L,)
    value: np.ndarray  # (n + L,) exact γ-value
    rule_value: np.ndarray  # (n + L,) value known from the rules (terminals), NaN otherwise
    best_edge: np.ndarray  # (n,) tablebase move
    scores: np.ndarray  # (E,) exact move scores
    inputs: dict[str, np.ndarray]  # name -> (n, d) float32
    train: np.ndarray  # node ids
    test: np.ndarray  # node ids
    gamma: float


def chebyshev(a, b):
    return np.maximum(np.abs(a % 8 - b % 8), np.abs(a // 8 - b // 8))


def manhattan(a, b):
    return np.abs(a % 8 - b % 8) + np.abs(a // 8 - b // 8)


def onehot_inputs(stm, wk, rook, bk) -> np.ndarray:
    x = np.zeros((len(stm), 193), dtype=np.float32)
    rows = np.arange(len(stm))
    x[rows, wk] = 1
    x[rows, 64 + rook] = 1
    x[rows, 128 + bk] = 1
    x[:, 192] = stm
    return x


def feature_inputs(stm, wk, rook, bk) -> np.ndarray:
    f, r = bk % 8, bk // 8
    edge = np.minimum(np.minimum(f, 7 - f), np.minimum(r, 7 - r))
    corner = np.minimum(f, 7 - f) + np.minimum(r, 7 - r)  # Manhattan distance to nearest corner
    kk = chebyshev(wk, bk)
    attacked = chebyshev(rook, bk) == 1
    defended = chebyshev(rook, wk) == 1
    same_line = (rook % 8 == bk % 8) | (rook // 8 == bk // 8)
    feats = [
        kk / 7,
        manhattan(wk, bk) / 14,
        edge / 3,
        corner / 6,
        chebyshev(rook, bk) / 7,
        chebyshev(rook, wk) / 7,
        attacked,
        defended,
        attacked & ~defended,
        same_line,
        stm,
        stm * attacked * ~defended,  # Black to move can take an undefended rook
    ]
    return np.stack([np.asarray(x, dtype=np.float32) for x in feats], axis=1)


def load_krk(gamma: float = 0.9, seed: int = 0, table: Table | None = None) -> KRKData:
    table = table or load_or_solve("KRK")
    graph = build_graph("KRK")
    n = graph.n
    wdl = np.concatenate([table.wdl[graph.slots], graph.leaf_wdl])
    dtm = np.concatenate([table.dtm[graph.slots], graph.leaf_dtm])
    value = gamma_values(wdl, dtm, gamma)
    rule = np.full(n + graph.num_leaves, np.nan)
    terminal = graph.outdegree == 0
    rule[:n][terminal] = np.where(graph.in_check[terminal], -1.0, 0.0)
    rule[n:] = value[n:]  # leaves are bare kings: a draw by the rules
    stm, wk, rook, bk = decode(graph.spec, graph.slots)
    inputs = {
        "onehot": onehot_inputs(stm, wk, rook, bk),
        "features": feature_inputs(stm, wk, rook, bk),
    }
    perm = np.random.default_rng(seed).permutation(n)
    cut = int(0.8 * n)
    return KRKData(
        graph=graph,
        wdl=wdl,
        dtm=dtm,
        value=value,
        rule_value=rule,
        best_edge=best_edges(graph, wdl, dtm),
        scores=edge_scores(graph, wdl, dtm),
        inputs=inputs,
        train=np.sort(perm[:cut]),
        test=np.sort(perm[cut:]),
        gamma=gamma,
    )


# --- models ------------------------------------------------------------------------------------


def make_model(hidden: int | None, in_dim: int):
    from torch import nn

    if hidden is None:
        return nn.Linear(in_dim, 1)
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


def predict(model, x: np.ndarray, batch: int = 8192) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(model(torch.from_numpy(x[i : i + batch])).squeeze(-1).numpy())
    return np.clip(np.concatenate(out).astype(np.float64), -1.0, 1.0)


def fit(model, x: np.ndarray, y: np.ndarray, epochs: int, lr: float, seed: int, batch: int = 512):
    import torch

    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(xt), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = torch.mean((model(xt[idx]).squeeze(-1) - yt[idx]) ** 2)
            loss.backward()
            opt.step()
    return model


def full_values(data: KRKData, node_pred: np.ndarray) -> np.ndarray:
    """Predicted values for all nodes + leaves, with terminal positions taken from the rules."""
    v = np.concatenate([node_pred, np.zeros(data.graph.num_leaves)])
    known = ~np.isnan(data.rule_value)
    v[known] = data.rule_value[known]
    return v


def backup_targets(data: KRKData, v: np.ndarray) -> np.ndarray:
    """``max_a -γ v(next)`` for every node (terminals keep their rule value)."""
    g = data.graph
    has = g.outdegree > 0
    out = v[: g.n].copy()
    out[has] = np.maximum.reduceat(-data.gamma * v[g.succ], g.succ_ptr[:-1][has])
    return out


def train_supervised(
    data: KRKData, inputs: str, hidden: int | None, epochs: int, seed: int, lr: float = 1e-3
):
    import torch

    torch.manual_seed(seed)
    x = data.inputs[inputs]
    model = make_model(hidden, x.shape[1])
    fit(model, x[data.train], data.value[data.train], epochs, lr, seed)
    return model


def train_fitted_vi(
    data: KRKData,
    inputs: str,
    hidden: int,
    rounds: int,
    epochs_per_round: int,
    seed: int,
    lr: float = 1e-3,
    log: list | None = None,
):
    """Fitted value iteration: regress on ``max -γ V̂(next)`` from the model's own predictions."""
    import torch

    torch.manual_seed(seed)
    x = data.inputs[inputs]
    model = make_model(hidden, x.shape[1])
    v = full_values(data, np.zeros(data.graph.n))
    for k in range(rounds):
        targets = backup_targets(data, v)
        fit(model, x[data.train], targets[data.train], epochs_per_round, lr, seed + k)
        v = full_values(data, predict(model, x))
        if log is not None:
            err = v[data.test] - data.value[data.test]
            log.append({"round": k + 1, "test_mse": float(np.mean(err**2))})
    return model


# --- evaluation -----------------------------------------------------------------------------


def lookahead_edges(data: KRKData, v: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Edge chosen by one-move lookahead ``argmax -V̂(next)`` for each node (first on ties)."""
    g = data.graph
    starts, ends = g.succ_ptr[nodes], g.succ_ptr[nodes + 1]
    lens = ends - starts
    owner = np.repeat(np.arange(len(nodes)), lens)
    edges = np.repeat(starts, lens) + (
        np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    )
    score = -v[g.succ[edges]]
    order = np.lexsort((edges, -score, owner))
    first = np.ones(len(order), dtype=bool)
    first[1:] = owner[order][1:] != owner[order][:-1]
    chosen = np.full(len(nodes), -1, dtype=np.int64)
    chosen[owner[order][first]] = edges[order][first]
    return chosen


def bucket_of(dtm: np.ndarray, wdl: np.ndarray) -> np.ndarray:
    labels = np.full(len(dtm), "draw", dtype=object)
    for lo, hi, name in BUCKETS:
        labels[(wdl != 0) & (dtm >= lo) & (dtm <= hi)] = name
    return labels


def mate_conversion(data: KRKData, v: np.ndarray, starts: np.ndarray, max_moves: int = 50):
    """White plays the lookahead move, Black the tablebase move. Returns mated-in-time flags."""
    g = data.graph
    node = starts.copy()
    done = np.zeros(len(node), dtype=bool)
    mated = np.zeros(len(node), dtype=bool)
    for _ in range(max_moves):
        active = np.flatnonzero(~done)
        if active.size == 0:
            break
        child = g.succ[lookahead_edges(data, v, node[active])]
        is_leaf = child >= g.n  # rook captured (cannot happen on White's move) -> failure
        mate = ~is_leaf & (g.outdegree[np.minimum(child, g.n - 1)] == 0)
        mate &= ~is_leaf & g.in_check[np.minimum(child, g.n - 1)]
        stale = ~is_leaf & (g.outdegree[np.minimum(child, g.n - 1)] == 0) & ~mate
        mated[active[mate]] = True
        done[active[mate | stale | is_leaf]] = True
        cont = active[~(mate | stale | is_leaf)]
        reply = g.succ[data.best_edge[child[~(mate | stale | is_leaf)]]]
        captured = reply >= g.n
        done[cont[captured]] = True
        node[cont[~captured]] = reply[~captured]
    return mated


def evaluate(data: KRKData, node_pred: np.ndarray, mate_starts: np.ndarray, max_moves: int = 50):
    """Overall and per-bucket metrics on test positions."""
    g = data.graph
    v = full_values(data, node_pred)
    test = data.test
    err = node_pred[test] - data.value[test]
    decisive = data.wdl[test] != 0
    rel = np.full(len(test), np.nan)
    rel[decisive] = np.abs(err[decisive]) / np.abs(data.value[test][decisive])
    tau = 0.5 * data.gamma ** int(data.dtm[: g.n].max())
    pred_wdl = np.where(node_pred[test] > tau, WIN, np.where(node_pred[test] < -tau, LOSS, 0))
    sign_ok = pred_wdl == data.wdl[test]

    movers = test[g.outdegree[test] > 0]
    chosen = lookahead_edges(data, v, movers)
    best_score = np.maximum.reduceat(data.scores, g.succ_ptr[:-1][g.outdegree > 0])
    best_by_node = np.full(g.n, np.iinfo(np.int64).min)
    best_by_node[g.outdegree > 0] = best_score
    optimal = data.scores[chosen] == best_by_node[movers]
    child = g.succ[chosen]
    keeps = np.where(data.wdl[movers] == LOSS, True, -data.wdl[child] == data.wdl[movers])

    mated = mate_conversion(data, v, mate_starts, max_moves)

    def by_bucket(nodes, values):
        labels = bucket_of(data.dtm[nodes], data.wdl[nodes])
        return {
            name: float(np.nanmean(values[labels == name]))
            if np.any(labels == name) and not np.all(np.isnan(values[labels == name]))
            else np.nan
            for name in [b[2] for b in BUCKETS] + ["draw"]
        }

    return {
        "mse": float(np.mean(err**2)),
        "rel_err": float(np.nanmean(rel)),
        "sign_acc": float(np.mean(sign_ok)),
        "optimal_move": float(np.mean(optimal)),
        "keeps_result": float(np.mean(keeps)),
        "mate_rate": float(np.mean(mated)),
        "by_bucket": {
            "mse": by_bucket(test, err**2),
            "rel_err": by_bucket(test, rel),
            "sign_acc": by_bucket(test, sign_ok),
            "optimal_move": by_bucket(movers, optimal),
            "keeps_result": by_bucket(movers, keeps),
            "mate_rate": by_bucket(mate_starts, mated),
        },
        "errors": {"abs_err": np.abs(err), "nodes": test},
    }


def mate_start_positions(data: KRKData, count: int, seed: int) -> np.ndarray:
    """Won White-to-move test positions (sampled with a fixed seed)."""
    g = data.graph
    stm = decode(g.spec, g.slots)[0]
    pool = data.test[(stm[data.test] == 0) & (data.wdl[data.test] == WIN)]
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool, size=min(count, len(pool)), replace=False))


def timed(fn, *args, **kwargs):
    started = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - started


class SlotValueEvaluator:
    """MCTS ``Evaluator`` with uniform priors and a learned value looked up by table slot.

    ``values_by_slot`` holds the model's prediction for every slot of ``table`` (side to move's
    perspective). Positions of other material get value 0, like the uniform evaluator.
    """

    def __init__(self, values_by_slot: np.ndarray, table: str = "KRK") -> None:
        self.values_by_slot = values_by_slot
        self.table = table

    def __call__(self, boards):
        from . import probe
        from .index import board_to_index

        priors, values = [], np.zeros(len(boards), dtype=np.float32)
        for i, board in enumerate(boards):
            moves = list(board.legal_moves)
            priors.append({m: 1.0 / len(moves) for m in moves} if moves else {})
            found = probe._own_table(board)
            if found is not None and found[0] == self.table:
                values[i] = self.values_by_slot[board_to_index(found[1], self.table)]
        return priors, values


def values_by_slot(data: KRKData, node_pred: np.ndarray) -> np.ndarray:
    out = np.zeros(data.graph.spec.size, dtype=np.float32)
    out[data.graph.slots] = node_pred
    return out
