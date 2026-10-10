"""Solved tables: build (graph + retrograde), save to and load from ``data/endgames/<name>.npz``.

Arrays are indexed by slot (``index.py``), so a lookup is ``table.wdl[board_to_index(board)]``.
Illegal and non-canonical slots hold ``legal = False``, ``wdl = 0`` and ``dtm = -1``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np

from alphachess.common.config import load_config, repo_path

from .graph import Graph, build_graph
from .index import TableSpec, get_spec
from .solve import LOSS, WIN, Solution, solve

# Tables a table depends on through promotion.
DEPENDENCIES = {"KQK": [], "KRK": [], "KPK": ["KQK", "KRK"]}


@dataclass
class Table:
    spec: TableSpec
    legal: np.ndarray  # (size,) bool
    wdl: np.ndarray  # (size,) int8, side to move
    dtm: np.ndarray  # (size,) int16 plies, -1 for draws / illegal
    best_from: np.ndarray  # (size,) int8, -1 if none
    best_to: np.ndarray  # (size,) int8
    best_promo: np.ndarray  # (size,) int8, 0 for none
    meta: dict = field(default_factory=dict)

    def best_move(self, slot: int) -> chess.Move | None:
        if self.best_from[slot] < 0:
            return None
        return chess.Move(
            int(self.best_from[slot]), int(self.best_to[slot]), int(self.best_promo[slot]) or None
        )

    def lookup(self, slot: int) -> tuple[int, int]:
        return int(self.wdl[slot]), int(self.dtm[slot])


def endgame_dir() -> Path:
    return repo_path(load_config()["paths"]["endgame_dir"])


def table_from_solution(graph: Graph, sol: Solution, meta: dict | None = None) -> Table:
    spec = graph.spec
    legal = np.zeros(spec.size, dtype=bool)
    legal[graph.slots] = True
    wdl = np.zeros(spec.size, dtype=np.int8)
    dtm = np.full(spec.size, -1, dtype=np.int16)
    wdl[graph.slots] = sol.wdl[: graph.n]
    dtm[graph.slots] = sol.dtm[: graph.n]
    best_from = np.full(spec.size, -1, dtype=np.int8)
    best_to = np.full(spec.size, -1, dtype=np.int8)
    best_promo = np.zeros(spec.size, dtype=np.int8)
    has = sol.best_edge >= 0
    edges = sol.best_edge[has]
    slots = graph.slots[has]
    best_from[slots] = graph.move_from[edges]
    best_to[slots] = graph.move_to[edges]
    best_promo[slots] = graph.move_promo[edges]
    return Table(spec, legal, wdl, dtm, best_from, best_to, best_promo, meta or {})


def summarise(table: Table) -> dict:
    """Position counts, win/draw/loss shares and longest mates, per side to move."""
    out = {}
    half = table.spec.half
    for stm, name in ((0, "white_to_move"), (1, "black_to_move")):
        sl = slice(stm * half, (stm + 1) * half)
        legal = table.legal[sl]
        wdl, dtm = table.wdl[sl][legal], table.dtm[sl][legal]
        n = int(legal.sum())
        out[name] = {
            "positions": n,
            "win": int((wdl == WIN).sum()),
            "draw": int((wdl == 0).sum()),
            "loss": int((wdl == LOSS).sum()),
            "max_dtm_plies": int(dtm.max()) if n else 0,
        }
    out["positions"] = out["white_to_move"]["positions"] + out["black_to_move"]["positions"]
    return out


def solve_table(name: str, tables: dict[str, Table] | None = None) -> tuple[Table, Graph]:
    """Build the graph and solve ``name``. ``tables`` supplies solved promotion targets."""
    spec = get_spec(name)
    tables = dict(tables or {})
    for dep in DEPENDENCIES[spec.name]:
        if dep not in tables:
            tables[dep] = load_or_solve(dep)

    def external(dep: str, slot: int) -> tuple[int, int]:
        return tables[dep].lookup(slot)

    graph = build_graph(spec, external if DEPENDENCIES[spec.name] else None)
    sol = solve(graph)
    meta = {
        "name": spec.name,
        "build_seconds": round(graph.build_seconds, 3),
        "solve_seconds": round(sol.seconds, 3),
        "nodes": graph.n,
        "edges": graph.num_edges,
        "levels": sol.levels,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    table = table_from_solution(graph, sol, meta)
    table.meta["summary"] = summarise(table)
    return table, graph


def save_table(table: Table, directory: str | Path | None = None) -> Path:
    directory = Path(directory) if directory is not None else endgame_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{table.spec.name}.npz"
    np.savez_compressed(
        path,
        legal=table.legal,
        wdl=table.wdl,
        dtm=table.dtm,
        best_from=table.best_from,
        best_to=table.best_to,
        best_promo=table.best_promo,
        meta=np.array(json.dumps(table.meta)),
    )
    return path


def load_table(name: str, directory: str | Path | None = None) -> Table:
    directory = Path(directory) if directory is not None else endgame_dir()
    spec = get_spec(name)
    with np.load(directory / f"{spec.name}.npz") as data:
        return Table(
            spec,
            data["legal"],
            data["wdl"],
            data["dtm"],
            data["best_from"],
            data["best_to"],
            data["best_promo"],
            json.loads(str(data["meta"])),
        )


def load_or_solve(name: str, directory: str | Path | None = None, save: bool = True) -> Table:
    """Load ``name`` from disk, solving (and saving) it first if the file is missing."""
    directory = Path(directory) if directory is not None else endgame_dir()
    try:
        return load_table(name, directory)
    except FileNotFoundError:
        table, _ = solve_table(name)
        if save:
            save_table(table, directory)
        return table
