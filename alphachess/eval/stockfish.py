"""Stockfish opponent and reference-analysis adapters."""

from __future__ import annotations

import time
from typing import Any

import chess
import chess.engine

from .player import MoveDecision


class StockfishPlayer:
    def __init__(
        self,
        path: str,
        *,
        elo: int | None = None,
        depth: int | None = 8,
        time_limit: float | None = None,
        threads: int = 1,
        hash_mb: int = 64,
        engine: Any | None = None,
    ) -> None:
        self.engine = engine or chess.engine.SimpleEngine.popen_uci(path)
        self.depth = depth
        self.time_limit = time_limit
        self.name = f"stockfish-{elo}" if elo is not None else "stockfish"
        options: dict[str, int | bool] = {"Threads": threads, "Hash": hash_mb}
        if elo is not None:
            options.update({"UCI_LimitStrength": True, "UCI_Elo": elo})
        supported = {key: value for key, value in options.items() if key in self.engine.options}
        if elo is not None and "UCI_Elo" not in supported:
            raise ValueError("this Stockfish build does not expose UCI_Elo")
        self.engine.configure(supported)

    def select_move(self, board: chess.Board) -> MoveDecision:
        limit = chess.engine.Limit(depth=self.depth, time=self.time_limit)
        started = time.perf_counter()
        result = self.engine.play(board, limit)
        elapsed = time.perf_counter() - started
        return MoveDecision(move=result.move, elapsed_seconds=elapsed)

    def close(self) -> None:
        self.engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class StockfishAnalyser:
    """Return a best move and centipawn scores for every legal candidate."""

    def __init__(self, engine: Any, *, depth: int = 12) -> None:
        self.engine = engine
        self.depth = depth

    def __call__(self, board: chess.Board) -> tuple[chess.Move, dict[chess.Move, int]]:
        scores = {}
        turn = board.turn
        for move in board.legal_moves:
            child = board.copy(stack=False)
            child.push(move)
            info = self.engine.analyse(child, chess.engine.Limit(depth=self.depth))
            score = info["score"].pov(turn).score(mate_score=100_000)
            scores[move] = int(score if score is not None else 0)
        if not scores:
            raise ValueError("cannot analyse a terminal position")
        best_move = max(scores, key=lambda move: (scores[move], move.uci()))
        return best_move, scores
