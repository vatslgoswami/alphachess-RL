"""Minimal UCI protocol wrapper shared by command-line tournament runners."""

from __future__ import annotations

import shlex
import sys
import time
from collections.abc import Callable, Iterable
from typing import TextIO

import chess

from .player import MCTSPlayer, MoveDecision, Player


class UCIEngine:
    def __init__(
        self,
        player: Player,
        *,
        player_factory: Callable[[int, int | None], Player] | None = None,
        name: str = "AlphaChessRL",
        author: str = "Agriya, Samyak, Vatsl",
    ) -> None:
        self.player = player
        self.player_factory = player_factory
        self.name = name
        self.author = author
        self.board = chess.Board()
        self.simulations = getattr(player, "simulations", 0)
        self.seed = getattr(player, "seed", None)

    def _set_position(self, tokens: list[str]) -> None:
        if not tokens:
            raise ValueError("position requires startpos or fen")
        if tokens[0] == "startpos":
            board = chess.Board()
            move_index = 1
        elif tokens[0] == "fen":
            try:
                moves_marker = tokens.index("moves")
            except ValueError:
                moves_marker = len(tokens)
            fen = " ".join(tokens[1:moves_marker])
            board = chess.Board(fen)
            move_index = moves_marker
        else:
            raise ValueError("position requires startpos or fen")
        if move_index < len(tokens) and tokens[move_index] == "moves":
            for value in tokens[move_index + 1 :]:
                board.push_uci(value)
        self.board = board

    def _set_option(self, tokens: list[str]) -> None:
        lowered = [token.lower() for token in tokens]
        if "name" not in lowered or "value" not in lowered:
            return
        name_index = lowered.index("name") + 1
        value_index = lowered.index("value")
        name = " ".join(tokens[name_index:value_index]).lower()
        value = " ".join(tokens[value_index + 1 :])
        if name == "simulations":
            self.simulations = max(0, int(value))
        elif name == "seed":
            self.seed = int(value)
        if self.player_factory is not None:
            self.player = self.player_factory(self.simulations, self.seed)

    def _go(self, tokens: list[str]) -> MoveDecision:
        if "nodes" in tokens and self.player_factory is not None:
            index = tokens.index("nodes")
            simulations = max(0, int(tokens[index + 1]))
            self.player = self.player_factory(simulations, self.seed)
        return self.player.select_move(self.board.copy(stack=True))

    def handle(self, command: str) -> tuple[list[str], bool]:
        tokens = shlex.split(command)
        if not tokens:
            return [], True
        verb, args = tokens[0], tokens[1:]
        if verb == "uci":
            return [
                f"id name {self.name}",
                f"id author {self.author}",
                f"option name Simulations type spin default {self.simulations} min 0 max 1000000",
                f"option name Seed type spin default {self.seed or 0} min 0 max 2147483647",
                "uciok",
            ], True
        if verb == "isready":
            return ["readyok"], True
        if verb == "ucinewgame":
            self.board = chess.Board()
            return [], True
        if verb == "position":
            self._set_position(args)
            return [], True
        if verb == "setoption":
            self._set_option(args)
            return [], True
        if verb == "go":
            started = time.perf_counter()
            decision = self._go(args)
            milliseconds = max(0, round((time.perf_counter() - started) * 1000))
            if decision.move is None:
                return ["bestmove 0000"], True
            nps = round(decision.simulations / max(milliseconds / 1000, 1e-9))
            return [
                f"info nodes {decision.simulations} time {milliseconds} nps {nps}",
                f"bestmove {decision.move.uci()}",
            ], True
        if verb == "stop":
            return [], True
        if verb == "quit":
            return [], False
        return [f"info string unsupported command: {verb}"], True


def run_uci_loop(
    engine: UCIEngine,
    *,
    input_lines: Iterable[str] | None = None,
    output: TextIO | None = None,
) -> None:
    input_lines = sys.stdin if input_lines is None else input_lines
    output = sys.stdout if output is None else output
    for line in input_lines:
        try:
            responses, keep_running = engine.handle(line.strip())
        except Exception as exc:  # noqa: BLE001 - protocol loop must survive bad commands
            print(f"info string error: {type(exc).__name__}: {exc}", file=output, flush=True)
            continue
        for response in responses:
            print(response, file=output, flush=True)
        if not keep_running:
            break


def mcts_player_factory(evaluator, *, c_puct: float = 1.5, batch_size: int = 1):
    def factory(simulations: int, seed: int | None) -> Player:
        if simulations < 1:
            from .player import PolicyPlayer

            return PolicyPlayer(evaluator)
        return MCTSPlayer(
            evaluator,
            simulations,
            c_puct=c_puct,
            batch_size=batch_size,
            seed=seed,
        )

    return factory
