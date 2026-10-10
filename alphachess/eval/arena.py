"""Reproducible paired-game evaluation harness."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import chess.pgn

from .openings import Opening
from .player import Player


@dataclass(frozen=True)
class GameRecord:
    game_id: str
    opening_id: str
    white: str
    black: str
    result: str
    result_for_candidate: float
    termination: str
    plies: int
    elapsed_seconds: float
    pgn: str
    error: str | None = None

    def to_json(self) -> str:
        payload = asdict(self)
        payload.pop("pgn")
        return json.dumps(payload, sort_keys=True)


def _score_for_colour(result: str, candidate_colour: chess.Color) -> float:
    if result == "1/2-1/2":
        return 0.5
    candidate_won = (result == "1-0") == (candidate_colour == chess.WHITE)
    return 1.0 if candidate_won else 0.0


def play_game(
    white: Player,
    black: Player,
    opening: Opening,
    *,
    game_id: str,
    candidate_colour: chess.Color,
    max_plies: int = 300,
) -> GameRecord:
    """Play one game, treating crashes and illegal moves as forfeits."""
    board = opening.board()
    game = chess.pgn.Game.from_board(board)
    game.headers.update(
        {
            "Event": "AlphaChess evaluation",
            "White": white.name,
            "Black": black.name,
            "OpeningId": opening.opening_id,
        }
    )
    node = game
    started = time.perf_counter()
    error = None
    termination = "normal"
    forced_result = None

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        player = white if board.turn == chess.WHITE else black
        try:
            decision = player.select_move(board.copy(stack=True))
            move = decision.move
            if move is None or move not in board.legal_moves:
                raise ValueError(f"illegal move returned: {move}")
        except Exception as exc:  # noqa: BLE001 - engine failures become match results
            error = f"{type(exc).__name__}: {exc}"
            termination = "forfeit"
            forced_result = "0-1" if board.turn == chess.WHITE else "1-0"
            break
        board.push(move)
        node = node.add_variation(move)

    if forced_result is not None:
        result = forced_result
    elif board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        result = outcome.result() if outcome is not None else "1/2-1/2"
        termination = outcome.termination.name.lower() if outcome is not None else "normal"
    else:
        result = "1/2-1/2"
        termination = "max_plies"

    game.headers["Result"] = result
    game.headers["Termination"] = termination
    elapsed = time.perf_counter() - started
    return GameRecord(
        game_id=game_id,
        opening_id=opening.opening_id,
        white=white.name,
        black=black.name,
        result=result,
        result_for_candidate=_score_for_colour(result, candidate_colour),
        termination=termination,
        plies=board.ply(),
        elapsed_seconds=elapsed,
        pgn=str(game),
        error=error,
    )


def play_opening_pair(
    candidate: Player,
    opponent: Player,
    opening: Opening,
    *,
    max_plies: int = 300,
) -> tuple[GameRecord, GameRecord]:
    first = play_game(
        candidate,
        opponent,
        opening,
        game_id=f"{opening.opening_id}-candidate-white",
        candidate_colour=chess.WHITE,
        max_plies=max_plies,
    )
    second = play_game(
        opponent,
        candidate,
        opening,
        game_id=f"{opening.opening_id}-candidate-black",
        candidate_colour=chess.BLACK,
        max_plies=max_plies,
    )
    return first, second


class ResultWriter:
    """Append-only JSONL and PGN writer with completed-game discovery."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "games.jsonl"
        self.pgn_path = self.run_dir / "games.pgn"

    def completed_game_ids(self) -> set[str]:
        if not self.jsonl_path.exists():
            return set()
        completed = set()
        for line in self.jsonl_path.read_text().splitlines():
            if line.strip():
                completed.add(json.loads(line)["game_id"])
        return completed

    def append(self, record: GameRecord) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_json() + "\n")
        with self.pgn_path.open("a", encoding="utf-8") as handle:
            handle.write(record.pgn.rstrip() + "\n\n")


def run_paired_match(
    candidate: Player,
    opponent: Player,
    openings: list[Opening],
    writer: ResultWriter,
    *,
    max_plies: int = 300,
) -> list[GameRecord]:
    completed = writer.completed_game_ids()
    records = []
    for opening in openings:
        white_id = f"{opening.opening_id}-candidate-white"
        if white_id not in completed:
            record = play_game(
                candidate,
                opponent,
                opening,
                game_id=white_id,
                candidate_colour=chess.WHITE,
                max_plies=max_plies,
            )
            writer.append(record)
            records.append(record)
        black_id = f"{opening.opening_id}-candidate-black"
        if black_id not in completed:
            record = play_game(
                opponent,
                candidate,
                opening,
                game_id=black_id,
                candidate_colour=chess.BLACK,
                max_plies=max_plies,
            )
            writer.append(record)
            records.append(record)
    return records
