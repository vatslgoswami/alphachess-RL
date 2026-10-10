"""Tests for players, UCI, arena persistence, openings, and rating."""

import io
import json
import subprocess
import sys
from types import SimpleNamespace

import chess
import numpy as np
import pytest

from alphachess.eval.arena import ResultWriter, play_game, run_paired_match
from alphachess.eval.metrics import (
    evaluate_lichess_puzzles,
    evaluate_positions,
    puzzle_solve_rate,
)
from alphachess.eval.openings import Opening, load_fens
from alphachess.eval.player import MoveDecision, PolicyPlayer
from alphachess.eval.rating import score_to_elo, summarise_results
from alphachess.eval.stockfish import StockfishPlayer
from alphachess.eval.uci import UCIEngine, mcts_player_factory, run_uci_loop
from alphachess.mcts import uniform_evaluator


class FirstLegalPlayer:
    def __init__(self, name="first-legal"):
        self.name = name

    def select_move(self, board):
        return MoveDecision(next(iter(board.legal_moves), None))


class IllegalPlayer:
    name = "illegal"

    def select_move(self, _board):
        return MoveDecision(chess.Move.from_uci("a1a8"))


def test_policy_player_uses_largest_legal_prior():
    def evaluator(boards):
        return [{chess.Move.from_uci("e2e4"): 0.9}], np.zeros(len(boards), dtype=np.float32)

    decision = PolicyPlayer(evaluator).select_move(chess.Board())
    assert decision.move == chess.Move.from_uci("e2e4")


def test_uci_transcript_and_position_reconstruction():
    factory = mcts_player_factory(uniform_evaluator)
    engine = UCIEngine(factory(1, 4), player_factory=factory)
    output = io.StringIO()
    run_uci_loop(
        engine,
        input_lines=[
            "uci\n",
            "isready\n",
            "position startpos moves e2e4 e7e5\n",
            "go nodes 2\n",
            "quit\n",
        ],
        output=output,
    )
    lines = output.getvalue().splitlines()
    assert "uciok" in lines
    assert "readyok" in lines
    bestmove = next(line.split()[1] for line in lines if line.startswith("bestmove "))
    assert chess.Move.from_uci(bestmove) in engine.board.legal_moves


def test_uci_script_works_as_a_real_subprocess():
    process = subprocess.run(
        [sys.executable, "-m", "scripts.run_uci", "--simulations", "1"],
        input="uci\nisready\nposition startpos\ngo nodes 1\nquit\n",
        text=True,
        capture_output=True,
        check=True,
    )
    assert "uciok" in process.stdout
    assert "readyok" in process.stdout
    assert "bestmove " in process.stdout


def test_arena_records_illegal_move_as_forfeit():
    opening = Opening("start", chess.STARTING_FEN)
    record = play_game(
        IllegalPlayer(),
        FirstLegalPlayer(),
        opening,
        game_id="forfeit",
        candidate_colour=chess.WHITE,
    )
    assert record.result == "0-1"
    assert record.termination == "forfeit"
    assert record.result_for_candidate == 0.0
    assert record.error is not None


def test_paired_match_is_resumable(tmp_path):
    writer = ResultWriter(tmp_path)
    opening = Opening("start", chess.STARTING_FEN)
    candidate = FirstLegalPlayer("candidate")
    opponent = FirstLegalPlayer("opponent")
    first = run_paired_match(candidate, opponent, [opening], writer, max_plies=2)
    second = run_paired_match(candidate, opponent, [opening], writer, max_plies=2)
    assert len(first) == 2
    assert second == []
    rows = [json.loads(line) for line in writer.jsonl_path.read_text().splitlines()]
    assert len(rows) == 2
    assert {row["result"] for row in rows} == {"1/2-1/2"}


def test_load_fens_ignores_comments(tmp_path):
    path = tmp_path / "openings.fen"
    path.write_text(f"# suite\n{chess.STARTING_FEN}\n")
    openings = load_fens(path)
    assert len(openings) == 1
    assert openings[0].board().fen() == chess.STARTING_FEN


def test_paired_bootstrap_rating_is_finite_and_reproducible():
    pairs = [(1.0, 0.5), (0.5, 0.0), (1.0, 1.0)]
    first = summarise_results(pairs, bootstrap_samples=500, seed=7)
    second = summarise_results(pairs, bootstrap_samples=500, seed=7)
    assert first == second
    assert first.wins == 3 and first.draws == 2 and first.losses == 1
    assert first.elo_low <= first.elo_difference <= first.elo_high
    assert np.isfinite(score_to_elo(1.0, games=10))
    assert score_to_elo(0.75, games=10) == pytest.approx(400 * np.log10(3))


def test_position_metrics_and_puzzle_rate():
    board = chess.Board()
    chosen = next(iter(board.legal_moves))
    player = FirstLegalPlayer()

    def analyse(_board):
        return chosen, {chosen: 50}

    summary = evaluate_positions(player, [board], analyse)
    assert summary.top_move_agreement == 1.0
    assert summary.mean_centipawn_loss == 0
    assert puzzle_solve_rate([True, False, True]) == pytest.approx(2 / 3)


def test_complete_lichess_puzzle_line_is_checked():
    row = {
        "FEN": chess.STARTING_FEN,
        "Moves": "e2e4 e7e5 g1f3",
    }

    class RecordedPlayer:
        name = "recorded"

        def select_move(self, _board):
            return MoveDecision(chess.Move.from_uci("e7e5"))

    summary = evaluate_lichess_puzzles(RecordedPlayer(), [row])
    assert summary.attempted == 1
    assert summary.solved == 1


def test_stockfish_player_configures_limited_strength_and_returns_move():
    class FakeEngine:
        def __init__(self):
            self.options = {
                "Threads": object(),
                "Hash": object(),
                "UCI_LimitStrength": object(),
                "UCI_Elo": object(),
            }

        def configure(self, options):
            self.configured = options

        def play(self, _board, _limit):
            return SimpleNamespace(move=chess.Move.from_uci("e2e4"))

        def quit(self):
            self.closed = True

    engine = FakeEngine()
    player = StockfishPlayer("unused", elo=1500, engine=engine)
    assert engine.configured["UCI_Elo"] == 1500
    assert player.select_move(chess.Board()).move == chess.Move.from_uci("e2e4")
    player.close()
    assert engine.closed
