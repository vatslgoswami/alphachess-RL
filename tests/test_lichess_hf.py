"""Tests for the Hugging Face Parquet reader (alphachess.lichess.hf)."""

from __future__ import annotations

import random

import chess
import chess.pgn
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from alphachess.lichess import hf
from alphachess.lichess.sample import SampleConfig


def random_movetext(rng: random.Random, plies: int, evals: bool) -> str:
    game = chess.pgn.Game()
    node = game
    for _ in range(plies):
        moves = list(node.board().legal_moves)
        if not moves:
            break
        node = node.add_variation(rng.choice(moves))
        if evals:
            node.comment = f"[%eval {rng.uniform(-2, 2):.2f}] [%clk 0:05:00]"
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=True)
    return game.accept(exporter)


def make_parquet(path, n=60, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        rows.append({
            "Event": "Rated Blitz game" if i % 5 else "Casual Blitz game",
            "Site": f"https://lichess.org/g{i:07d}",
            "White": "a", "Black": "b",
            "Result": "1-0",
            "WhiteElo": 2000 if i % 3 else 1500,   # every third game fails the Elo filter
            "BlackElo": 1950,
            "TimeControl": "300+0",
            "movetext": random_movetext(rng, 40, evals=i % 2 == 0),  # half have evals
        })
    pq.write_table(pa.Table.from_pylist(rows), path)
    return rows


def expected_games(rows):
    return [r for i, r in enumerate(rows)
            if i % 5 and i % 3 and i % 2 == 0]


def test_process_parquet_filters_and_samples(tmp_path):
    rows = make_parquet(tmp_path / "x.parquet")
    out = hf.process_parquet(tmp_path / "x.parquet", SampleConfig(), batch_size=7)
    assert out["games_seen"] == 60
    kept = expected_games(rows)
    assert out["games_with_evals"] == len(kept) == len(out["rows"])
    for game_rows in out["rows"]:
        assert 1 <= len(game_rows) <= 8
        for row in game_rows:
            board = chess.Board(row["fen"])
            assert chess.Move.from_uci(row["move_uci"]) in board.legal_moves
            assert row["eval_cp"] is not None
            assert row["ply"] >= 8


def test_row_to_game_round_trips_headers():
    game = hf.row_to_game({"Event": 'Rated "x"', "WhiteElo": 2100, "Site": "s", "movetext": "1. e4 e5 *"})
    assert game.headers["Event"] == 'Rated "x"'
    assert game.headers["WhiteElo"] == "2100"
    assert [m.uci() for m in game.mainline_moves()] == ["e2e4", "e7e5"]


def test_insecure_only_for_huggingface():
    with pytest.raises(ValueError):
        hf._open("https://example.com/file", insecure=True)


def test_file_url_quotes_path():
    url = hf.file_url("data/year=2024/month=01/train-00000-of-00061.parquet")
    assert url.endswith("/resolve/main/data/year=2024/month=01/train-00000-of-00061.parquet")
