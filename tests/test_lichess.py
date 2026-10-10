"""Tests for the Lichess pipeline: streaming, filtering, sampling, shards and encoding.

The PGN fixture is built in the test (no network). Parquet and zstd tests skip when the optional
``[data]`` extra isn't installed.
"""

from __future__ import annotations

import io

import chess
import chess.pgn
import numpy as np
import pytest

from alphachess.lichess import encode, sample, stream

OPENING = [
    "e4",
    "e5",
    "Nf3",
    "Nc6",
    "Bb5",
    "a6",
    "Ba4",
    "Nf6",
    "O-O",
    "Be7",
    "Re1",
    "b5",
    "Bb3",
    "d6",
    "c3",
    "O-O",
    "h3",
    "Nb8",
    "d4",
    "Nbd7",
]


def make_game(gid: str, white_elo=2000, black_elo=1950, tc="300+0", evals=True, mate_at=None):
    game = chess.pgn.Game()
    game.headers.update(
        {
            "Event": "Rated Blitz game",
            "Site": f"https://lichess.org/{gid}",
            "WhiteElo": str(white_elo),
            "BlackElo": str(black_elo),
            "TimeControl": tc,
            "Result": "1-0",
        }
    )
    node = game
    for i, san in enumerate(OPENING):
        node = node.add_variation(node.board().parse_san(san))
        if evals:
            # White-POV eval after move i: +i centipawns, so the sign tells us the perspective.
            node.comment = (
                f"[%eval #{mate_at[1]}]"
                if mate_at and i == mate_at[0]
                else (f"[%eval {i / 100:.2f}]")
            )
    return game


def fixture_pgn() -> str:
    games = [
        make_game("good0001"),
        make_game("lowelo01", white_elo=1500),
        make_game("bullet01", tc="60+0"),
        make_game("noeval01", evals=False),
        make_game("mate0001", mate_at=(11, -4)),
    ]
    return "\n\n".join(str(g) for g in games) + "\n"


CFG = sample.SampleConfig(min_elo=1800, min_base_time=180, positions_per_game=8, skip_first_plies=8)


def test_header_filter():
    assert sample.keep_headers(make_game("a").headers, CFG)
    assert not sample.keep_headers(make_game("b", black_elo=1700).headers, CFG)
    assert not sample.keep_headers(make_game("c", tc="120+1").headers, CFG)
    assert not sample.keep_headers(make_game("d", tc="-").headers, CFG)
    casual = make_game("e")
    casual.headers["Event"] = "Casual Blitz game"
    assert not sample.keep_headers(casual.headers, CFG)


def test_iter_games_skips_filtered_games_before_parsing_moves():
    stats = {}
    games = list(
        stream.iter_games(io.StringIO(fixture_pgn()), lambda h: sample.keep_headers(h, CFG), stats)
    )
    assert stats["games_seen"] == 5
    assert [sample.game_id(g.headers) for g in games] == ["good0001", "noeval01", "mate0001"]
    assert len(list(games[0].mainline_moves())) == len(OPENING)


def test_sample_game_rows_and_side_to_move_perspective():
    rows = sample.sample_game(make_game("good0001"), CFG)
    assert len(rows) == 8
    assert set(rows[0]) == set(sample.COLUMNS)
    for row in rows:
        assert row["ply"] >= CFG.skip_first_plies
        board = chess.Board(row["fen"])
        assert chess.Move.from_uci(row["move_uci"]) in board.legal_moves
        # The position after ply p-1 carries White-POV eval (p-1) cp; stored for the side to move.
        white_pov = row["ply"] - 1
        expected = white_pov if board.turn == chess.WHITE else -white_pov
        assert row["eval_cp"] == expected
        assert row["mate"] is None
        assert row["game_id"] == "good0001" and row["eval_source"] == "lichess"
    assert rows == sample.sample_game(make_game("good0001"), CFG)  # reproducible


def test_mate_scores_are_converted_to_side_to_move():
    cfg = sample.SampleConfig(positions_per_game=100, skip_first_plies=8)
    rows = {r["ply"]: r for r in sample.sample_game(make_game("mate0001", mate_at=(11, -4)), cfg)}
    row = rows[12]  # position after move index 11, White to move; White-POV "#-4"
    assert chess.Board(row["fen"]).turn == chess.WHITE
    assert row["mate"] == -4 and row["eval_cp"] is None


def test_games_without_evals_are_skipped_unless_allowed():
    game = make_game("noeval01", evals=False)
    assert not sample.has_evals(game)
    assert sample.sample_game(game, CFG) == []
    loose = sample.SampleConfig(require_eval=False)
    rows = sample.sample_game(game, loose)
    assert len(rows) == 8 and all(r["eval_cp"] is None for r in rows)


def test_split_is_deterministic_and_by_game():
    ids = [f"game{i:05d}" for i in range(4000)]
    splits = [sample.split_of(g, CFG) for g in ids]
    assert splits == [sample.split_of(g, CFG) for g in ids]
    share = {s: splits.count(s) / len(ids) for s in sample.SPLITS}
    assert 0.87 < share["train"] < 0.93
    assert 0.03 < share["val"] < 0.07 and 0.03 < share["test"] < 0.07


def test_build_dataset_writes_parquet_shards(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    stats_stream = {}
    games = stream.iter_games(
        io.StringIO(fixture_pgn()), lambda h: sample.keep_headers(h, CFG), stats_stream
    )
    writer = sample.ShardWriter(tmp_path, shard_size=5)
    stats = sample.build_dataset(
        games, writer, CFG, target_positions=12, stream_stats=stats_stream, log=lambda _: None
    )
    assert stats.positions == 16  # stops after the game that crosses the target
    assert stats.games_used == 2 and stats.games_with_evals == 2
    assert stats.games_seen == 5 and stats.games_kept_headers == 3
    rows = []
    for name in writer.shards:
        rows += pq.read_table(tmp_path / name).to_pylist()
    assert len(rows) == 16
    assert {r["game_id"] for r in rows} == {"good0001", "mate0001"}
    assert all(r["eval_source"] == "lichess" for r in rows)
    summary = sample.write_summary(stats, tmp_path)
    assert summary.exists()


def test_open_pgn_reads_plain_and_zstd_files(tmp_path):
    text = fixture_pgn()
    plain = tmp_path / "games.pgn"
    plain.write_text(text)
    with stream.open_pgn(plain) as handle:
        assert sum(1 for _ in stream.iter_games(handle)) == 5
    zstd = pytest.importorskip("zstandard")
    compressed = tmp_path / "games.pgn.zst"
    compressed.write_bytes(zstd.ZstdCompressor().compress(text.encode()))
    with stream.open_pgn(compressed) as handle:
        assert sum(1 for _ in stream.iter_games(handle)) == 5


def test_month_url():
    assert stream.month_url("2024-01").endswith("lichess_db_standard_rated_2024-01.pgn.zst")


# --- T9: encoding -----------------------------------------------------------------------------


def fake_encode_board(board: chess.Board) -> np.ndarray:
    planes = np.zeros((19, 8, 8), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        planes[piece.piece_type - 1 + (0 if piece.color == board.turn else 6), sq // 8, sq % 8] = 1
    return planes


def fake_move_to_index(move: chess.Move, board: chess.Board) -> int:
    return move.from_square * 73 + move.to_square % 73


def test_value_target():
    assert encode.value_target(0, None, 400) == 0.0
    assert encode.value_target(400, None, 400) == pytest.approx(np.tanh(1.0))
    assert encode.value_target(-400, None, 400) == pytest.approx(-np.tanh(1.0))
    assert encode.value_target(None, 3, 400) == 1.0
    assert encode.value_target(None, -2, 400) == -1.0
    with pytest.raises(ValueError):
        encode.value_target(None, None, 400)


def test_encode_rows_with_fake_encoder():
    rows = sample.sample_game(make_game("good0001"), CFG)
    arrays = encode.encode_rows(rows, fake_encode_board, fake_move_to_index, 400)
    n = len(rows)
    assert arrays["planes"].shape == (n, 19, 8, 8) and arrays["planes"].dtype == np.float32
    assert arrays["pi_idx"].shape == (n, 1) and arrays["pi_idx"].dtype == np.int16
    assert np.all(arrays["pi_p"] == 1.0)
    expected = [np.tanh(r["eval_cp"] / 400) for r in rows]
    assert np.allclose(arrays["z"], expected)
    first = chess.Board(rows[0]["fen"])
    assert arrays["pi_idx"][0, 0] == fake_move_to_index(
        chess.Move.from_uci(rows[0]["move_uci"]), first
    )


def test_encoder_ready_reports_the_stub():
    from alphachess.encoding.board import encode_board, move_to_index

    message = encode.encoder_ready(encode_board, move_to_index)
    if message is not None:  # still a stub
        assert "Samyak" in message
    assert encode.encoder_ready(fake_encode_board, fake_move_to_index) is None


def test_encode_parquet_round_trip(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = sample.sample_game(make_game("good0001"), CFG)
    src = tmp_path / "train-00000.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=sample.SCHEMA()), src)
    n = encode.encode_parquet(
        src, tmp_path / "out" / "train-00000.npz", fake_encode_board, fake_move_to_index, 400
    )
    assert n == len(rows)
    with np.load(tmp_path / "out" / "train-00000.npz") as data:
        assert set(data.files) == {"planes", "pi_idx", "pi_p", "z"}
        assert data["planes"].shape == (n, 19, 8, 8)
