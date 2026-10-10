"""Tests for the exact endgame solver, indexing, probe API and verification checks."""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest

from alphachess.common.config import load_config, repo_path
from alphachess.endgames import index as ix
from alphachess.endgames import probe, tables, verify

SYZYGY_DIR = repo_path(load_config()["paths"]["tablebase_dir"])


def _has_syzygy(name: str) -> bool:
    return (Path(SYZYGY_DIR) / f"{name}.rtbw").exists()


@pytest.fixture(scope="session")
def solved(tmp_path_factory):
    """KQK and KRK solved once into a temporary directory used by the probe."""
    directory = tmp_path_factory.mktemp("endgames")
    out = {}
    for name in ("KQK", "KRK"):
        table, graph = tables.solve_table(name)
        tables.save_table(table, directory)
        out[name] = (table, graph)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tables, "endgame_dir", lambda: directory)
        probe.get_table.cache_clear()
        yield out
    probe.get_table.cache_clear()


# --- T1: indexing ----------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["KQK", "KRK", "KPK"])
def test_every_legal_index_round_trips(name):
    spec = ix.get_spec(name)
    legal = np.flatnonzero(ix.legal_mask(spec))
    canon, _ = ix.canonical(spec, *ix.decode(spec, legal))
    assert np.array_equal(canon, legal)
    rng = np.random.default_rng(0)
    for slot in rng.choice(legal, 300, replace=False):
        board = ix.index_to_board(int(slot), spec)
        assert board.is_valid()
        assert ix.board_to_index(board, spec) == slot
        assert ix.is_legal_index(int(slot), spec)


def test_slot_counts():
    assert ix.TABLES["KQK"].size == 81_920
    assert ix.TABLES["KPK"].size == 196_608


def _transform(board: chess.Board, sym: int) -> chess.Board:
    out = chess.Board(None)
    for sq, piece in board.piece_map().items():
        out.set_piece_at(int(ix.SYM[sym][sq]), piece)
    out.turn = board.turn
    return out


@pytest.mark.parametrize("name", ["KQK", "KRK"])
def test_all_eight_symmetric_images_share_an_index(name):
    spec = ix.get_spec(name)
    rng = np.random.default_rng(1)
    legal = np.flatnonzero(ix.legal_mask(spec))
    for slot in rng.choice(legal, 200, replace=False):
        board = ix.index_to_board(int(slot), spec)
        assert {ix.board_to_index(_transform(board, k), spec) for k in range(8)} == {slot}


def test_kpk_uses_left_right_mirror_only():
    board = chess.Board("8/8/8/8/2k5/8/6P1/6K1 w - - 0 1")
    mirrored = _transform(board, 1)
    assert ix.board_to_index(board, "KPK") == ix.board_to_index(mirrored, "KPK")
    flipped = _transform(board, 2)  # a vertical flip is a different KPK position
    assert ix.board_to_index(flipped, "KPK") != ix.board_to_index(board, "KPK")


def test_illegal_slots_are_marked():
    spec = ix.get_spec("KQK")
    # Kings adjacent: White a1, Black b2.
    slot, _ = ix.canonical(spec, 0, chess.A1, chess.H8, chess.B2)
    assert not ix.is_legal_index(int(slot), spec)
    # Side not to move in check: Black king on h1 attacked by the queen, White to move.
    slot, _ = ix.canonical(spec, 0, chess.A1, chess.H8, chess.H1)
    assert not ix.is_legal_index(int(slot), spec)


# --- T2: solver --------------------------------------------------------------------------------


def test_kqk_longest_mate_is_mate_in_10(solved):
    table, _ = solved["KQK"]
    summary = table.meta["summary"]["white_to_move"]
    assert summary["max_dtm_plies"] == 19
    assert summary["win"] == summary["positions"]


def test_krk_longest_mate_is_mate_in_16(solved):
    table, _ = solved["KRK"]
    summary = table.meta["summary"]["white_to_move"]
    assert summary["max_dtm_plies"] == 31
    assert summary["win"] == summary["positions"]


def test_black_to_move_is_lost_unless_the_piece_can_be_taken_or_stalemate(solved):
    for name in ("KQK", "KRK"):
        table, _ = solved[name]
        half = table.spec.half
        for slot in np.flatnonzero(table.legal[half:] & (table.wdl[half:] == 0))[:200] + half:
            board = ix.index_to_board(int(slot), name)
            captures = any(board.is_capture(m) for m in board.legal_moves)
            assert captures or board.is_stalemate()


def test_table_save_load_round_trip(solved, tmp_path):
    table, _ = solved["KRK"]
    tables.save_table(table, tmp_path)
    loaded = tables.load_table("KRK", tmp_path)
    for field in ("legal", "wdl", "dtm", "best_from", "best_to", "best_promo"):
        assert np.array_equal(getattr(table, field), getattr(loaded, field))
    assert loaded.meta["summary"] == table.meta["summary"]


# --- T3: verification -------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["KQK", "KRK"])
def test_bellman_consistency_and_best_moves(solved, name):
    table, graph = solved[name]
    assert verify.check_bellman(table, graph).ok


@pytest.mark.parametrize("name", ["KQK", "KRK"])
def test_value_iteration_matches_retrograde(solved, name):
    table, graph = solved[name]
    result = verify.check_value_iteration(table, graph, gamma=0.9)
    assert result.ok, result.details


@pytest.mark.skipif(not _has_syzygy("KQvK"), reason="Syzygy 3-piece files not present")
def test_syzygy_agrees_on_every_kqk_position(solved):
    table, _ = solved["KQK"]
    result = verify.check_syzygy(table, str(SYZYGY_DIR))
    assert result.ok, result.details


@pytest.mark.slow
@pytest.mark.skipif(verify.stockfish_path() is None, reason="Stockfish not installed")
def test_stockfish_never_finds_a_shorter_mate(solved):
    table, _ = solved["KQK"]
    result = verify.check_stockfish_dtm(table, samples=40, nodes=100_000)
    assert result.ok, result.details


# --- T3: probe API ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fen",
    [
        "k7/8/1K6/8/8/8/7Q/8 w - - 0 1",  # Qh8# (or Qb8#)
        "k7/8/1K6/8/8/8/8/7R w - - 0 1",  # Rh8#
    ],
)
def test_mate_in_one(solved, fen):
    board = chess.Board(fen)
    assert probe.probe_wdl(board) == 1
    assert probe.probe_dtm(board) == 1
    move = probe.best_move(board)
    board.push(move)
    assert board.is_checkmate()
    assert probe.probe_wdl(board) == -1
    assert probe.probe_dtm(board) == 0
    assert probe.best_move(board) is None


def test_stalemate_is_a_draw(solved):
    board = chess.Board("k7/8/1Q6/8/8/8/8/K7 b - - 0 1")
    assert board.is_stalemate()
    assert probe.probe_wdl(board) == 0
    assert probe.probe_dtm(board) == 0
    assert probe.best_move(board) is None


def test_hanging_rook_is_a_draw(solved):
    board = chess.Board("8/8/8/8/8/2k5/2R5/7K b - - 0 1")
    assert probe.probe_wdl(board) == 0
    assert probe.best_move(board) == chess.Move.from_uci("c3c2")
    # Defended by the white king, the rook can't be taken and Black is lost.
    defended = chess.Board("8/8/8/8/8/2k5/2R5/2K5 b - - 0 1")
    assert probe.probe_wdl(defended) == -1
    assert probe.probe_dtm(defended) < 0


def test_colour_flipped_position(solved):
    board = chess.Board("k7/8/1K6/8/8/8/7Q/8 w - - 0 1").mirror()  # Black has the queen
    assert board.turn == chess.BLACK
    assert probe.probe_wdl(board) == 1
    assert probe.probe_dtm(board) == 1
    move = probe.best_move(board)
    board.push(move)
    assert board.is_checkmate()


def test_best_move_is_legal_under_every_symmetry(solved):
    table, _ = solved["KRK"]
    rng = np.random.default_rng(2)
    for slot in rng.choice(np.flatnonzero(table.legal & (table.wdl != 0)), 50, replace=False):
        base = ix.index_to_board(int(slot), "KRK")
        for k in range(8):
            for board in (_transform(base, k), _transform(base, k).mirror()):
                move = probe.best_move(board)
                assert move in board.legal_moves
                dtm = probe.probe_dtm(board)
                board.push(move)
                # One ply closer to mate, seen from the other side.
                expected = -(dtm - 1) if dtm > 0 else -dtm - 1
                assert probe.probe_dtm(board) == expected


def test_uncovered_material_without_tablebase_returns_none(solved, monkeypatch):
    monkeypatch.setattr(probe, "_syzygy", lambda: None)
    board = chess.Board()
    assert probe.probe_wdl(board) is None
    assert probe.probe_dtm(board) is None
    assert probe.best_move(board) is None


def test_graph_has_no_dangling_edges(solved):
    _, graph = solved["KRK"]
    assert graph.succ.min() >= 0
    assert graph.succ.max() < graph.n + graph.num_leaves
    assert len(graph.pred) == graph.num_edges
    assert np.array_equal(np.bincount(graph.pred, minlength=graph.n), graph.outdegree)
