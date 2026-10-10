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


# --- T4: KPK (solving takes ~15 s, so these are slow tests) -----------------------------------


@pytest.fixture(scope="session")
def kpk(solved):
    table, graph = tables.solve_table("KPK", {name: t for name, (t, _) in solved.items()})
    tables.save_table(table, tables.endgame_dir())
    probe.get_table.cache_clear()
    return table, graph


@pytest.mark.slow
def test_kpk_longest_mate_and_draw_share(kpk):
    table, _ = kpk
    white = table.meta["summary"]["white_to_move"]
    assert white["max_dtm_plies"] == 55  # mate in 28, including the KQK phase
    assert 0 < white["draw"] < white["win"]
    assert white["loss"] == 0


@pytest.mark.slow
@pytest.mark.parametrize(
    ("fen", "wdl"),
    [
        ("3k4/8/3K4/8/3P4/8/8/8 b - - 0 1", -1),  # king on a key square: win for White
        ("8/8/4k3/8/4K3/4P3/8/8 w - - 0 1", 0),  # Black has the opposition: draw
        ("8/8/4k3/8/4K3/4P3/8/8 b - - 0 1", -1),  # White has the opposition: win
        ("7k/8/6K1/7P/8/8/8/8 w - - 0 1", 0),  # rook pawn, defender in the corner: draw
        ("4k3/4P3/4K3/8/8/8/8/8 b - - 0 1", 0),  # stalemate
        ("8/5P2/8/8/8/8/k7/4K3 w - - 0 1", 1),  # promotes into the KQK table
    ],
)
def test_kpk_textbook_positions(kpk, fen, wdl):
    assert probe.probe_wdl(chess.Board(fen)) == wdl


@pytest.mark.slow
def test_kpk_promotion_and_colour_flip(kpk):
    board = chess.Board("8/8/8/8/8/7K/k4p2/8 b - - 0 1")  # Black pawn, Black to move
    assert probe.probe_wdl(board) == 1
    move = probe.best_move(board)
    assert move == chess.Move.from_uci("f2f1q")
    dtm = probe.probe_dtm(board)
    board.push(move)
    assert probe.probe_dtm(board) == -(dtm - 1)


@pytest.mark.slow
def test_kpk_bellman_and_value_iteration(kpk):
    table, graph = kpk
    assert verify.check_bellman(table, graph).ok
    assert verify.check_value_iteration(table, graph, gamma=0.9).ok


@pytest.mark.slow
@pytest.mark.skipif(not _has_syzygy("KPvK"), reason="Syzygy 3-piece files not present")
def test_syzygy_agrees_on_every_kpk_position(kpk):
    table, _ = kpk
    result = verify.check_syzygy(table, str(SYZYGY_DIR))
    assert result.ok, result.details


# --- T5: value iteration vs RTDP --------------------------------------------------------------


def _short_starts(table, graph, max_dtm: int, count: int) -> np.ndarray:
    from alphachess.endgames.index import decode

    stm = decode(graph.spec, graph.slots)[0]
    dtm = table.dtm[graph.slots]
    pool = np.flatnonzero((stm == 0) & (dtm > 0) & (dtm <= max_dtm))
    return np.random.default_rng(0).choice(pool, count, replace=False)


def _exact(table, graph, gamma=0.9):
    from alphachess.endgames.solve import best_edges, gamma_values

    wdl = np.concatenate([table.wdl[graph.slots], graph.leaf_wdl])
    dtm = np.concatenate([table.dtm[graph.slots], graph.leaf_dtm])
    return wdl, dtm, gamma_values(wdl, dtm, gamma), best_edges(graph, wdl, dtm)


def test_bounded_rtdp_solves_short_starts_with_exact_values(solved):
    from alphachess.endgames.rtdp import BoundedRTDP

    table, graph = solved["KQK"]
    _, dtm, values, defence = _exact(table, graph)
    starts = _short_starts(table, graph, 3, 5)
    rtdp = BoundedRTDP(graph, values, defence, gamma=0.9)
    result = rtdp.run(starts, dtm, max_passes=3000)
    assert result.converged
    for s in starts:
        assert rtdp.bounds(int(s))[1] == pytest.approx(values[s])
    assert result.touched < graph.n  # only part of the table was needed


def test_plain_rtdp_backup_matches_bellman(solved):
    from alphachess.endgames.rtdp import RTDP

    table, graph = solved["KQK"]
    _, dtm, values, defence = _exact(table, graph)
    start = int(_short_starts(table, graph, 1, 1)[0])  # mate in 1
    rtdp = RTDP(graph, values, defence, gamma=0.9)
    rtdp.trial(start)
    assert rtdp.V[start] == pytest.approx(0.9)
    assert rtdp.solved(start, int(dtm[start]))


# --- T7: harness hooks ------------------------------------------------------------------------


def test_dp_evaluator_puts_all_mass_on_the_table_move(solved):
    from alphachess.endgames.hooks import DPEvaluator

    mate_in_one = chess.Board("k7/8/1K6/8/8/8/8/7R w - - 0 1")
    priors, values = DPEvaluator()([mate_in_one, chess.Board()])
    assert priors[0] == {probe.best_move(mate_in_one): 1.0}
    assert values[0] == 1.0
    assert len(priors[1]) == 20 and values[1] == 0.0  # uniform fallback
    _, gamma_values = DPEvaluator(gamma=0.9)([mate_in_one])
    assert gamma_values[0] == pytest.approx(0.9)


def test_tablebase_player_and_fallback(solved):
    from alphachess.endgames.hooks import TablebasePlayer
    from alphachess.eval.player import RandomPlayer

    player = TablebasePlayer(RandomPlayer(seed=1))
    board = chess.Board("k7/8/1K6/8/8/8/8/7R w - - 0 1")
    assert player.select_move(board).move == probe.best_move(board)
    start = chess.Board()
    assert player.select_move(start).move in start.legal_moves


def test_exact_reference_analyser_scores(solved):
    from alphachess.endgames.hooks import WIN_SCORE, exact_reference_analyser

    board = chess.Board("k7/8/1K6/8/8/8/8/7R w - - 0 1")
    best, scores = exact_reference_analyser(board)
    assert scores[best] == WIN_SCORE - 1
    board.push(best)
    assert board.is_checkmate()
    hanging = chess.Board("8/8/8/8/8/2k5/2R5/7K b - - 0 1")
    best, scores = exact_reference_analyser(hanging)
    assert best == chess.Move.from_uci("c3c2") and scores[best] == 0
    assert all(v < 0 for m, v in scores.items() if m != best)


def test_suite_with_evaluate_positions(solved):
    from alphachess.endgames.hooks import TablebasePlayer, exact_reference_analyser, load_suite
    from alphachess.eval.metrics import evaluate_positions
    from alphachess.eval.player import RandomPlayer

    suite = load_suite(repo_path("data/endgames/suite.fen"))
    assert {k: len(v) for k, v in suite.items()} == {"KQK": 100, "KRK": 100, "KPK": 100}
    boards = suite["KRK"][:15]
    summary = evaluate_positions(TablebasePlayer(RandomPlayer()), boards, exact_reference_analyser)
    assert summary.top_move_agreement == 1.0
    assert summary.mean_centipawn_loss == 0.0


# --- T6: approximation study (needs PyTorch) ---------------------------------------------------


def test_approx_metrics_are_perfect_for_exact_values(solved):
    from alphachess.endgames import approx

    table, _ = solved["KRK"]
    data = approx.load_krk(table=table)
    assert data.inputs["onehot"].shape == (data.graph.n, 193)
    assert len(data.train) + len(data.test) == data.graph.n
    starts = approx.mate_start_positions(data, 50, 0)
    res = approx.evaluate(data, data.value[: data.graph.n].copy(), starts)
    assert res["mse"] == 0.0
    for key in ("sign_acc", "optimal_move", "keeps_result", "mate_rate"):
        assert res[key] == 1.0, key


def test_fitted_value_iteration_smoke(solved):
    pytest.importorskip("torch")
    from alphachess.endgames import approx

    table, _ = solved["KRK"]
    data = approx.load_krk(table=table)
    log = []
    model = approx.train_fitted_vi(
        data, "features", 16, rounds=2, epochs_per_round=1, seed=0, log=log
    )
    assert len(log) == 2
    pred = approx.predict(model, data.inputs["features"])
    assert pred.shape == (data.graph.n,) and np.all(np.abs(pred) <= 1)
