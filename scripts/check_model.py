"""Check a trained policy-value checkpoint: held-out test split, known positions, games.

    python -m scripts.check_model --checkpoint models/sl_jan-feb2024_6x64.pt
    python -m scripts.check_model --checkpoint <ckpt> --months 2024-01,2024-02 --split test

Three checks, each optional:

1. ``--months``: policy accuracy / value error on encoded ``<split>-*.npz`` shards (default the
   ``test`` split, which training never looks at). Needs the encoded data (HPC).
2. Known positions: top moves and value for openings, mates in one, a hanging queen.
3. ``--games N``: the raw policy (no search, greedy) plays N games against a random mover.

Runs on a laptop CPU for checks 2 and 3; check 1 is fastest on a GPU.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import chess
import torch

from alphachess.common.config import load_config, repo_path
from alphachess.encoding.board import encode_board, move_to_index
from alphachess.model.evaluator import load_model
from alphachess.train.supervised import evaluate, format_metrics, pick_device

POSITIONS = {
    "start (White to move)": chess.STARTING_FEN,
    "after 1.e4 (Black)": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "French 1.e4 e6 2.d4 d5 (White)": "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
    "Ruy Lopez after 4.Ba4 (Black)": "r1bqkbnr/1ppp1ppp/p1n5/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 4",
    "mate in 1: Rd8# (White)": "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "mate in 1: Qxf7# (White)": "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "hanging queen: Bxg4 (Black)": "rnbqkbnr/ppp1pppp/8/3p4/4P1Q1/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2",
    "White a queen up (White to move)": "4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1",
    "White a queen up (Black to move)": "3k4/8/8/8/8/8/3Q4/4K3 b - - 0 1",
}


def policy(net, board: chess.Board, device: str):
    """Legal moves, their probabilities (masked softmax) and the value, side to move's view."""
    x = torch.from_numpy(encode_board(board))[None].to(device)
    with torch.no_grad():
        logits, value = net(x)
    moves = list(board.legal_moves)
    idx = torch.tensor([move_to_index(m, board) for m in moves], device=device)
    return moves, torch.softmax(logits[0, idx].float(), 0).cpu(), float(value[0])


def check_positions(net, device: str) -> None:
    print("\n== known positions (top moves, value for the side to move)")
    for name, fen in POSITIONS.items():
        board = chess.Board(fen)
        moves, probs, value = policy(net, board, device)
        top = sorted(zip(moves, probs.tolist(), strict=True), key=lambda t: -t[1])[:4]
        shown = "  ".join(f"{board.san(m)} {p:.0%}" for m, p in top)
        print(f"{name:34s} v={value:+.2f} | {shown}")


def play_vs_random(net, device: str, games: int, max_plies: int = 300, seed: int = 0) -> dict:
    rng = random.Random(seed)
    tally = {"wins": 0, "draws": 0, "losses": 0, "unfinished": 0}
    for g in range(games):
        board, net_color = chess.Board(), chess.WHITE if g % 2 == 0 else chess.BLACK
        while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
            if board.turn == net_color:
                moves, probs, _ = policy(net, board, device)
                board.push(moves[int(probs.argmax())])
            else:
                board.push(rng.choice(list(board.legal_moves)))
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            tally["unfinished"] += 1
        elif outcome.winner is None:
            tally["draws"] += 1
        else:
            tally["wins" if outcome.winner == net_color else "losses"] += 1
    print(f"\n== {games} games, greedy policy (no search) vs random mover: {tally}")
    return tally


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--months", default=None, help="e.g. 2024-01,2024-02 (encoded data)")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = str(pick_device(args.device))
    net = load_model(args.checkpoint, device)
    print(f"checkpoint {args.checkpoint} on {device}")
    report: dict = {"checkpoint": args.checkpoint}

    if args.months:
        base = repo_path(load_config()["paths"]["lichess_dir"]) / "encoded"
        shards = sorted(str(p) for m in args.months.split(",")
                        for p in (base / m.strip()).glob(f"{args.split}-*.npz"))
        if not shards:
            raise SystemExit(f"no {args.split}-*.npz shards for {args.months} under {base}")
        metrics = evaluate(net, shards, torch.device(device))
        print(f"\n== {args.split} split ({len(shards)} shards): {format_metrics(metrics)}")
        report[f"{args.split}_split"] = metrics

    net.eval()  # BatchNorm must use its training statistics, not a 1-position batch
    check_positions(net, device)
    if args.games:
        report["games_vs_random"] = play_vs_random(net, device, args.games)

    out = Path(args.checkpoint).with_suffix(".check.json")
    try:
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nreport -> {out}")
    except OSError:
        pass


if __name__ == "__main__":
    main()
