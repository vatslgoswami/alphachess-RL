"""UCI wrapper, match harness, ratings, and position metrics."""

from .arena import GameRecord, ResultWriter, play_game, play_opening_pair, run_paired_match
from .openings import Opening, load_fens, load_pgn
from .player import MCTSPlayer, MoveDecision, Player, PolicyPlayer, RandomPlayer
from .rating import RatingSummary, score_to_elo, summarise_results
from .uci import UCIEngine, run_uci_loop

__all__ = [
    "GameRecord",
    "MCTSPlayer",
    "MoveDecision",
    "Opening",
    "Player",
    "PolicyPlayer",
    "RandomPlayer",
    "RatingSummary",
    "ResultWriter",
    "UCIEngine",
    "load_fens",
    "load_pgn",
    "play_game",
    "play_opening_pair",
    "run_paired_match",
    "run_uci_loop",
    "score_to_elo",
    "summarise_results",
]
