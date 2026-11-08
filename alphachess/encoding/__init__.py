"""Board -> planes, move <-> index (4672), legal-move mask.

Owner: Samyak. Interface: docs/INTERFACES.md.
"""

from alphachess.encoding.board import (
    NUM_MOVES,
    NUM_PLANES,
    encode_board,
    index_to_move,
    legal_mask,
    move_to_index,
)

__all__ = [
    "NUM_MOVES",
    "NUM_PLANES",
    "encode_board",
    "index_to_move",
    "legal_mask",
    "move_to_index",
]
