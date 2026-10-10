"""Adapters between policy-value models and the MCTS evaluator contract."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import chess
import numpy as np


class TorchEvaluator:
    """Batch a PyTorch model without depending on a particular model class.

    The encoder and move-index function are injected so this adapter stays stable
    while the shared encoding/model boundary is being implemented elsewhere.
    """

    def __init__(
        self,
        model: Any,
        encode_board: Callable[[chess.Board], np.ndarray],
        move_to_index: Callable[[chess.Move, chess.Board], int],
        *,
        device: str = "cpu",
    ) -> None:
        import torch

        self.torch = torch
        self.model = model.to(device)
        self.model.eval()
        self.encode_board = encode_board
        self.move_to_index = move_to_index
        self.device = device

    def __call__(
        self, boards: list[chess.Board]
    ) -> tuple[list[dict[chess.Move, float]], np.ndarray]:
        torch = self.torch
        planes = np.stack([self.encode_board(board) for board in boards])
        inputs = torch.from_numpy(planes).to(self.device)
        with torch.inference_mode():
            logits, values = self.model(inputs)
        logits_np = logits.detach().cpu().numpy()
        values_np = values.detach().cpu().numpy().reshape(-1).astype(np.float32)

        priors_batch = []
        for board, row in zip(boards, logits_np, strict=True):
            moves = list(board.legal_moves)
            indices = np.array([self.move_to_index(move, board) for move in moves], dtype=np.int64)
            legal_logits = np.asarray(row[indices], dtype=np.float64)
            legal_logits -= np.max(legal_logits)
            probabilities = np.exp(legal_logits)
            probabilities /= probabilities.sum()
            priors_batch.append(
                {
                    move: float(probability)
                    for move, probability in zip(moves, probabilities, strict=True)
                }
            )
        return priors_batch, values_np
