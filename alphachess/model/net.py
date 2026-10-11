"""Policy-value ResNet. Interface: docs/INTERFACES.md §3.

Shape of the network (defaults ``blocks=6, channels=64`` from ``config.yaml``)::

    planes (B, 19, 8, 8)
      -> stem: 3x3 conv -> BN -> ReLU                      (B, channels, 8, 8)
      -> `blocks` residual blocks:
             x + BN(conv3x3(ReLU(BN(conv3x3(x)))))  then ReLU
      -> policy head: 3x3 conv -> BN -> ReLU -> 1x1 conv to 73 planes
             (B, 73, 8, 8) -> permute to (B, 8, 8, 73) -> flatten -> (B, 4672) logits
      -> value head:  1x1 conv to 1 plane -> BN -> ReLU -> FC 64 -> 128 -> ReLU -> FC 1 -> tanh

The policy head is fully convolutional: plane ``t`` at square ``s`` is the logit of move type
``t`` from square ``s``. Permuting to ``(rank, file, type)`` before flattening makes the flat
position ``s * 73 + t``, which is exactly ``move_to_index`` (INTERFACES §2). Illegal moves are
masked outside the network.
"""

from __future__ import annotations

import torch
from torch import nn

from alphachess.encoding.board import NUM_MOVES, NUM_PLANES

NUM_MOVE_TYPES = 73


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(x + out)


class PolicyValueNet(nn.Module):
    """``policy_logits, value = net(x)`` with x of shape (B, C, 8, 8)."""

    def __init__(
        self,
        blocks: int = 6,
        channels: int = 64,
        in_planes: int = NUM_PLANES,
        value_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.channels = channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, NUM_MOVE_TYPES, 1),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.tower(self.stem(x))
        policy = self.policy_head(h).permute(0, 2, 3, 1).reshape(x.shape[0], NUM_MOVES)
        value = self.value_head(h).squeeze(-1)
        return policy, value


def build_model(cfg: dict) -> PolicyValueNet:
    """Build the network from the ``model`` section of ``config.yaml``."""
    model_cfg = cfg.get("model", {})
    if model_cfg.get("value_head", "tanh") != "tanh":
        raise NotImplementedError("only the tanh value head is implemented (INTERFACES §3)")
    return PolicyValueNet(blocks=model_cfg.get("blocks", 6), channels=model_cfg.get("channels", 64))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
