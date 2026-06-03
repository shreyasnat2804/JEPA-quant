"""Projection heads with mandatory LayerNorm.

CRITICAL (see CLAUDE.md rule 1): a LayerNorm MUST follow every projection that
maps between financial-latent space and LM/embedding space. Their scale and
geometry are incompatible; without the LayerNorm early training is unstable.
This module is the single place projections are built so the rule cannot be
forgotten at a call site.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ProjectionHead(nn.Module):
    """Linear -> (GELU -> Linear)* -> LayerNorm.

    Always terminates in a LayerNorm over ``out_dim``. A single hidden layer is
    used when ``hidden_dim`` is given, otherwise it is a plain affine map.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if hidden_dim is None:
            layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
        else:
            layers = [
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            ]
        layers.append(nn.LayerNorm(out_dim))  # non-negotiable terminal LayerNorm
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
