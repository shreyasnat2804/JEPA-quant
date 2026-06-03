"""EMA target encoder — produces ``z_target`` from the future window.

CRITICAL (CLAUDE.md rule 3): this module NEVER receives gradients. It is a
weight-space EMA copy of the context price encoder, updated *after* each
optimizer step via :meth:`update`, outside the optimizer. If gradients ever
flow through it the self-supervised signal collapses.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch import Tensor

from ..training.ema import ema_update


class TargetEncoder(nn.Module):
    def __init__(self, source: nn.Module, decay: float) -> None:
        super().__init__()
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.encoder = copy.deepcopy(source)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def update(self, source: nn.Module) -> None:
        """θ_target = α·θ_target + (1−α)·θ_context. Call once per optimizer step."""
        ema_update(self.encoder, source, self.decay)
