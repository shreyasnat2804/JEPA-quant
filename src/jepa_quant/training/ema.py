"""EMA weight update — the canonical formula used by the target encoder.

θ_target = decay·θ_target + (1−decay)·θ_source

Runs under ``no_grad`` and is called *after* the optimizer step (CLAUDE.md
rule 3): the target encoder must never receive gradients.
"""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def ema_update(target: nn.Module, source: nn.Module, decay: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.mul_(decay).add_(sp.detach(), alpha=1.0 - decay)
    for tb, sb in zip(target.buffers(), source.buffers()):
        tb.copy_(sb)
