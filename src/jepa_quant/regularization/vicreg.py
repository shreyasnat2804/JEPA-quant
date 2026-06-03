"""VICReg anti-collapse — variance + covariance only.

The invariance term is intentionally omitted (CLAUDE.md rule 2): it conflicts
with the JEPA temporal objective because context and target are different time
slices, not augmentations.

Applied to ``z_price`` only (never ``z_pred`` or ``z_text``). Covariance
estimates are noisy below batch size 256 — keep the effective batch large
(CLAUDE.md rule 4).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import Metrics, Regularizer, register_regularizer
from ..config import JEPAConfig, VICRegConfig


def variance_loss(z: Tensor, gamma: float, eps: float) -> Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


def covariance_loss(z: Tensor) -> Tensor:
    n, d = z.shape
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return off_diag.pow(2).sum() / d


class VICReg(Regularizer):
    name = "vicreg"

    def __init__(self, cfg: VICRegConfig) -> None:
        super().__init__()
        self.cfg = cfg

    # transform is identity (inherited) — z_price feeds the predictor unchanged.

    def loss(self, z_price: Tensor, z_used: Tensor) -> Tuple[Tensor, Metrics]:
        v = variance_loss(z_price, self.cfg.gamma, self.cfg.eps)
        c = covariance_loss(z_price)
        total = self.cfg.lambda_v * v + self.cfg.lambda_c * c
        metrics: Metrics = {
            "reg/v": float(v.detach()),
            "reg/c": float(c.detach()),
            "reg/z_std_mean": float(z_price.detach().std(dim=0).mean()),
        }
        return total, metrics


@register_regularizer("vicreg")
def _build(cfg: JEPAConfig) -> VICReg:
    return VICReg(cfg.vicreg)
