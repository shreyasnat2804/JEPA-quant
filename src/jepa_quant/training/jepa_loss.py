"""JEPA prediction loss: predicted latent vs EMA-target latent.

``L_jepa = 1 - cos(z_pred, z_target)`` by default (or normalized L2). The
regularization terms live in the regularizer modules and are added by the
trainer; this file is only the prediction objective.
"""

from __future__ import annotations

from typing import Literal

import torch.nn.functional as F
from torch import Tensor

JepaKind = Literal["cosine", "l2"]


def jepa_loss(z_pred: Tensor, z_target: Tensor, kind: JepaKind = "cosine") -> Tensor:
    if kind == "cosine":
        return (1.0 - F.cosine_similarity(z_pred, z_target, dim=-1)).mean()
    if kind == "l2":
        return F.mse_loss(z_pred, z_target)  # mean over batch & dim
    raise ValueError(f"Unknown jepa loss kind: {kind!r}")
