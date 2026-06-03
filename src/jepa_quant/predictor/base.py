"""Predictor interface.

A predictor maps the context latent ``z_price`` (and optionally ``z_text``) to
``z_pred``, the prediction of the EMA target's future latent. All backends
expose two parameter groups so the phased trainer can warm up projections
before enabling adapters (CLAUDE.md rule 6):

* ``projection_parameters()`` — input projections, query token, output head.
  Trained from step 1.
* ``adapter_parameters()``     — LoRA adapters (LM) or transformer body
  (Option A). Enabled only in Phase 2.
"""

from __future__ import annotations

from typing import Iterator

import torch.nn as nn
from torch import Tensor


class Predictor(nn.Module):
    latent_dim: int

    def forward(self, z_price: Tensor, z_text: Tensor | None = None) -> Tensor:
        raise NotImplementedError

    def projection_parameters(self) -> Iterator[nn.Parameter]:
        raise NotImplementedError

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        raise NotImplementedError
