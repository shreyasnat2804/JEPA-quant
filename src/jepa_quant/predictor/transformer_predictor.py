"""Option A predictor — a lightweight transformer over prefix tokens.

``z_price`` (and optionally ``z_text``) are projected to ``d_model`` and used as
prefix tokens; a learned query token is appended. A small transformer encoder
processes the sequence and the query token's output is projected to ``z_pred``.
Cheap, runs without downloads — the CI-tested predictor path.
"""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor

from .base import Predictor
from ..encoders.projection import ProjectionHead
from ..config import PredictorConfig


class TransformerPredictor(Predictor):
    def __init__(self, cfg: PredictorConfig, in_dim: int, use_text: bool) -> None:
        super().__init__()
        self.latent_dim = cfg.latent_dim
        self.use_text = use_text
        d = cfg.d_model
        self.price_in = ProjectionHead(in_dim, d, hidden_dim=d)
        self.text_in = ProjectionHead(in_dim, d, hidden_dim=d) if use_text else None
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.query, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=4 * d,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.body = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.out = ProjectionHead(d, cfg.latent_dim, hidden_dim=cfg.latent_dim)

    def forward(self, z_price: Tensor, z_text: Tensor | None = None) -> Tensor:
        toks = [self.price_in(z_price).unsqueeze(1)]
        if self.use_text and z_text is not None and self.text_in is not None:
            toks.append(self.text_in(z_text).unsqueeze(1))
        q = self.query.expand(z_price.size(0), -1, -1)
        toks.append(q)
        seq = torch.cat(toks, dim=1)  # [B, n_prefix+1, d]
        h = self.body(seq)
        return self.out(h[:, -1])  # query token output

    def projection_parameters(self) -> Iterator[nn.Parameter]:
        mods: list[nn.Module] = [self.price_in, self.out]
        if self.text_in is not None:
            mods.append(self.text_in)
        for m in mods:
            yield from m.parameters()
        yield self.query

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        return self.body.parameters()
