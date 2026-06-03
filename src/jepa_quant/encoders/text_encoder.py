"""Optional text / aux encoder producing ``z_text``.

Disabled by default (price-only JEPA). When enabled, the ``finbert`` backend
loads a frozen FinBERT, mean-pools its last hidden state over real tokens, and
projects to ``latent_dim`` through a ``ProjectionHead`` (Linear -> LayerNorm).
"""

from __future__ import annotations

from typing import Iterator, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .projection import ProjectionHead
from ..config import TextEncoderConfig


class FinBERTTextEncoder(nn.Module):
    """Frozen FinBERT + trainable projection head."""

    def __init__(self, cfg: TextEncoderConfig) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("text encoder requires `transformers`.") from exc

        self.latent_dim = cfg.latent_dim
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.finbert_name)
        self.backbone = AutoModel.from_pretrained(cfg.finbert_name)
        if cfg.freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self._frozen_backbone = cfg.freeze_backbone
        self.head = ProjectionHead(self.backbone.config.hidden_size, cfg.latent_dim, hidden_dim=cfg.latent_dim)

    def forward(self, texts: Sequence[str]) -> Tensor:
        enc = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(next(self.head.parameters()).device) for k, v in enc.items()}
        with torch.set_grad_enabled(not self._frozen_backbone):
            out = self.backbone(**enc).last_hidden_state  # [B, T, H]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.head(pooled)

    def projection_parameters(self) -> Iterator[nn.Parameter]:
        return self.head.parameters()

    def tunable_backbone_parameters(self) -> Iterator[nn.Parameter]:
        if self._frozen_backbone:
            return iter(())
        return self.backbone.parameters()


def build_text_encoder(cfg: TextEncoderConfig) -> FinBERTTextEncoder | None:
    if not cfg.enabled:
        return None
    if cfg.backend == "finbert":
        return FinBERTTextEncoder(cfg)
    raise ValueError(f"Unknown / unsupported text encoder backend: {cfg.backend!r}")
