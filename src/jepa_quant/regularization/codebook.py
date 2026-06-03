"""Soft codebook anti-collapse — alternative to VICReg.

``z_price`` is softly quantized against a learned codebook; the quantized latent
``z_codebook`` feeds the predictor, and a commitment loss pulls ``z_price``
toward its assigned codes:

    z_codebook = Σ_k softmax(−||z − e_k||² / τ)_k · e_k
    L_commit   = λ_commit · ||z_price − sg(z_codebook)||²

Initialise the codebook with k-means on a frozen forward pass (CLAUDE.md rule 5)
via :meth:`maybe_kmeans_init`; random init causes dead codes. Codebook
utilisation is logged so a collapsing codebook is visible early.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import Metrics, Regularizer, register_regularizer
from ..config import CodebookConfig, JEPAConfig


class SoftCodebook(Regularizer):
    name = "codebook"

    def __init__(self, cfg: CodebookConfig, latent_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.codes = nn.Parameter(torch.randn(cfg.n_codes, latent_dim))
        self._initialised = not cfg.kmeans_init  # if kmeans requested, wait for data

    def _assign(self, z: Tensor) -> Tensor:
        # squared distances [B, K]
        dist = torch.cdist(z, self.codes) ** 2
        return F.softmax(-dist / self.cfg.temperature, dim=1)

    def transform(self, z_price: Tensor) -> Tensor:
        weights = self._assign(z_price)  # [B, K]
        return weights @ self.codes  # [B, D] soft-quantized

    def loss(self, z_price: Tensor, z_used: Tensor) -> Tuple[Tensor, Metrics]:
        commit = self.cfg.lambda_commit * F.mse_loss(z_price, z_used.detach())
        with torch.no_grad():
            hard = self._assign(z_price).argmax(dim=1)
            util = hard.unique().numel() / self.cfg.n_codes
        metrics: Metrics = {
            "reg/commit": float(commit.detach()),
            "reg/codebook_util": float(util),
        }
        return commit, metrics

    @torch.no_grad()
    def maybe_kmeans_init(self, embeddings: Tensor) -> None:
        if self._initialised:
            return
        try:
            from sklearn.cluster import KMeans  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("codebook k-means init requires scikit-learn.") from exc
        km = KMeans(n_clusters=self.cfg.n_codes, n_init=10)
        km.fit(embeddings.detach().cpu().numpy())
        centroids = torch.as_tensor(km.cluster_centers_, dtype=self.codes.dtype, device=self.codes.device)
        self.codes.copy_(centroids)
        self._initialised = True


@register_regularizer("codebook")
def _build(cfg: JEPAConfig) -> SoftCodebook:
    return SoftCodebook(cfg.codebook, latent_dim=cfg.price_encoder.latent_dim)
