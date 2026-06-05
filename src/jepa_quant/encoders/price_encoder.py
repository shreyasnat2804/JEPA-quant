"""Context price encoder producing ``z_price``.

Two interchangeable backends behind one interface:

* ``moirai``      — Salesforce Moirai time-series foundation model (real FM,
                    frozen backbone, requires ``uni2ts`` + HF access).
* ``transformer`` — lightweight built-in transformer that runs with no
                    downloads; used for fast iteration and CI smoke tests.

Both return ``z_price`` of shape ``[B, latent_dim]`` and both terminate in a
``ProjectionHead`` (Linear -> LayerNorm), so the rest of the system is agnostic
to which backbone is in use.

Every encoder exposes ``projection_parameters()`` and
``tunable_backbone_parameters()`` so the phased trainer can build param groups
without string-matching parameter names.
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor

from .projection import ProjectionHead
from ..config import PriceEncoderConfig


class PriceEncoder(nn.Module):
    """Common interface. ``forward(x)`` maps a price window to ``z_price``.

    Args:
        x: ``[B, L, F]`` float tensor of per-timestep features.
    Returns:
        ``[B, latent_dim]`` context latent.
    """

    latent_dim: int

    def projection_parameters(self) -> Iterator[nn.Parameter]:
        raise NotImplementedError

    def tunable_backbone_parameters(self) -> Iterator[nn.Parameter]:
        """Backbone params eligible for fine-tuning in Phase 2/3 (may be empty
        when the backbone is permanently frozen)."""
        raise NotImplementedError


class _SinusoidalPositions(nn.Module):
    """Fixed sinusoidal positional encoding added to token embeddings."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class TransformerPriceEncoder(PriceEncoder):
    """Lightweight transformer over the [B, L, F] feature window."""

    def __init__(self, cfg: PriceEncoderConfig) -> None:
        super().__init__()
        self.latent_dim = cfg.latent_dim
        self.input_proj = nn.Linear(cfg.n_features, cfg.d_model)
        self.pos = _SinusoidalPositions(cfg.d_model, max_len=cfg.context_length + 1)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        # GPT-2 scaled init: bound residual stream variance accumulation with depth.
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        with torch.no_grad():
            for enc_layer in self.backbone.layers:
                enc_layer.self_attn.out_proj.weight.mul_(scale)
                enc_layer.linear2.weight.mul_(scale)
        self.head = ProjectionHead(cfg.d_model, cfg.latent_dim, hidden_dim=cfg.latent_dim)
        self._frozen_backbone = cfg.freeze_backbone
        if cfg.freeze_backbone:
            for p in self._backbone_params():
                p.requires_grad_(False)

    def _backbone_params(self) -> Iterator[nn.Parameter]:
        yield from self.input_proj.parameters()
        yield from self.backbone.parameters()
        yield self.cls

    def forward(self, x: Tensor) -> Tensor:
        h = self.input_proj(x)  # [B, L, d]
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.pos(h)
        h = self.backbone(h)
        pooled = h[:, 0]  # CLS token
        return self.head(pooled)

    def projection_parameters(self) -> Iterator[nn.Parameter]:
        return self.head.parameters()

    def tunable_backbone_parameters(self) -> Iterator[nn.Parameter]:
        if self._frozen_backbone:
            return iter(())
        return self._backbone_params()


class MoiraiPriceEncoder(PriceEncoder):
    """Wraps a frozen Salesforce Moirai module and taps its encoder hidden
    states as the price representation.

    The chosen feature channel (default: the last column, e.g. normalized
    log-return of close) is patched and packed into Moirai's any-variate format,
    the frozen encoder is run, and per-patch hidden states are mean-pooled.

    NOTE: the packing below targets the uni2ts 1.x ``MoiraiModule`` API. If your
    installed uni2ts version drifts, this single ``_pack`` / ``forward`` method
    is the only thing to adjust — the projection head, freezing, and param
    grouping are version-independent. The CI-tested path is
    ``backend='transformer'``.
    """

    def __init__(self, cfg: PriceEncoderConfig, patch_size: int = 32, target_channel: int = -1) -> None:
        super().__init__()
        try:
            from uni2ts.model.moirai import MoiraiModule  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "backend='moirai' requires uni2ts. Install with "
                "`pip install uni2ts` and ensure HF access to "
                f"'{cfg.moirai_name}'."
            ) from exc

        self.latent_dim = cfg.latent_dim
        self.patch_size = patch_size
        self.target_channel = target_channel
        self.module = MoiraiModule.from_pretrained(cfg.moirai_name)
        backbone_dim = getattr(self.module, "d_model", None) or self.module.encoder.layers[0].self_attn.embed_dim

        # Always freeze the whole backbone, then (Phase 3) selectively re-enable
        # ONLY the top-2 encoder layers — the exact set
        # ``tunable_backbone_parameters`` exposes to the optimizer. Leaving the
        # rest of the backbone with ``requires_grad=True`` would compute and
        # store gradients that are never applied (memory/compute waste) and
        # escape grad clipping, since they sit in no optimizer param group.
        for p in self.module.parameters():
            p.requires_grad_(False)
        self._frozen_backbone = cfg.freeze_backbone
        if not cfg.freeze_backbone:
            for p in self.module.encoder.layers[-2:].parameters():
                p.requires_grad_(True)

        self.head = ProjectionHead(backbone_dim, cfg.latent_dim, hidden_dim=cfg.latent_dim)

        # Capture encoder output via a forward hook so we don't depend on the
        # module returning hidden states from its top-level forward.
        self._captured: Tensor | None = None
        self.module.encoder.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output) -> None:
        self._captured = output[0] if isinstance(output, (tuple, list)) else output

    def _pack(self, series: Tensor) -> dict[str, Tensor]:
        """Pack a univariate [B, L] series into Moirai any-variate tensors."""
        b, length = series.shape
        ps = self.patch_size
        n_patches = (length + ps - 1) // ps
        pad = n_patches * ps - length
        if pad:
            series = torch.nn.functional.pad(series, (0, pad))
        target = series.view(b, n_patches, ps)
        observed = torch.ones_like(target, dtype=torch.bool)
        if pad:
            observed[:, -1, ps - pad :] = False
        dev = series.device
        return {
            "target": target,
            "observed_mask": observed,
            "sample_id": torch.ones(b, n_patches, dtype=torch.long, device=dev),
            "time_id": torch.arange(n_patches, device=dev).expand(b, n_patches),
            "variate_id": torch.zeros(b, n_patches, dtype=torch.long, device=dev),
            "prediction_mask": torch.zeros(b, n_patches, dtype=torch.bool, device=dev),
            "patch_size": torch.full((b, n_patches), ps, dtype=torch.long, device=dev),
        }

    def forward(self, x: Tensor) -> Tensor:
        series = x[..., self.target_channel]  # [B, L]
        packed = self._pack(series)
        self._captured = None
        with torch.set_grad_enabled(not self._frozen_backbone):
            self.module(**packed)
        hidden = self._captured
        if hidden is None:  # pragma: no cover - defensive
            raise RuntimeError("Moirai encoder hook did not capture hidden states.")
        pooled = hidden.mean(dim=1)  # mean over patches -> [B, backbone_dim]
        return self.head(pooled)

    def projection_parameters(self) -> Iterator[nn.Parameter]:
        return self.head.parameters()

    def tunable_backbone_parameters(self) -> Iterator[nn.Parameter]:
        if self._frozen_backbone:
            return iter(())
        # Phase 3: only the top 2 encoder layers are ever unfrozen.
        return self.module.encoder.layers[-2:].parameters()


def build_price_encoder(cfg: PriceEncoderConfig) -> PriceEncoder:
    if cfg.backend == "moirai":
        return MoiraiPriceEncoder(cfg)
    if cfg.backend == "transformer":
        return TransformerPriceEncoder(cfg)
    raise ValueError(f"Unknown price encoder backend: {cfg.backend!r}")
