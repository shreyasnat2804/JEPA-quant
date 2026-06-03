"""Option B predictor (preferred) — Qwen2.5-1.5B + LoRA, prefix-token wiring.

``z_price`` and ``z_text`` are each projected into the LM embedding dim and
prepended as prefix tokens. A learned query token is appended; the LM's final
hidden state at the query position is projected to ``z_pred``.

The base model is frozen; only LoRA adapters are trainable, and they are
enabled only in Phase 2 (the trainer toggles ``adapter_parameters`` into the
optimizer after the projection warmup).
"""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor

from .base import Predictor
from ..encoders.projection import ProjectionHead
from ..config import PredictorConfig

_QWEN_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


class LMPredictor(Predictor):
    def __init__(self, cfg: PredictorConfig, in_dim: int, use_text: bool) -> None:
        super().__init__()
        try:
            from transformers import AutoModel  # type: ignore
            from peft import LoraConfig, get_peft_model  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "backend='lm' requires `transformers` and `peft`. "
                "Install them and ensure HF access to "
                f"'{cfg.lm_name}'."
            ) from exc

        self.latent_dim = cfg.latent_dim
        self.use_text = use_text

        # Qwen2.5 ships in bf16; load it in its native precision instead of
        # upcasting to fp32 (~halves weight memory, faster matmuls, no quality
        # loss — fp32 just zero-pads the mantissa). Projection heads stay fp32
        # for training stability; the forward bridges dtypes at the LM boundary.
        base = AutoModel.from_pretrained(cfg.lm_name, torch_dtype=torch.bfloat16)
        self._lm_dtype = next(base.parameters()).dtype
        for p in base.parameters():
            p.requires_grad_(False)
        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=_QWEN_LORA_TARGETS,
            bias="none",
        )
        self.lm = get_peft_model(base, lora)
        h = base.config.hidden_size

        self.price_in = ProjectionHead(in_dim, h, hidden_dim=h)
        self.text_in = ProjectionHead(in_dim, h, hidden_dim=h) if use_text else None
        self.query = nn.Parameter(torch.zeros(1, 1, h))
        nn.init.normal_(self.query, std=0.02)
        self.out = ProjectionHead(h, cfg.latent_dim, hidden_dim=cfg.latent_dim)

    def forward(self, z_price: Tensor, z_text: Tensor | None = None) -> Tensor:
        toks = [self.price_in(z_price).unsqueeze(1)]
        if self.use_text and z_text is not None and self.text_in is not None:
            toks.append(self.text_in(z_text).unsqueeze(1))
        q = self.query.expand(z_price.size(0), -1, -1)
        toks.append(q)
        embeds = torch.cat(toks, dim=1)  # [B, T, H] — fp32 from the projections
        attn = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
        # Bridge fp32 projections <-> bf16 LM: cast embeds into the LM dtype,
        # cast the query hidden state back out for the fp32 output projection.
        out = self.lm(
            inputs_embeds=embeds.to(self._lm_dtype), attention_mask=attn
        ).last_hidden_state
        return self.out(out[:, -1].to(embeds.dtype))  # query token hidden state

    def _lora_parameters(self) -> Iterator[nn.Parameter]:
        for n, p in self.lm.named_parameters():
            if "lora_" in n and p.requires_grad:
                yield p

    def projection_parameters(self) -> Iterator[nn.Parameter]:
        mods: list[nn.Module] = [self.price_in, self.out]
        if self.text_in is not None:
            mods.append(self.text_in)
        for m in mods:
            yield from m.parameters()
        yield self.query

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        return self._lora_parameters()
