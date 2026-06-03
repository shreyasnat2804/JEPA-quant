"""Predictor package — backend factory."""

from __future__ import annotations

from .base import Predictor
from .transformer_predictor import TransformerPredictor
from ..config import PredictorConfig


def build_predictor(cfg: PredictorConfig, in_dim: int, use_text: bool) -> Predictor:
    if cfg.backend == "lm":
        from .lm_predictor import LMPredictor  # imported lazily (heavy deps)

        return LMPredictor(cfg, in_dim, use_text)
    if cfg.backend == "transformer":
        return TransformerPredictor(cfg, in_dim, use_text)
    raise ValueError(f"Unknown predictor backend: {cfg.backend!r}")


__all__ = ["Predictor", "TransformerPredictor", "build_predictor"]
