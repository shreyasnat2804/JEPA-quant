"""Anti-collapse regularizer interface + registry.

This is the seam that makes anti-collapse methods swappable and comparable. A
regularizer does two things in the training step:

1. ``transform(z_price) -> z_used`` — optionally reshape the context latent
   before it enters the predictor (identity for VICReg; quantized for the
   codebook).
2. ``loss(z_price, z_used) -> (loss, metrics)`` — the anti-collapse penalty,
   already scaled by its lambda weights, plus a metrics dict for logging.

New methods register themselves by name with :func:`register_regularizer`, so
the notebook compares them by changing ``cfg.regularizer`` only.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple, TYPE_CHECKING

import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from ..config import JEPAConfig

Metrics = Dict[str, float]


class Regularizer(nn.Module):
    name: str = "base"

    def transform(self, z_price: Tensor) -> Tensor:
        """Latent actually fed to the predictor. Identity by default."""
        return z_price

    def loss(self, z_price: Tensor, z_used: Tensor) -> Tuple[Tensor, Metrics]:
        raise NotImplementedError

    def maybe_kmeans_init(self, embeddings: Tensor) -> None:
        """Hook for methods that need data-driven init (e.g. codebook)."""
        return None


_BUILDERS: Dict[str, "Callable[[JEPAConfig], Regularizer]"] = {}


def register_regularizer(name: str) -> "Callable[[Callable[[JEPAConfig], Regularizer]], Callable[[JEPAConfig], Regularizer]]":
    def deco(builder: "Callable[[JEPAConfig], Regularizer]") -> "Callable[[JEPAConfig], Regularizer]":
        if name in _BUILDERS:
            raise ValueError(f"regularizer {name!r} already registered")
        _BUILDERS[name] = builder
        return builder

    return deco


def build_regularizer(cfg: "JEPAConfig") -> Regularizer:
    if cfg.regularizer not in _BUILDERS:
        raise ValueError(
            f"Unknown regularizer {cfg.regularizer!r}. "
            f"Registered: {sorted(_BUILDERS)}"
        )
    return _BUILDERS[cfg.regularizer](cfg)


def registered_regularizers() -> list[str]:
    return sorted(_BUILDERS)
