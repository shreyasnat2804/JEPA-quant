"""JEPA-quant — modular multimodal JEPA for financial time series.

Public API used by the training notebooks::

    from jepa_quant import JEPAConfig, build_dataloaders, build_components, JEPATrainer

Anti-collapse methods are pluggable: set ``cfg.regularizer`` to any registered
name (``registered_regularizers()`` lists them) — VICReg and soft codebook ship
built-in, and new methods register themselves without touching the trainer.
"""

from __future__ import annotations

from .config import (
    CodebookConfig,
    DataConfig,
    EMAConfig,
    JEPAConfig,
    PredictorConfig,
    PriceEncoderConfig,
    TextEncoderConfig,
    TrainConfig,
    VICRegConfig,
)
from .data import build_dataloaders
from .regularization import build_regularizer, registered_regularizers
from .training import JEPATrainer, build_components

__all__ = [
    "JEPAConfig",
    "PriceEncoderConfig",
    "TextEncoderConfig",
    "PredictorConfig",
    "EMAConfig",
    "DataConfig",
    "TrainConfig",
    "VICRegConfig",
    "CodebookConfig",
    "build_dataloaders",
    "build_components",
    "JEPATrainer",
    "build_regularizer",
    "registered_regularizers",
]
