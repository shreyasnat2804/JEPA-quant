"""Typed configuration for the JEPA-quant model and training loop.

Every knob lives here as an immutable dataclass so the notebook only has to
construct a config and hand it to the builders. Swapping an anti-collapse
method, an encoder backend, or a training schedule is a one-line config change,
never a code edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ----------------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------------

PriceBackend = Literal["moirai", "transformer"]
TextBackend = Literal["finbert", "transformer"]


@dataclass(frozen=True)
class PriceEncoderConfig:
    """Context price encoder. ``moirai`` is the real foundation model;
    ``transformer`` is a lightweight built-in that runs without downloads."""

    backend: PriceBackend = "moirai"
    # Salesforce Moirai checkpoint (HF). small/base/large all supported.
    moirai_name: str = "Salesforce/moirai-1.1-R-small"
    # Number of OHLCV-derived features per timestep fed to the transformer backend.
    n_features: int = 6
    context_length: int = 64  # L — context window length
    # Lightweight transformer backend hyper-params (ignored for moirai).
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    # Output latent dim AFTER the projection head (shared by both backends).
    latent_dim: int = 256
    freeze_backbone: bool = True


@dataclass(frozen=True)
class TextEncoderConfig:
    """Optional text/aux encoder. Disabled by default so the price-only JEPA
    runs out of the box; enable when conditioning text is wired up."""

    enabled: bool = False
    backend: TextBackend = "finbert"
    finbert_name: str = "ProsusAI/finbert"
    latent_dim: int = 256
    freeze_backbone: bool = True


# ----------------------------------------------------------------------------
# Predictor
# ----------------------------------------------------------------------------

PredictorBackend = Literal["lm", "transformer"]


@dataclass(frozen=True)
class PredictorConfig:
    """``lm`` = Qwen2.5 + LoRA prefix-token predictor (Option B, preferred).
    ``transformer`` = lightweight transformer predictor (Option A)."""

    backend: PredictorBackend = "lm"
    lm_name: str = "Qwen/Qwen2.5-1.5B"
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Lightweight transformer backend (ignored for lm).
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    latent_dim: int = 256  # must match target_encoder latent dim


# ----------------------------------------------------------------------------
# EMA target encoder
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class EMAConfig:
    decay: float = 0.998  # alpha in [0.996, 0.999]


# ----------------------------------------------------------------------------
# Regularization (anti-collapse) — pluggable
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class VICRegConfig:
    """V + C only. The invariance term is intentionally absent — it conflicts
    with the JEPA temporal objective (context/target are time slices, not
    augmentations)."""

    name: str = "vicreg"
    lambda_v: float = 25.0  # variance weight
    lambda_c: float = 1.0  # covariance weight
    gamma: float = 1.0  # target std (hinge threshold)
    eps: float = 1e-4


@dataclass(frozen=True)
class CodebookConfig:
    """Soft codebook bottleneck — alternative anti-collapse method, registered
    alongside VICReg so the two can be compared by name."""

    name: str = "codebook"
    n_codes: int = 256
    lambda_commit: float = 1.0
    temperature: float = 1.0
    kmeans_init: bool = True


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class DataConfig:
    data_dir: str = "data/raw/stocks"  # overridden by the notebook to the Drive path
    context_length: int = 64  # L
    horizon: int = 16  # H — future window length
    stride: int = 1
    # Per-ticker chronological split fractions.
    val_fraction: float = 0.15
    # Feature columns pulled from each parquet (after deriving log-return etc.).
    price_cols: tuple[str, ...] = ("open", "high", "low", "close", "vwap", "volume")
    normalize: bool = True


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    # VICReg covariance estimates are noisy below 256 — use grad accumulation
    # if GPU memory forces a smaller micro-batch.
    batch_size: int = 256
    grad_accum_steps: int = 1
    max_steps: int = 2000
    warmup_proj_steps: int = 500  # Phase 1: projections only, predictor adapters frozen
    lr_proj: float = 1e-4
    lr_adapter: float = 3e-5  # LoRA / encoder fine-tune lr in Phase 2
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    log_every: int = 25
    val_every: int = 250
    seed: int = 42
    device: str = "cuda"  # notebook resolves to cpu if unavailable
    num_workers: int = 2


# ----------------------------------------------------------------------------
# Aggregate
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class JEPAConfig:
    price_encoder: PriceEncoderConfig = field(default_factory=PriceEncoderConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    # Which regularizer to use, by registry name: "vicreg" or "codebook".
    regularizer: str = "vicreg"
    vicreg: VICRegConfig = field(default_factory=VICRegConfig)
    codebook: CodebookConfig = field(default_factory=CodebookConfig)
