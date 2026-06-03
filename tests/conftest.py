"""Shared fixtures for the JEPA-quant test suite.

Everything here targets the lightweight, download-free path: ``transformer``
price encoder + ``transformer`` predictor on CPU with tiny dimensions, so the
whole suite runs in seconds and needs only torch / numpy / pandas / sklearn.
The foundation-model backends (Moirai, Qwen+LoRA, FinBERT) are covered only at
their import-guard / factory branches, not end to end.
"""

from __future__ import annotations

import dataclasses as dc

import numpy as np
import pandas as pd
import pytest

from jepa_quant import JEPAConfig

# Tiny but valid geometry shared across tests.
CONTEXT_LENGTH = 8
HORIZON = 4
LATENT_DIM = 16
N_FEATURES = 6
PRICE_COLS = ("open", "high", "low", "close", "vwap", "volume")


def _make_series(n_rows: int, seed: int) -> pd.DataFrame:
    """A strictly-positive synthetic OHLCV frame indexed by timestamp ``ts``."""
    rng = np.random.default_rng(seed)
    base = 100.0 + np.cumsum(rng.normal(0, 1, size=n_rows))
    base = np.clip(base, 1.0, None)
    ts = pd.date_range("2020-01-01", periods=n_rows, freq="D", name="ts")
    return pd.DataFrame(
        {
            "open": base,
            "high": base + rng.uniform(0, 1, n_rows),
            "low": base - rng.uniform(0, 1, n_rows),
            "close": base + rng.normal(0, 0.5, n_rows),
            "vwap": base + rng.normal(0, 0.3, n_rows),
            "volume": rng.uniform(1e4, 1e5, n_rows),
        },
        index=ts,
    )


@pytest.fixture
def data_dir(tmp_path):
    """Two ticker parquets, ~100 rows each -> plenty of train/val windows."""
    d = tmp_path / "stocks"
    d.mkdir()
    for i, ticker in enumerate(("AAA", "BBB")):
        _make_series(100, seed=i).to_parquet(d / f"{ticker}.parquet")
    return str(d)


@pytest.fixture
def cfg(data_dir) -> JEPAConfig:
    """Small CPU/transformer config wired to the synthetic data dir."""
    base = JEPAConfig()
    return dc.replace(
        base,
        regularizer="vicreg",
        price_encoder=dc.replace(
            base.price_encoder,
            backend="transformer",
            n_features=N_FEATURES,
            context_length=CONTEXT_LENGTH,
            d_model=32,
            n_heads=4,
            n_layers=2,
            latent_dim=LATENT_DIM,
            freeze_backbone=False,  # so adapter params exist for the phase test
        ),
        predictor=dc.replace(
            base.predictor,
            backend="transformer",
            d_model=32,
            n_heads=4,
            n_layers=2,
            latent_dim=LATENT_DIM,
        ),
        data=dc.replace(
            base.data,
            data_dir=data_dir,
            context_length=CONTEXT_LENGTH,
            horizon=HORIZON,
            price_cols=PRICE_COLS,
        ),
        train=dc.replace(
            base.train,
            device="cpu",
            batch_size=8,
            num_workers=0,
            max_steps=6,
            warmup_proj_steps=3,
            log_every=1,
            val_every=3,
            seed=0,
        ),
    )
