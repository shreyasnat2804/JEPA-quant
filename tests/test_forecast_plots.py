"""Smoke + consistency tests for the nb05 diagnosis helpers (forecast_plots).

Runs on the synthetic-parquet ``cfg`` fixture (real loader / PriceWindowDataset)
on CPU — no TimesFM, no downloads. The real TimesFM paths (``timesfm_forecast_fn``,
``instrument_timesfm_shapes``) are exercised only via stubs / their import-guard,
matching conftest's policy for foundation-model backends.

Key guard: ``probe_regression_predictions`` must report the SAME val R² / best
alpha as ``probe_regression_features`` for the same encode_fn — proving the
``return_val_predictions`` branch added to ``_fit_regression`` did not change the
scored numbers, and that the returned (y_true, y_pred) are the points behind that R².
"""

from __future__ import annotations

import torch

from jepa_quant import build_components
from jepa_quant.eval.backbone_features import raw_feature_encode_fn, raw_sigma_encode_fn
from jepa_quant.eval.linear_probe import CLOSE_IDX, probe_regression_features
from jepa_quant.eval.forecast_plots import (
    collect_timesfm_forecasts,
    probe_regression_predictions,
)

_PROBE_ARGS = dict(n_train_batches=4, n_val_batches=2, seed=42, device="cpu")


def _eval_encoder(cfg):
    torch.manual_seed(0)
    components = build_components(cfg)
    components.price_encoder.to("cpu").eval()
    return components.price_encoder


def test_predictions_match_features_val_r2(cfg):
    """val_r2/best_alpha from the prediction collector == the scored probe."""
    encoder = _eval_encoder(cfg)

    for target_kind in ("future_return", "future_volatility"):
        feat = probe_regression_features(
            encoder, cfg, target_kind=target_kind, source_label="x", **_PROBE_ARGS
        )
        preds = probe_regression_predictions(
            encoder, cfg, target_kind=target_kind, source_label="x", **_PROBE_ARGS
        )
        assert preds["val_r2"] == feat["val_r2"]
        assert preds["best_alpha"] == feat["best_alpha"]
        assert preds["train_r2"] == feat["train_r2_at_best_alpha"]


def test_predictions_arrays_reproduce_reported_r2(cfg):
    """Recomputing R² from the returned (y_true, y_pred) matches the reported val_r2."""
    encoder = _eval_encoder(cfg)
    preds = probe_regression_predictions(
        encoder, cfg, target_kind="future_volatility", **_PROBE_ARGS
    )
    y_true, y_pred = preds["y_true"], preds["y_pred"]
    assert y_true.shape == y_pred.shape
    assert y_true.ndim == 1 and y_true.numel() > 0

    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    recomputed = float(1.0 - ss_res / ss_tot)
    assert abs(recomputed - preds["val_r2"]) < 1e-4


def test_collect_forecasts_shapes_and_horizon_alignment(cfg):
    """Forecast collection aligns forecast/target to a common horizon length.

    Stub forecast_fn returns MORE steps than the target horizon; the collector
    must slice both to H = cfg.data.horizon for the overlay.
    """
    H = cfg.data.horizon
    L = cfg.data.context_length

    def stub_forecast_fn(ctx_norm):  # [B, L, F] -> [B, H+2] (over-long on purpose)
        close = ctx_norm[..., CLOSE_IDX]            # [B, L]
        return close[:, : H + 2]                    # longer than the target window

    n_samples = 5
    out = collect_timesfm_forecasts(
        cfg, stub_forecast_fn, n_samples=n_samples, split="val", device="cpu"
    )
    assert out["context"].shape == (n_samples, L)
    assert out["target_true"].shape == (n_samples, H)   # sliced down from H+2
    assert out["forecast"].shape == (n_samples, H)
    assert out["channel"] == CLOSE_IDX
    assert torch.isfinite(out["forecast"]).all()


def test_normalize_context_flag_changes_features(cfg):
    """normalize_context=False must actually feed raw context (different fit)."""
    enc = raw_feature_encode_fn()
    r_norm = probe_regression_features(
        enc, cfg, target_kind="future_volatility",
        source_label="norm", normalize_context=True, **_PROBE_ARGS,
    )
    r_raw = probe_regression_features(
        enc, cfg, target_kind="future_volatility",
        source_label="raw", normalize_context=False, **_PROBE_ARGS,
    )
    # Normalized context (std≈1 per channel) vs raw context (true std) produce
    # different feature matrices → the fit must differ. Proves the flag is plumbed.
    assert r_norm["val_r2"] != r_raw["val_r2"]


def test_raw_sigma_encode_fn_shape_and_probe(cfg):
    """raw_sigma_encode_fn yields a single feature and probes end-to-end."""
    import torch as _t

    enc = raw_sigma_encode_fn()
    feat = enc(_t.randn(5, cfg.data.context_length, 6))
    assert feat.shape == (5, 1)

    res = probe_regression_predictions(
        enc, cfg, target_kind="future_volatility",
        source_label="raw_sigma", normalize_context=False, **_PROBE_ARGS,
    )
    assert res["y_true"].shape == res["y_pred"].shape
    assert _t.isfinite(_t.tensor(res["val_r2"]))


def test_plot_functions_smoke(cfg):
    """Plotters run end-to-end if matplotlib is available (skipped otherwise)."""
    import pytest

    pytest.importorskip("matplotlib")
    from jepa_quant.eval.forecast_plots import (
        plot_forecast_overlays,
        plot_pred_vs_actual_grid,
    )

    encoder = _eval_encoder(cfg)
    preds = probe_regression_predictions(
        encoder, cfg, target_kind="future_volatility", source_label="z", **_PROBE_ARGS
    )
    fig1 = plot_pred_vs_actual_grid([preds], ncols=1)
    assert fig1 is not None

    def stub_forecast_fn(ctx_norm):
        return ctx_norm[..., CLOSE_IDX][:, : cfg.data.horizon]

    forecasts = collect_timesfm_forecasts(
        cfg, stub_forecast_fn, n_samples=4, split="val", device="cpu"
    )
    fig2 = plot_forecast_overlays(forecasts, n=4, ncols=2)
    assert fig2 is not None

    import matplotlib.pyplot as plt

    plt.close("all")
