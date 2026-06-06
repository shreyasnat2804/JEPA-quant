"""Behavior-preserving regression test for the linear_probe encode_fn refactor.

The Stage 1 probe was refactored so the collection/fitting core is shared between
the components path (``linear_probe_regression`` / ``linear_probe_direction``) and
a new arbitrary-feature-extractor path (``probe_regression_features`` /
``probe_direction_features``, used by nb04's TimesFM ceiling probe).

These tests prove the refactor did NOT change Stage 1 numbers: feeding an
encode_fn that wraps the very same ``price_encoder`` produces a result dict
bit-identical (modulo the ``probe_source`` label) to the components path at the
same seed. Uses the synthetic-parquet ``cfg`` fixture (real loader, real
PriceWindowDataset) on CPU — no TimesFM, no downloads.
"""

from __future__ import annotations

import torch

from jepa_quant import build_components
from jepa_quant.eval.linear_probe import (
    linear_probe_direction,
    linear_probe_regression,
    probe_direction_features,
    probe_regression_features,
)

# n_batches kept tiny but >0; the synthetic data dir yields a handful of windows.
_PROBE_ARGS = dict(n_train_batches=4, n_val_batches=2, seed=42, device="cpu")


def _strip_label(d: dict) -> dict:
    """Drop the source label so we compare only the computed numbers."""
    return {k: v for k, v in d.items() if k != "probe_source"}


def _build_eval_encoder(cfg):
    """Same frozen encoder used by both paths; eval mode pins dropout off so the
    two paths are deterministic and directly comparable."""
    torch.manual_seed(0)
    components = build_components(cfg)
    components.price_encoder.to("cpu").eval()
    return components


def test_regression_features_matches_components_path(cfg):
    components = _build_eval_encoder(cfg)

    for target_kind in ("future_return", "future_volatility"):
        comp = linear_probe_regression(
            components, cfg, target_kind=target_kind,
            probe_source="z_price", **_PROBE_ARGS,
        )
        feat = probe_regression_features(
            components.price_encoder, cfg, target_kind=target_kind,
            source_label="z_price_via_fn", **_PROBE_ARGS,
        )
        assert _strip_label(comp) == _strip_label(feat), (
            f"{target_kind}: encode_fn path diverged from components path"
        )
        # Sanity: the label is carried through distinctly on each path.
        assert comp["probe_source"] == "z_price"
        assert feat["probe_source"] == "z_price_via_fn"


def test_direction_features_matches_components_path(cfg):
    components = _build_eval_encoder(cfg)

    comp = linear_probe_direction(
        components, cfg, probe_source="z_price", **_PROBE_ARGS,
    )
    feat = probe_direction_features(
        components.price_encoder, cfg, source_label="z_price_via_fn", **_PROBE_ARGS,
    )
    assert _strip_label(comp) == _strip_label(feat)
    assert comp["probe_source"] == "z_price"
    assert feat["probe_source"] == "z_price_via_fn"
