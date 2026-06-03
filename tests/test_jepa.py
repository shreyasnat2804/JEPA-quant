"""Unit + smoke tests for the download-free JEPA-quant path.

Organised by module. The foundation-model backends (Moirai / Qwen+LoRA /
FinBERT) are exercised only at their factory + import-guard branches, gated so
they never trigger a network download when the heavy dep happens to be present.
"""

from __future__ import annotations

import dataclasses as dc
import importlib.util

import numpy as np
import pytest
import torch
import torch.nn as nn

from jepa_quant import build_components, build_dataloaders
from jepa_quant.data.price_dataset import PriceWindowDataset, _timestep_features
from jepa_quant.encoders.price_encoder import (
    TransformerPriceEncoder,
    _SinusoidalPositions,
    build_price_encoder,
)
from jepa_quant.encoders.projection import ProjectionHead
from jepa_quant.encoders.target_encoder import TargetEncoder
from jepa_quant.encoders.text_encoder import build_text_encoder
from jepa_quant.predictor import build_predictor
from jepa_quant.predictor.base import Predictor
from jepa_quant.predictor.transformer_predictor import TransformerPredictor
from jepa_quant.regularization import (
    build_regularizer,
    register_regularizer,
    registered_regularizers,
)
from jepa_quant.regularization.base import Regularizer
from jepa_quant.regularization.codebook import SoftCodebook
from jepa_quant.regularization.vicreg import VICReg, covariance_loss, variance_loss
from jepa_quant.training import JEPATrainer, set_seed
from jepa_quant.training.ema import ema_update
from jepa_quant.training.jepa_loss import jepa_loss

# Mirror of the fixture geometry in conftest.py (kept in sync there).
LATENT_DIM = 16
N_FEATURES = 6


def _dep_missing(name: str) -> bool:
    return importlib.util.find_spec(name) is None


# ---------------------------------------------------------------------------
# ema.py
# ---------------------------------------------------------------------------


def test_ema_update_blends_params_toward_source():
    target = nn.Linear(4, 4)
    source = nn.Linear(4, 4)
    with torch.no_grad():
        for p in target.parameters():
            p.fill_(0.0)
        for p in source.parameters():
            p.fill_(1.0)
    ema_update(target, source, decay=0.9)
    for p in target.parameters():
        # 0.9*0 + 0.1*1
        assert torch.allclose(p, torch.full_like(p, 0.1))


def test_ema_update_copies_buffers():
    target = nn.BatchNorm1d(3)
    source = nn.BatchNorm1d(3)
    with torch.no_grad():
        source.running_mean.fill_(5.0)
    ema_update(target, source, decay=0.5)
    assert torch.allclose(target.running_mean, source.running_mean)


# ---------------------------------------------------------------------------
# jepa_loss.py
# ---------------------------------------------------------------------------


def test_jepa_cosine_is_zero_for_identical_vectors():
    z = torch.randn(8, LATENT_DIM)
    assert jepa_loss(z, z.clone(), "cosine").item() == pytest.approx(0.0, abs=1e-6)


def test_jepa_cosine_is_one_for_orthogonal_vectors():
    a = torch.zeros(1, 2)
    a[0, 0] = 1.0
    b = torch.zeros(1, 2)
    b[0, 1] = 1.0
    assert jepa_loss(a, b, "cosine").item() == pytest.approx(1.0, abs=1e-6)


def test_jepa_l2_is_zero_for_identical_vectors():
    z = torch.randn(4, LATENT_DIM)
    assert jepa_loss(z, z.clone(), "l2").item() == pytest.approx(0.0, abs=1e-7)


def test_jepa_unknown_kind_raises():
    with pytest.raises(ValueError):
        jepa_loss(torch.randn(2, 3), torch.randn(2, 3), "bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# projection.py
# ---------------------------------------------------------------------------


def test_projection_always_ends_in_layernorm():
    for hidden in (None, 32):
        head = ProjectionHead(8, LATENT_DIM, hidden_dim=hidden)
        assert isinstance(head.net[-1], nn.LayerNorm)


def test_projection_output_shape_and_normalization():
    head = ProjectionHead(8, LATENT_DIM, hidden_dim=16)
    out = head(torch.randn(5, 8))
    assert out.shape == (5, LATENT_DIM)
    # LayerNorm output is ~zero-mean across the feature dim.
    assert out.mean(dim=-1).abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
# vicreg.py
# ---------------------------------------------------------------------------


def test_variance_loss_zero_when_variance_above_gamma():
    z = torch.randn(2000, 4) * 5.0
    assert variance_loss(z, gamma=1.0, eps=1e-4).item() == pytest.approx(0.0, abs=1e-3)


def test_variance_loss_positive_when_collapsed():
    z = torch.ones(100, 4) * 3.0  # zero variance across the batch
    assert variance_loss(z, gamma=1.0, eps=1e-4).item() == pytest.approx(1.0, abs=1e-2)


def test_covariance_loss_small_for_decorrelated_features():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(4000, 4, generator=g)
    assert covariance_loss(z).item() < 0.05


def test_covariance_loss_large_for_correlated_features():
    col = torch.randn(500, 1)
    z = col.repeat(1, 4)  # perfectly correlated columns
    assert covariance_loss(z).item() > 1.0


def test_vicreg_transform_is_identity_and_loss_metrics():
    from jepa_quant import VICRegConfig

    reg = VICReg(VICRegConfig())
    z = torch.randn(256, LATENT_DIM)
    assert torch.equal(reg.transform(z), z)
    loss, metrics = reg.loss(z, z)
    assert loss.item() >= 0.0
    assert {"reg/v", "reg/c", "reg/z_std_mean"} <= set(metrics)


# ---------------------------------------------------------------------------
# regularization/base.py + registry
# ---------------------------------------------------------------------------


def test_registry_lists_builtins_sorted():
    names = registered_regularizers()
    assert names == sorted(names)
    assert {"vicreg", "codebook"} <= set(names)


def test_build_unknown_regularizer_raises(cfg):
    bad = dc.replace(cfg, regularizer="nope")
    with pytest.raises(ValueError):
        build_regularizer(bad)


def test_register_duplicate_name_raises():
    with pytest.raises(ValueError):

        @register_regularizer("vicreg")  # already registered
        def _dup(_cfg):  # pragma: no cover - body never runs
            ...


def test_base_regularizer_defaults():
    reg = Regularizer()
    z = torch.randn(3, 4)
    assert torch.equal(reg.transform(z), z)  # identity default
    assert reg.maybe_kmeans_init(z) is None
    with pytest.raises(NotImplementedError):
        reg.loss(z, z)


# ---------------------------------------------------------------------------
# codebook.py
# ---------------------------------------------------------------------------


def _codebook(n_codes=4, kmeans_init=False):
    from jepa_quant import CodebookConfig

    return SoftCodebook(
        dc.replace(CodebookConfig(), n_codes=n_codes, kmeans_init=kmeans_init), LATENT_DIM
    )


def test_codebook_transform_shape_and_commit_loss():
    cb = _codebook()
    z = torch.randn(8, LATENT_DIM)
    z_used = cb.transform(z)
    assert z_used.shape == z.shape
    loss, metrics = cb.loss(z, z_used)
    assert loss.item() >= 0.0
    assert 0.0 <= metrics["reg/codebook_util"] <= 1.0


def test_codebook_kmeans_init_sets_codes_and_is_idempotent():
    if _dep_missing("sklearn"):
        pytest.skip("sklearn not installed")
    cb = _codebook(n_codes=4, kmeans_init=True)
    assert cb._initialised is False
    embeddings = torch.randn(64, LATENT_DIM)
    before = cb.codes.clone()
    cb.maybe_kmeans_init(embeddings)
    assert cb._initialised is True
    assert not torch.equal(before, cb.codes)
    # second call is a no-op
    snapshot = cb.codes.clone()
    cb.maybe_kmeans_init(torch.randn(64, LATENT_DIM))
    assert torch.equal(snapshot, cb.codes)


# ---------------------------------------------------------------------------
# encoders/price_encoder.py
# ---------------------------------------------------------------------------


def test_sinusoidal_positions_adds_encoding_without_changing_shape():
    pos = _SinusoidalPositions(8, max_len=16)
    x = torch.zeros(2, 5, 8)
    out = pos(x)
    assert out.shape == x.shape
    assert not torch.equal(out, x)  # something was added


def test_transformer_price_encoder_forward_shape(cfg):
    enc = build_price_encoder(cfg.price_encoder)
    assert isinstance(enc, TransformerPriceEncoder)
    z = enc(torch.randn(3, cfg.price_encoder.context_length, N_FEATURES))
    assert z.shape == (3, LATENT_DIM)


def test_price_encoder_param_groups_respect_freeze():
    from jepa_quant import PriceEncoderConfig

    frozen = TransformerPriceEncoder(
        dc.replace(PriceEncoderConfig(), backend="transformer", freeze_backbone=True)
    )
    assert list(frozen.tunable_backbone_parameters()) == []
    assert len(list(frozen.projection_parameters())) > 0

    live = TransformerPriceEncoder(
        dc.replace(PriceEncoderConfig(), backend="transformer", freeze_backbone=False)
    )
    assert len(list(live.tunable_backbone_parameters())) > 0


def test_build_price_encoder_unknown_backend_raises():
    from jepa_quant import PriceEncoderConfig

    with pytest.raises(ValueError):
        build_price_encoder(dc.replace(PriceEncoderConfig(), backend="bogus"))  # type: ignore[arg-type]


def test_build_moirai_without_uni2ts_raises_importerror():
    if not _dep_missing("uni2ts"):
        pytest.skip("uni2ts installed — would attempt a real download")
    from jepa_quant import PriceEncoderConfig

    with pytest.raises(ImportError):
        build_price_encoder(dc.replace(PriceEncoderConfig(), backend="moirai"))


# ---------------------------------------------------------------------------
# encoders/target_encoder.py
# ---------------------------------------------------------------------------


def test_target_encoder_is_frozen_copy(cfg):
    src = build_price_encoder(cfg.price_encoder)
    tgt = TargetEncoder(src, decay=0.99)
    assert all(not p.requires_grad for p in tgt.parameters())
    out = tgt(torch.randn(2, cfg.price_encoder.context_length, N_FEATURES))
    assert out.shape == (2, LATENT_DIM)


def test_target_encoder_rejects_bad_decay(cfg):
    src = build_price_encoder(cfg.price_encoder)
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            TargetEncoder(src, decay=bad)


def test_target_encoder_update_moves_toward_source(cfg):
    src = build_price_encoder(cfg.price_encoder)
    tgt = TargetEncoder(src, decay=0.5)
    # mutate the source, then EMA-update.
    with torch.no_grad():
        for p in src.parameters():
            p.add_(1.0)
    before = [p.clone() for p in tgt.encoder.parameters()]
    tgt.update(src)
    moved = any(not torch.equal(b, a) for b, a in zip(before, tgt.encoder.parameters()))
    assert moved
    assert all(not p.requires_grad for p in tgt.parameters())


# ---------------------------------------------------------------------------
# encoders/text_encoder.py
# ---------------------------------------------------------------------------


def test_text_encoder_disabled_returns_none():
    from jepa_quant import TextEncoderConfig

    assert build_text_encoder(TextEncoderConfig()) is None


def test_text_encoder_unknown_backend_raises():
    from jepa_quant import TextEncoderConfig

    bad = dc.replace(TextEncoderConfig(), enabled=True, backend="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_text_encoder(bad)


def test_text_encoder_without_transformers_raises_importerror():
    if not _dep_missing("transformers"):
        pytest.skip("transformers installed — would attempt a real download")
    from jepa_quant import TextEncoderConfig

    enabled = dc.replace(TextEncoderConfig(), enabled=True, backend="finbert")
    with pytest.raises(ImportError):
        build_text_encoder(enabled)


# ---------------------------------------------------------------------------
# predictor/*
# ---------------------------------------------------------------------------


def test_transformer_predictor_forward_price_only(cfg):
    pred = build_predictor(cfg.predictor, in_dim=LATENT_DIM, use_text=False)
    assert isinstance(pred, TransformerPredictor)
    out = pred(torch.randn(4, LATENT_DIM), None)
    assert out.shape == (4, LATENT_DIM)
    assert len(list(pred.adapter_parameters())) > 0
    assert len(list(pred.projection_parameters())) > 0


def test_transformer_predictor_uses_text_token(cfg):
    pred = TransformerPredictor(cfg.predictor, in_dim=LATENT_DIM, use_text=True)
    out = pred(torch.randn(4, LATENT_DIM), torch.randn(4, LATENT_DIM))
    assert out.shape == (4, LATENT_DIM)


def test_predictor_base_is_abstract():
    p = Predictor()
    with pytest.raises(NotImplementedError):
        p.forward(torch.randn(1, 2))
    with pytest.raises(NotImplementedError):
        list(p.projection_parameters())
    with pytest.raises(NotImplementedError):
        list(p.adapter_parameters())


def test_build_predictor_unknown_backend_raises(cfg):
    with pytest.raises(ValueError):
        build_predictor(
            dc.replace(cfg.predictor, backend="bogus"), in_dim=LATENT_DIM, use_text=False  # type: ignore[arg-type]
        )


def test_build_lm_predictor_without_peft_raises_importerror(cfg):
    if not (_dep_missing("peft") or _dep_missing("transformers")):
        pytest.skip("peft + transformers installed — would attempt a real download")
    with pytest.raises(ImportError):
        build_predictor(dc.replace(cfg.predictor, backend="lm"), in_dim=LATENT_DIM, use_text=False)


# ---------------------------------------------------------------------------
# data/price_dataset.py
# ---------------------------------------------------------------------------


def test_timestep_features_drops_first_row_and_log_transforms():
    import pandas as pd

    df = pd.DataFrame({"close": [10.0, 11.0, 12.0], "volume": [100.0, 0.0, 50.0]})
    feats = _timestep_features(df, ("close", "volume"))
    assert feats.shape == (2, 2)  # T-1 rows, 2 cols
    assert feats.dtype == np.float32


def test_dataset_window_shapes_and_causal_normalization(cfg):
    ds = PriceWindowDataset(cfg.data, "train")
    assert len(ds) > 0
    sample = ds[0]
    assert sample["context"].shape == (cfg.data.context_length, N_FEATURES)
    assert sample["target"].shape == (cfg.data.horizon, N_FEATURES)
    # causal normalization: context is standardized to ~zero mean / unit std.
    ctx = sample["context"]
    assert ctx.mean(dim=0).abs().max().item() < 1e-4
    assert (ctx.std(dim=0) - 1.0).abs().max().item() < 0.2


def test_dataset_split_is_chronological_and_disjoint(cfg):
    train = PriceWindowDataset(cfg.data, "train")
    val = PriceWindowDataset(cfg.data, "val")
    train_starts: dict[int, list[int]] = {}
    val_starts: dict[int, list[int]] = {}
    for ti, s in train._index:
        train_starts.setdefault(ti, []).append(s)
    for ti, s in val._index:
        val_starts.setdefault(ti, []).append(s)
    for ti in val_starts:
        assert max(train_starts[ti]) < min(val_starts[ti])


def test_dataset_raises_on_empty_dir(tmp_path):
    from jepa_quant import DataConfig

    empty = tmp_path / "empty"
    empty.mkdir()
    cfg_data = dc.replace(DataConfig(), data_dir=str(empty))
    with pytest.raises(FileNotFoundError):
        PriceWindowDataset(cfg_data, "train")


def test_dataset_raises_when_series_too_short(tmp_path):
    import pandas as pd

    from jepa_quant import DataConfig

    d = tmp_path / "short"
    d.mkdir()
    short = pd.DataFrame(
        {c: np.linspace(1, 5, 5) for c in ("open", "high", "low", "close", "vwap", "volume")}
    )
    short.to_parquet(d / "S.parquet")
    cfg_data = dc.replace(DataConfig(), data_dir=str(d), context_length=8, horizon=4)
    with pytest.raises(ValueError):
        PriceWindowDataset(cfg_data, "train")


def test_build_dataloaders_yields_batches(cfg):
    train_loader, val_loader = build_dataloaders(cfg)
    batch = next(iter(train_loader))
    assert batch["context"].shape == (cfg.train.batch_size, cfg.data.context_length, N_FEATURES)
    assert batch["target"].shape == (cfg.train.batch_size, cfg.data.horizon, N_FEATURES)
    assert len(val_loader) > 0


# ---------------------------------------------------------------------------
# training/trainer.py
# ---------------------------------------------------------------------------


def test_set_seed_is_deterministic():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_build_components_wires_target_as_frozen_copy(cfg):
    comp = build_components(cfg)
    assert comp.text_encoder is None  # disabled by default
    assert all(not p.requires_grad for p in comp.target_encoder.parameters())


def test_trainer_smoke_runs_and_advances_phase(cfg):
    comp = build_components(cfg)
    train_loader, val_loader = build_dataloaders(cfg)
    trainer = JEPATrainer(cfg, comp, train_loader, val_loader)
    assert trainer.phase == 1
    history = trainer.train()
    assert len(history) > 0
    assert trainer.phase == 2  # crossed warmup_proj_steps
    # target encoder never accrued grad.
    assert all(not p.requires_grad for p in comp.target_encoder.parameters())
    assert np.isfinite(history[-1]["loss"])


def test_trainer_validate_returns_finite_metrics(cfg):
    comp = build_components(cfg)
    train_loader, val_loader = build_dataloaders(cfg)
    trainer = JEPATrainer(cfg, comp, train_loader, val_loader)
    out = trainer.validate(max_batches=2)
    assert set(out) == {"val_jepa", "val_z_std"}
    assert np.isfinite(out["val_jepa"])


def test_trainer_raises_on_empty_loader(cfg):
    huge_batch = dc.replace(cfg, train=dc.replace(cfg.train, batch_size=100_000))
    comp = build_components(huge_batch)
    train_loader, val_loader = build_dataloaders(huge_batch)
    with pytest.raises(ValueError, match="empty"):
        JEPATrainer(huge_batch, comp, train_loader, val_loader)


def test_trainer_codebook_kmeans_init_path(cfg):
    if _dep_missing("sklearn"):
        pytest.skip("sklearn not installed")
    cb_cfg = dc.replace(
        cfg,
        regularizer="codebook",
        codebook=dc.replace(cfg.codebook, n_codes=4, kmeans_init=True),
    )
    comp = build_components(cb_cfg)
    train_loader, val_loader = build_dataloaders(cb_cfg)
    trainer = JEPATrainer(cb_cfg, comp, train_loader, val_loader)
    assert comp.regularizer._initialised is False
    trainer.maybe_kmeans_init(n_samples=16)
    assert comp.regularizer._initialised is True
