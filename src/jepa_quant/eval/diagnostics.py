"""Stage 0 diagnostics for JEPA-quant checkpoint quality.

Five tests in priority order (see CLAUDE.md Research Workflow section):

1. shuffled_target_control  - decisive: is the temporal link real or degenerate?
2. compute_baselines        - untrained projection + adjacent-timestep persistence
3. collapse_audit           - per-dim std, covariance eigenspectrum, effective rank
4. level_dependence_test    - R^2 of z_price vs volatility proxy (per-window log-return scale)
5. regime_clustering        - PCA of z_price + regime labels (confirmatory only)

All functions return plain dicts of numbers. Plotting belongs in nb00_diagnostics.ipynb.
Use n_batches >= 50 - the trainer's default of 20 is too noisy for signal.

Inputs to every encoder call are normalized log-returns [B, L, F] as produced by
PriceWindowDataset. The per-sample normalization (context mean/std applied to both
windows) is part of the dataset; the encoder never sees raw prices.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..config import JEPAConfig
from ..data.price_dataset import PriceWindowDataset
from ..training.jepa_loss import jepa_loss
from ..training.trainer import JEPAComponents, build_components, resolve_device


# ==========================================================================
# Checkpoint loading
# ==========================================================================


def load_checkpoint(
    path: str,
    cfg: JEPAConfig,
    device: Optional[str] = None,
) -> JEPAComponents:
    """Restore price_encoder and predictor from a JEPATrainer checkpoint.

    Checkpoint format (written by JEPATrainer.on_new_best):
        {step, val_jepa, price_encoder: state_dict, predictor: state_dict, opt: state_dict}

    The target_encoder is hard-copied from the loaded price_encoder. This approximates
    the saved-step state (high EMA decay means target tracks source closely). The
    checkpoint does not store the target_encoder separately.
    """
    dev = resolve_device(device or cfg.train.device)
    components = build_components(cfg)
    # weights_only=False: the checkpoint opt state dict contains non-tensor Python objects.
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    components.price_encoder.load_state_dict(ckpt["price_encoder"])
    components.predictor.load_state_dict(ckpt["predictor"])
    # Hard-copy loaded price_encoder into target_encoder.
    with torch.no_grad():
        for tp, sp in zip(
            components.target_encoder.encoder.parameters(),
            components.price_encoder.parameters(),
        ):
            tp.copy_(sp.detach())
        for tb, sb in zip(
            components.target_encoder.encoder.buffers(),
            components.price_encoder.buffers(),
        ):
            tb.copy_(sb)
    for mod in (
        components.price_encoder,
        components.target_encoder,
        components.predictor,
        components.regularizer,
    ):
        mod.to(dev).eval()
    return components


# ==========================================================================
# Internal helpers
# ==========================================================================


def _collect_latents(
    components: JEPAComponents,
    loader: DataLoader,
    n_batches: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Collect (z_price, z_pred, z_target) from up to n_batches.

    All returned tensors are float32 on CPU, shape [N, latent_dim].
    """
    zp, zpred, zt = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            ctx = batch["context"].to(device)
            tgt = batch["target"].to(device)
            z_price = components.price_encoder(ctx)
            z_used = components.regularizer.transform(z_price)
            z_pred_b = components.predictor(z_used, None)
            z_target_b = components.target_encoder(tgt)
            zp.append(z_price.cpu())
            zpred.append(z_pred_b.cpu())
            zt.append(z_target_b.cpu())
    return torch.cat(zp), torch.cat(zpred), torch.cat(zt)


def _build_val_loader(cfg: JEPAConfig) -> DataLoader:
    """Build a val DataLoader with the given cfg (respects cfg.data.normalize)."""
    val_ds = PriceWindowDataset(cfg.data, "val")
    return DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.train.num_workers,
    )


# ==========================================================================
# 1. Shuffled-target control
# ==========================================================================


def shuffled_target_control(
    components: JEPAComponents,
    val_loader: DataLoader,
    n_batches: int = 50,
    seed: int = 42,
    device: Optional[str] = None,
) -> dict:
    """Decisive degeneracy test.

    Compute val cosine similarity normally, then recompute with z_target randomly
    permuted across the batch so each z_pred is paired with an unrelated target.
    Uses a fixed seed for reproducibility.

    Decision rule:
      ratio = shuffled_cosine_mean / true_cosine_mean
      ratio >= 0.8  -> degenerate: temporal structure is not being captured
      ratio <  0.8  -> temporal:   the predictor encodes genuine future structure

    Rationale: if the representations have collapsed (all z_targets look similar),
    shuffling does not change the cosine similarity because every pair is equally
    similar. A real temporal link requires that the correct pairing scores higher
    than a random pairing.
    """
    dev = resolve_device(device or "cpu")
    for mod in (
        components.price_encoder,
        components.target_encoder,
        components.predictor,
        components.regularizer,
    ):
        mod.to(dev).eval()

    _, z_pred, z_target = _collect_latents(components, val_loader, n_batches, dev)

    true_cos = F.cosine_similarity(z_pred, z_target, dim=-1)  # [N]

    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(z_target.size(0), generator=rng)
    shuffled_cos = F.cosine_similarity(z_pred, z_target[perm], dim=-1)  # [N]

    true_mean = float(true_cos.mean())
    true_std = float(true_cos.std())
    shuf_mean = float(shuffled_cos.mean())
    shuf_std = float(shuffled_cos.std())
    ratio = shuf_mean / true_mean if abs(true_mean) > 1e-8 else float("nan")

    return {
        "true_cosine_mean": true_mean,
        "true_cosine_std": true_std,
        "true_jepa": 1.0 - true_mean,
        "shuffled_cosine_mean": shuf_mean,
        "shuffled_cosine_std": shuf_std,
        "shuffled_jepa": 1.0 - shuf_mean,
        "ratio_shuffled_over_true": ratio,
        "n_samples": int(z_pred.size(0)),
        "verdict": "degenerate" if ratio >= 0.8 else "temporal",
    }


# ==========================================================================
# 2. Baselines
# ==========================================================================


def compute_baselines(
    cfg: JEPAConfig,
    val_loader: DataLoader,
    n_batches: int = 50,
    seed: int = 42,
    device: Optional[str] = None,
) -> dict:
    """Two baselines the trained model must exceed.

    (a) Untrained: fresh random init with the same cfg. No training, no VICReg push.
        The projection head and predictor are completely random, backbone is frozen
        random. Gives the floor for any randomly-initialized system before learning.

    (b) Adjacent-timestep persistence: cosine similarity of the last context timestep
        vs the first target timestep in the normalized log-return feature space (F=6).
        This is the 6-d raw-feature baseline - not dimensionally comparable to the
        256-d JEPA loss, but gives context for how correlated adjacent log-returns are.

    Decision rule for (a): if trained_val_jepa >= untrained_jepa, training has not
    improved over random - investigate collapse or optimizer failure.
    """
    dev = resolve_device(device or "cpu")
    torch.manual_seed(seed)

    untrained = build_components(cfg)
    for mod in (
        untrained.price_encoder,
        untrained.target_encoder,
        untrained.predictor,
        untrained.regularizer,
    ):
        mod.to(dev).eval()

    ut_jepa, adj_cos = [], []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            ctx = batch["context"].to(dev)  # [B, L, F]
            tgt = batch["target"].to(dev)   # [B, H, F]

            z_price = untrained.price_encoder(ctx)
            z_used = untrained.regularizer.transform(z_price)
            z_pred = untrained.predictor(z_used, None)
            z_tgt = untrained.target_encoder(tgt)
            ut_jepa.append(float(jepa_loss(z_pred, z_tgt, "cosine")))

            # Adjacent-timestep persistence in 6-d normalized feature space
            ctx_last = ctx[:, -1, :]   # [B, F]
            tgt_first = tgt[:, 0, :]   # [B, F]
            cos = float(F.cosine_similarity(ctx_last, tgt_first, dim=-1).mean())
            adj_cos.append(cos)

    ut_arr = np.array(ut_jepa)
    ps_arr = np.array(adj_cos)
    return {
        "untrained_jepa_mean": float(ut_arr.mean()),
        "untrained_jepa_std": float(ut_arr.std()),
        "adjacent_cosine_mean": float(ps_arr.mean()),
        "adjacent_cosine_std": float(ps_arr.std()),
        "adjacent_jepa_proxy": 1.0 - float(ps_arr.mean()),
        "n_batches": len(ut_jepa),
    }


# ==========================================================================
# 3. Collapse audit
# ==========================================================================


def collapse_audit(
    components: JEPAComponents,
    val_loader: DataLoader,
    n_batches: int = 50,
    seed: int = 42,
    device: Optional[str] = None,
) -> dict:
    """Representation collapse diagnostic.

    Collects z_price over n_batches and computes:
    - Per-dimension std: should be ~1 if variance hinge is working on val
    - Covariance eigenvalue spectrum via SVD of centered Z
    - Effective rank (participation ratio): PR = (sum lambda_i)^2 / sum(lambda_i^2)
      Well-utilized 256-dim space -> PR near 256. Collapsed to k dims -> PR near k.
    - Top-k eigenvalue share: fraction of total variance explained by top 10 dims

    A train z_std ~1.07 but val z_std ~0.63-0.74 gap (observed in training runs)
    means VICReg variance hinge only acts on train batches; the val representations
    are less spread, suggesting the encoder has not generalized fully.
    """
    dev = resolve_device(device or "cpu")
    components.price_encoder.to(dev).eval()

    zp_list = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            z = components.price_encoder(batch["context"].to(dev))
            zp_list.append(z.cpu())

    Z = torch.cat(zp_list).float()  # [N, D]
    Zc = Z - Z.mean(dim=0, keepdim=True)

    # Eigenvalues of covariance via SVD: lambda_i = s_i^2 / (N-1)
    _, S, _ = torch.linalg.svd(Zc, full_matrices=False)
    eigs = (S ** 2) / max(Z.size(0) - 1, 1)

    sum_lam = float(eigs.sum())
    sum_lam2 = float((eigs ** 2).sum())
    eff_rank = (sum_lam ** 2) / sum_lam2 if sum_lam2 > 1e-12 else float("nan")

    per_dim_std = Zc.std(dim=0)  # [D]

    top10_share = float(eigs[:10].sum() / sum_lam) if sum_lam > 0 else float("nan")
    top1_share = float(eigs[0] / sum_lam) if sum_lam > 0 else float("nan")

    return {
        "z_std_mean": float(per_dim_std.mean()),
        "z_std_min": float(per_dim_std.min()),
        "z_std_max": float(per_dim_std.max()),
        "effective_rank": eff_rank,
        "top1_eigenvalue_share": top1_share,
        "top10_eigenvalue_share": top10_share,
        "per_dim_std": per_dim_std.tolist(),
        "eigenvalues": eigs.tolist(),
        "n_dims": int(Z.size(1)),
        "n_samples": int(Z.size(0)),
    }


# ==========================================================================
# 4. Level-dependence test
# ==========================================================================


def level_dependence_test(
    components: JEPAComponents,
    cfg: JEPAConfig,
    n_batches: int = 50,
    seed: int = 42,
    device: Optional[str] = None,
) -> dict:
    """How much of z_price variance is explained by per-window volatility regime?

    Builds an unnormalized val loader internally (DataConfig.normalize=False) to
    access the raw log-return scale. The volatility proxy is the mean per-feature
    temporal std of the context window before normalization - this is exactly the
    sigma that the dataset divides out. High R^2 means the representation encodes
    the scale (volatility regime) that normalization was supposed to remove.

    Note: some volatility sensitivity is expected and fine (the encoder SHOULD
    distinguish high-vol from low-vol regimes). It becomes a degeneracy flag only
    when R^2 is near 1.0, meaning z_price encodes almost nothing else.

    The context is normalized manually before encoding so the encoder sees the same
    distribution as during training.
    """
    dev = resolve_device(device or "cpu")
    components.price_encoder.to(dev).eval()

    # Unnormalized val loader: same windows, no mean/std division
    data_unnorm = dataclasses.replace(cfg.data, normalize=False)
    cfg_unnorm = dataclasses.replace(cfg, data=data_unnorm)
    val_unnorm = _build_val_loader(cfg_unnorm)

    zp_list, vol_list = [], []
    with torch.no_grad():
        for i, batch in enumerate(val_unnorm):
            if i >= n_batches:
                break
            ctx_raw = batch["context"]  # [B, L, F] unnormalized log-returns

            # Volatility proxy: mean temporal std across features before normalization
            vol = ctx_raw.std(dim=1).mean(dim=1)  # [B]
            vol_list.append(vol.numpy())

            # Normalize exactly as the training pipeline does before encoding
            mu = ctx_raw.mean(dim=1, keepdim=True)      # [B, 1, F]
            sigma = ctx_raw.std(dim=1, keepdim=True) + 1e-6
            ctx_norm = (ctx_raw - mu) / sigma

            z = components.price_encoder(ctx_norm.to(dev))
            zp_list.append(z.cpu().numpy())

    Z = np.concatenate(zp_list, axis=0)  # [N, D]
    vol = np.concatenate(vol_list)        # [N]

    # Standardize vol for numerically stable regression
    vol_z = (vol - vol.mean()) / (vol.std() + 1e-8)  # unit-variance predictor

    # Per-dimension R^2: OLS of z_d on vol_z
    # beta_d = dot(vol_z, z_d) / dot(vol_z, vol_z)
    N = len(vol_z)
    beta = (vol_z @ Z) / (np.dot(vol_z, vol_z) + 1e-8)  # [D]
    residuals = Z - np.outer(vol_z, beta)                 # [N, D]
    ss_res = (residuals ** 2).sum(axis=0)                 # [D]
    Z_c = Z - Z.mean(axis=0)
    ss_tot = (Z_c ** 2).sum(axis=0) + 1e-8               # [D]
    r2 = 1.0 - ss_res / ss_tot                            # [D]

    return {
        "mean_r2": float(r2.mean()),
        "median_r2": float(np.median(r2)),
        "max_r2": float(r2.max()),
        "n_dims_r2_gt_0p1": int((r2 > 0.1).sum()),
        "n_dims_r2_gt_0p3": int((r2 > 0.3).sum()),
        "top10_r2": sorted(r2.tolist(), reverse=True)[:10],
        "r2_per_dim": r2.tolist(),
        "n_samples": N,
    }


# ==========================================================================
# 5. Regime clustering (confirmatory)
# ==========================================================================


def regime_clustering(
    components: JEPAComponents,
    val_loader: DataLoader,
    n_batches: int = 50,
    seed: int = 42,
    device: Optional[str] = None,
) -> dict:
    """PCA of val z_price with regime labels for scatter-plot visualization.

    Confirmatory only - use tests 1-3 for decisions.

    Returns the top-20 PCA variance ratios, 2-d PCA projections, and two sets of
    regime labels derived from the normalized context windows (the same windows
    the encoder sees):
    - vol_quintile: quintile of mean absolute log-return (0=low vol, 4=high vol)
    - return_sign: 1 if mean close log-return > 0, else 0

    Since context windows are per-sample normalized, the absolute-value mean is
    a proxy for how far from zero the returns wandered within the window (momentum
    / trending vs mean-reverting), not raw volatility magnitude.
    """
    dev = resolve_device(device or "cpu")
    components.price_encoder.to(dev).eval()

    zp_list, ctx_list = [], []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            ctx = batch["context"]  # [B, L, F] normalized
            z = components.price_encoder(ctx.to(dev))
            zp_list.append(z.cpu())
            ctx_list.append(ctx)

    Z = torch.cat(zp_list).float()    # [N, D]
    C = torch.cat(ctx_list).float()   # [N, L, F]

    Zc = Z - Z.mean(dim=0, keepdim=True)
    U, S, Vt = torch.linalg.svd(Zc, full_matrices=False)
    var_ratios_full = ((S ** 2) / (S ** 2).sum()).numpy()  # [D]

    # 2-d scatter plot projections
    pca2 = (Zc @ Vt[:2].T).numpy()  # [N, 2]

    # Effective rank on all dims
    eff_rank = float(1.0 / (var_ratios_full ** 2).sum())

    # Regime labels from normalized context
    # Index 3 = close column (0=open,1=high,2=low,3=close,4=vwap,5=volume)
    abs_ret = C.abs().mean(dim=(1, 2)).numpy()   # [N] momentum proxy
    mean_close = C[:, :, 3].mean(dim=1).numpy()  # [N] close log-return mean

    # Quintile labels (0-4) for abs_ret
    ranks = np.argsort(np.argsort(abs_ret))
    vol_quintile = (ranks * 5 // len(ranks)).tolist()
    return_sign = (mean_close > 0).astype(int).tolist()

    return {
        "pca_variance_ratios_top20": var_ratios_full[:20].tolist(),
        "effective_rank": eff_rank,
        "top2_pc_variance": float(var_ratios_full[:2].sum()),
        "top10_pc_variance": float(var_ratios_full[:10].sum()),
        "pca2_projections": pca2.tolist(),
        "vol_quintile_labels": vol_quintile,
        "return_sign_labels": return_sign,
        "n_samples": int(Z.size(0)),
    }
