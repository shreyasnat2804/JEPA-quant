"""Stage 1 linear probe — downstream evaluation of frozen z_price.

This is the PRIMARY progress metric for JEPA-quant (CLAUDE.md). val_jepa is
secondary: a checkpoint that hits low val_jepa but has a flat probe is
degenerate (EMA collapse on the predictor/target heads). A checkpoint that
beats the untrained-encoder probe by more than sampling noise has learned
representations with genuine downstream utility.

Two probe families, both fit a single linear/logistic layer on top of the
frozen z_price representation:

  * ``linear_probe_regression``  — ridge to future cumulative log-return or
                                   future volatility
  * ``linear_probe_direction``   — logistic to sign(future cumulative
                                   log-return)

Probe targets are computed from RAW (unnormalized) target windows so the
target scale is comparable across samples. The encoder always receives
per-sample-normalized context (re-normalized inside the collector to match
PriceWindowDataset's normalize=True path exactly — the encoder must see the
same distribution it was trained on).

Pure torch implementation. No sklearn dependency:
  * ridge: closed-form normal equations
  * logistic: L-BFGS with strong-Wolfe line search

Decision rule (applied in the notebook, not enforced here):
    trained_val_metric > untrained_val_metric + sampling_noise
        -> encoder added decodable signal for this target
    otherwise
        -> training did not produce a useful representation for this target;
           pivot architecture/pretraining rather than tuning hyperparameters.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..config import JEPAConfig
from ..data.price_dataset import PriceWindowDataset
from ..training.trainer import JEPAComponents, resolve_device


# Index of the close column in DataConfig.price_cols =
# ("open","high","low","close","vwap","volume"). Hard-coded: the probe targets
# are defined on close log-returns specifically; if price_cols is reordered,
# update this constant rather than auto-detecting (probe semantics are a
# contract, not a runtime knob).
CLOSE_IDX = 3

RegressionTarget = Literal["future_return", "future_volatility"]

# Which representation the probe is fit on:
#   "z_price"  — the post-projection 256-d latent (the model's actual output)
#   "backbone" — the PRE-projection pooled hidden state (input to encoder.head)
# Under freeze_backbone=True the head is the only trained module in the encoder,
# so comparing the two isolates whether the trained head destroys downstream
# signal that the frozen backbone preserved (see Test 4 in nb03c).
ProbeSource = Literal["z_price", "backbone"]


# ==========================================================================
# Data collection
# ==========================================================================


def _build_split_loader(cfg: JEPAConfig, split: str, normalize: bool) -> DataLoader:
    """Deterministic loader for ``split``.

    ``shuffle=False`` so the encoder pass and the target derivation see the
    same sample order across runs. ``normalize=False`` is required when the
    probe target is in raw log-return units (the dataset's per-sample
    normalization erases the cross-sample scale we need to compare cumulative
    returns).
    """
    data_cfg = dataclasses.replace(cfg.data, normalize=normalize)
    ds = PriceWindowDataset(data_cfg, split)  # type: ignore[arg-type]
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.train.num_workers,
    )


def _collect_probe_pairs(
    components: JEPAComponents,
    cfg: JEPAConfig,
    split: str,
    n_batches: int,
    device: torch.device,
    probe_source: ProbeSource = "z_price",
) -> dict[str, Tensor]:
    """Run encoder over ``split`` and return latents + every probe target.

    Returned tensors (all CPU float32 except ``y_sign`` which is long):
        z            [N, D]  representation (per-sample-normalized context).
                             D = latent_dim for ``probe_source="z_price"``,
                             D = backbone hidden dim for ``probe_source="backbone"``.
        y_return     [N]     sum of close log-returns over horizon
        y_sign       [N]     1 if y_return > 0 else 0
        y_volatility [N]     std of close log-returns over horizon

    ``probe_source`` selects which representation populates ``z``:
        "z_price"  — the encoder's returned post-projection latent.
        "backbone" — the PRE-projection ``pooled`` hidden state, captured via a
                     forward pre-hook on ``price_encoder.head``. We still run the
                     full forward (so the hook fires) and discard the returned z.
                     Backend-agnostic: both TransformerPriceEncoder and
                     MoiraiPriceEncoder compute ``pooled`` then ``self.head(pooled)``,
                     so the hook taps the same tensor in either. We deliberately do
                     NOT call ``price_encoder.backbone(x)`` directly — it expects an
                     already-projected ``[B, L+1, d_model]`` sequence, not raw
                     ``[B, L, F]`` features.

    The encoder is moved to ``device`` and set to eval mode (idempotent —
    callers may pre-load weights).
    """
    if probe_source not in ("z_price", "backbone"):
        raise ValueError(
            f"probe_source must be 'z_price' or 'backbone', got {probe_source!r}"
        )
    components.price_encoder.to(device).eval()
    loader = _build_split_loader(cfg, split, normalize=False)

    # For the backbone probe, tap the input to the projection head. The hook
    # signature is hook(module, args); pooled = args[0]. We stash it per-batch
    # and remove the handle in the finally block so repeated calls don't stack
    # hooks on the shared encoder module.
    captured: dict[str, Tensor] = {}
    hook_handle = None
    if probe_source == "backbone":
        def _capture_pre_head(_module, args):  # noqa: ANN001 — torch hook signature
            captured["pooled"] = args[0]

        hook_handle = components.price_encoder.head.register_forward_pre_hook(
            _capture_pre_head
        )

    zs: list[Tensor] = []
    rets: list[Tensor] = []
    signs: list[Tensor] = []
    vols: list[Tensor] = []

    try:
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= n_batches:
                    break
                ctx_raw = batch["context"]  # [B, L, F] raw log-returns
                tgt_raw = batch["target"]   # [B, H, F] raw log-returns

                tgt_close = tgt_raw[:, :, CLOSE_IDX]            # [B, H]
                cum_ret = tgt_close.sum(dim=1)                  # [B] H-step log-return
                sign = (cum_ret > 0).long()                     # [B] 0/1
                vol = tgt_close.std(dim=1, unbiased=False)      # [B]

                # Re-normalize context exactly as PriceWindowDataset does when
                # normalize=True, so the encoder sees its training distribution.
                mu = ctx_raw.mean(dim=1, keepdim=True)          # [B, 1, F]
                sigma = ctx_raw.std(dim=1, keepdim=True) + 1e-6 # [B, 1, F]
                ctx_norm = (ctx_raw - mu) / sigma

                # Always run the full forward so the pre-head hook fires. For
                # probe_source="backbone" we use the captured pooled and discard
                # the returned z; otherwise we use the returned z directly.
                z_out = components.price_encoder(ctx_norm.to(device))
                rep = captured["pooled"] if probe_source == "backbone" else z_out
                z = rep.cpu()                                   # [B, D]

                zs.append(z)
                rets.append(cum_ret)
                signs.append(sign)
                vols.append(vol)
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    if not zs:
        raise ValueError(f"No batches collected for split={split!r}")

    return {
        "z": torch.cat(zs).float(),
        "y_return": torch.cat(rets).float(),
        "y_sign": torch.cat(signs).long(),
        "y_volatility": torch.cat(vols).float(),
    }


# ==========================================================================
# Linear algebra primitives — pure torch
# ==========================================================================


def _standardize(X_train: Tensor, X_val: Tensor) -> tuple[Tensor, Tensor]:
    """Standardize ``X`` using train statistics only — never let val stats
    leak into the fit. eps prevents division by zero on dead dimensions
    (which exist: collapse_audit reported effective_rank ~13/256).
    """
    mu = X_train.mean(dim=0, keepdim=True)
    sigma = X_train.std(dim=0, keepdim=True) + 1e-6
    return (X_train - mu) / sigma, (X_val - mu) / sigma


def _ridge_closed_form(X: Tensor, y: Tensor, alpha: float) -> tuple[Tensor, float]:
    """Solve min ||X w - (y - y_mean)||^2 + alpha * ||w||^2 in closed form.

    Returns ``(w, intercept)`` such that prediction = X_new @ w + intercept.
    The intercept absorbs ``y_train.mean()``; X is already standardized by
    the caller so no per-feature intercept is needed. Matches
    sklearn.linear_model.Ridge(alpha=alpha) up to the standardization step.

    Closed-form is fine for D=256 / N<=50k: one float64 D-by-D solve.
    """
    _, D = X.shape
    y_mean = float(y.mean())
    y_c = y - y_mean
    # Use float64 for the solve to avoid ill-conditioning blowups when
    # effective_rank << D (lots of near-zero eigenvalues in X.T @ X).
    Xd = X.double()
    A = Xd.T @ Xd + alpha * torch.eye(D, dtype=torch.float64)
    b = Xd.T @ y_c.double()
    w = torch.linalg.solve(A, b)
    return w.float(), y_mean


def _logistic_lbfgs(
    X: Tensor, y: Tensor, alpha: float, max_iter: int = 200
) -> tuple[Tensor, float]:
    """L2-regularized logistic regression via L-BFGS with strong-Wolfe line
    search.

    Loss = sum_i BCE(logits_i, y_i) + 0.5 * alpha * ||w||^2.
    Intercept ``b`` is unregularized. Sign convention matches
    sklearn LogisticRegression(C=1/alpha, fit_intercept=True).

    L-BFGS in float64 to keep the Hessian approximation well-conditioned —
    Adam/SGD-style optimizers need hundreds of epochs for a convex problem
    this size; L-BFGS converges in <100 sub-iterations.
    """
    Xd = X.double()
    yd = y.double()
    D = Xd.size(1)
    w = torch.zeros(D, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS(
        [w, b],
        max_iter=max_iter,
        tolerance_grad=1e-7,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        opt.zero_grad()
        logits = (Xd @ w + b).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, yd, reduction="sum")
        loss = loss + 0.5 * alpha * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().float(), float(b.detach())


def _r2(y_true: Tensor, y_pred: Tensor) -> float:
    """Coefficient of determination. Uses TEST-set y mean as the baseline
    (standard "out-of-sample R^2" definition). Negative R^2 means worse than
    predicting the val mean — a real possibility when the encoder is
    degenerate.
    """
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _auroc(y_true: Tensor, scores: Tensor) -> float:
    """ROC AUC via Mann-Whitney U. Continuous scores from logistic regression
    have effectively no ties, so the rank-statistic form is exact here.
    Returns NaN if one class is empty.
    """
    order = scores.argsort()
    y_sorted = y_true[order].double()
    n_pos = float(y_sorted.sum())
    n_neg = float(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = torch.arange(1, len(y_true) + 1, dtype=torch.float64)
    sum_pos_ranks = float((ranks * y_sorted).sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ==========================================================================
# Public probes
# ==========================================================================


def linear_probe_regression(
    components: JEPAComponents,
    cfg: JEPAConfig,
    *,
    target_kind: RegressionTarget = "future_return",
    n_train_batches: int = 200,
    n_val_batches: int = 50,
    ridge_alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    seed: int = 42,
    device: Optional[str] = None,
    probe_source: ProbeSource = "z_price",
) -> dict:
    """Ridge probe: z_price -> future_{return | volatility}.

    Fits at each alpha in ``ridge_alphas`` and picks the one with best val R^2.
    The alpha sweep is logarithmic by default; the probe is contractually
    "the best linear decoder of this target from z_price", and a single fixed
    alpha would conflate representation quality with regularization tuning.

    Reported numbers:
        val_r2                       headline — out-of-sample R^2 at best alpha
        train_r2_at_best_alpha       overfit check (gap from val_r2)
        val_mse                      raw MSE at best alpha
        baseline_mse                 MSE of "always predict y_train.mean()"
        val_mse_over_baseline        < 1.0 means the probe beats the trivial baseline
        sign_agreement               (return target only) fraction of samples where
                                     sign(y_pred) matches sign(y_true)
        per_alpha                    full sweep, for diagnostics

    Decision rule (applied in the caller):
        Compare `val_r2` from a trained-encoder call to `val_r2` from a call
        with `components = build_components(cfg)` (untrained). The trained
        encoder must beat untrained by more than sampling noise (~0.01 R^2)
        to count as progress.
    """
    if target_kind not in ("future_return", "future_volatility"):
        raise ValueError(
            f"target_kind must be future_return or future_volatility, got {target_kind!r}"
        )
    dev = resolve_device(device or "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = _collect_probe_pairs(
        components, cfg, "train", n_train_batches, dev, probe_source
    )
    val = _collect_probe_pairs(
        components, cfg, "val", n_val_batches, dev, probe_source
    )

    y_key = "y_return" if target_kind == "future_return" else "y_volatility"
    X_tr, X_va = _standardize(train["z"], val["z"])
    y_tr = train[y_key]
    y_va = val[y_key]

    per_alpha: list[dict] = []
    best_val_r2 = -float("inf")
    best_alpha = float(ridge_alphas[0])
    best_w: Optional[Tensor] = None
    best_intercept = 0.0

    for alpha in ridge_alphas:
        w, intercept = _ridge_closed_form(X_tr, y_tr, alpha)
        train_pred = X_tr @ w + intercept
        val_pred = X_va @ w + intercept
        train_r2 = _r2(y_tr, train_pred)
        val_r2 = _r2(y_va, val_pred)
        per_alpha.append(
            {"alpha": float(alpha), "train_r2": train_r2, "val_r2": val_r2}
        )
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_alpha = float(alpha)
            best_w = w
            best_intercept = intercept

    assert best_w is not None
    val_pred = X_va @ best_w + best_intercept
    val_mse = float(((y_va - val_pred) ** 2).mean())
    baseline_mse = float(((y_va - y_tr.mean()) ** 2).mean())  # predict train-mean
    train_r2_at_best = next(p["train_r2"] for p in per_alpha if p["alpha"] == best_alpha)

    out: dict = {
        "target_kind": target_kind,
        "probe_source": probe_source,
        "best_alpha": best_alpha,
        "val_r2": best_val_r2,
        "train_r2_at_best_alpha": train_r2_at_best,
        "val_mse": val_mse,
        "baseline_mse_predict_train_mean": baseline_mse,
        "val_mse_over_baseline": (
            val_mse / baseline_mse if baseline_mse > 1e-12 else float("nan")
        ),
        "per_alpha": per_alpha,
        "n_train": int(X_tr.size(0)),
        "n_val": int(X_va.size(0)),
        "latent_dim": int(X_tr.size(1)),
    }
    if target_kind == "future_return":
        # Direction agreement is only meaningful when y is centered near 0.
        # For volatility (strictly positive) it would degenerate to ~1.0.
        out["sign_agreement"] = float(((val_pred > 0) == (y_va > 0)).float().mean())
        out["positive_rate_val"] = float((y_va > 0).float().mean())
    return out


def linear_probe_direction(
    components: JEPAComponents,
    cfg: JEPAConfig,
    *,
    n_train_batches: int = 200,
    n_val_batches: int = 50,
    alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    seed: int = 42,
    device: Optional[str] = None,
    max_iter: int = 200,
    probe_source: ProbeSource = "z_price",
) -> dict:
    """Logistic probe: z_price -> sign(future cumulative log-return).

    Same alpha sweep pattern as the regression probe — best val accuracy
    picks the operating alpha. Reports:
        val_accuracy                 headline
        val_auroc                    threshold-independent ranking quality
        train_accuracy_at_best_alpha overfit check
        majority_class_accuracy      trivial baseline (always-predict-majority)
        positive_class_rate_val      class balance on val
        per_alpha                    full sweep

    Decision rule (applied in the caller):
        val_accuracy must beat majority_class_accuracy by more than the
        binomial standard error (~ sqrt(0.25/n_val) for balanced classes).
        Trained encoder must also beat untrained encoder on val_accuracy
        and val_auroc.
    """
    dev = resolve_device(device or "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = _collect_probe_pairs(
        components, cfg, "train", n_train_batches, dev, probe_source
    )
    val = _collect_probe_pairs(
        components, cfg, "val", n_val_batches, dev, probe_source
    )

    X_tr, X_va = _standardize(train["z"], val["z"])
    y_tr = train["y_sign"]
    y_va = val["y_sign"]

    per_alpha: list[dict] = []
    best_val_acc = -float("inf")
    best_alpha = float(alphas[0])

    for alpha in alphas:
        w, b = _logistic_lbfgs(X_tr, y_tr, alpha=alpha, max_iter=max_iter)
        train_scores = X_tr @ w + b
        val_scores = X_va @ w + b
        train_acc = float(((train_scores > 0) == (y_tr > 0)).float().mean())
        val_acc = float(((val_scores > 0) == (y_va > 0)).float().mean())
        val_auc = _auroc(y_va, val_scores)
        per_alpha.append(
            {
                "alpha": float(alpha),
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_auroc": val_auc,
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_alpha = float(alpha)

    best_row = next(r for r in per_alpha if r["alpha"] == best_alpha)
    pos_rate = float((y_va == 1).float().mean())
    majority_acc = max(pos_rate, 1.0 - pos_rate)

    return {
        "probe_source": probe_source,
        "best_alpha": best_alpha,
        "val_accuracy": best_val_acc,
        "val_auroc": best_row["val_auroc"],
        "train_accuracy_at_best_alpha": best_row["train_acc"],
        "majority_class_accuracy": majority_acc,
        "positive_class_rate_val": pos_rate,
        "per_alpha": per_alpha,
        "n_train": int(X_tr.size(0)),
        "n_val": int(X_va.size(0)),
        "latent_dim": int(X_tr.size(1)),
    }
