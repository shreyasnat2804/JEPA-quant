"""nb05 TimesFM diagnosis — prediction/forecast collection + plotting.

nb04 found RAW TimesFM (close channel) decodes future volatility at val R² −0.134,
*below* an untrained-random encoder (+0.097) and far below a 24-d moment baseline
(+0.333). This module supplies the reusable logic for nb05, which asks *why* and
makes it visible (per the user's "compare predictions, plot them to actual values").

Three collectors + two plotters, all forward-passes only (no training):

  * ``probe_regression_predictions`` — the (y_true, y_pred) points behind a probe's
    reported val R², for a predicted-vs-actual scatter. One scatter per
    representation shows the −0.134 *cloud* next to the +0.333 *slope*.
  * ``collect_timesfm_forecasts`` — TimesFM's actual point forecast (forecast head)
    vs the true target window, for overlay plots: does TimesFM, used as intended,
    even track our normalized log-return series?
  * ``instrument_timesfm_shapes`` — prints the real ``last_hidden_state`` shape /
    n_positions and mean-vs-last pool norms, settling whether the wrapper pads
    L=64 up to the model context length (which would corrupt mean-pooling).

Plotting deps are lazy-imported (matplotlib) so the non-plot collectors — and the
import itself — work in the download-free / matplotlib-free test environment.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

from ..config import JEPAConfig
from ..training.trainer import resolve_device
from .backbone_features import _load_timesfm_for_prediction
from .linear_probe import (
    CLOSE_IDX,
    EncodeFn,
    RegressionTarget,
    _build_split_loader,
    _collect_probe_pairs_fn,
    _fit_regression,
)

# Stage 1 default alpha grid; nb05 overrides with the wide grid for high-D reps.
_DEFAULT_ALPHAS: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)


# ==========================================================================
# Collectors (forward passes only)
# ==========================================================================


def probe_regression_predictions(
    encode_fn: EncodeFn,
    cfg: JEPAConfig,
    *,
    target_kind: RegressionTarget = "future_volatility",
    n_train_batches: int = 200,
    n_val_batches: int = 50,
    ridge_alphas: Sequence[float] = _DEFAULT_ALPHAS,
    seed: int = 42,
    device: Optional[str] = None,
    source_label: str = "custom",
    normalize_context: bool = True,
) -> dict:
    """Best-alpha ridge probe, returning the val (y_true, y_pred) for a scatter.

    Shares the exact collection + fitting path as ``probe_regression_features``
    (same ``_collect_probe_pairs_fn`` + ``_fit_regression``), so ``val_r2`` here
    equals the nb04 number at the same seed/grid — the scatter shows precisely the
    points behind that R². ``normalize_context`` is forwarded to the collector (set
    ``False`` to probe the raw un-normalized context, matching the nb05 lever).

    Returns CPU float Tensors ``y_true`` / ``y_pred`` (both in raw target units,
    e.g. volatility), plus ``val_r2``, ``train_r2``, ``best_alpha``, ``source_label``.
    """
    if target_kind not in ("future_return", "future_volatility"):
        raise ValueError(
            f"target_kind must be future_return or future_volatility, got {target_kind!r}"
        )
    dev = resolve_device(device or "cpu")
    torch.manual_seed(seed)

    train = _collect_probe_pairs_fn(
        encode_fn, cfg, "train", n_train_batches, dev, normalize_context
    )
    val = _collect_probe_pairs_fn(
        encode_fn, cfg, "val", n_val_batches, dev, normalize_context
    )

    out = _fit_regression(
        train, val, target_kind, ridge_alphas, return_val_predictions=True
    )
    return {
        "source_label": source_label,
        "target_kind": target_kind,
        "val_r2": out["val_r2"],
        "train_r2": out["train_r2_at_best_alpha"],
        "best_alpha": out["best_alpha"],
        "y_true": out["val_targets"],
        "y_pred": out["val_predictions"],
    }


def collect_timesfm_forecasts(
    cfg: JEPAConfig,
    forecast_fn: EncodeFn,
    *,
    n_samples: int = 9,
    channel: int = CLOSE_IDX,
    split: str = "val",
    device: Optional[str] = None,
) -> dict:
    """Collect (context, true target, TimesFM forecast) for ``n_samples`` windows.

    Everything is in the SAME normalized space the encoder/forecaster sees: the
    context is per-sample normalized exactly as ``PriceWindowDataset(normalize=True)``
    (mean/std over the context window, +1e-6), and the target is normalized with the
    *context's* mean/std — so the forecast (continuation of the normalized series)
    and the true target overlay on one axis.

    ``forecast_fn(ctx_norm[B,L,F]) -> [B, h]`` is any callable (real
    ``timesfm_forecast_fn`` or a stub in tests). Forecast and target are sliced to
    a common horizon length for the overlay.

    Returns CPU Tensors: ``context`` [n, L], ``target_true`` [n, H], ``forecast``
    [n, H] (all the close channel), plus ``channel``.
    """
    dev = resolve_device(device or "cpu")
    loader = _build_split_loader(cfg, split, normalize=False)

    ctxs: list[Tensor] = []
    tgts: list[Tensor] = []
    fcsts: list[Tensor] = []
    collected = 0

    with torch.no_grad():
        for batch in loader:
            if collected >= n_samples:
                break
            ctx_raw = batch["context"]  # [B, L, F] raw log-returns
            tgt_raw = batch["target"]   # [B, H, F] raw log-returns

            mu = ctx_raw.mean(dim=1, keepdim=True)           # [B, 1, F]
            sigma = ctx_raw.std(dim=1, keepdim=True) + 1e-6  # [B, 1, F]
            ctx_norm = (ctx_raw - mu) / sigma
            tgt_norm = (tgt_raw - mu) / sigma                # same μ/σ as the dataset

            fc = forecast_fn(ctx_norm.to(dev)).cpu()         # [B, h]
            h = min(fc.size(1), tgt_norm.size(1))            # align overlay length

            ctxs.append(ctx_norm[..., channel])              # [B, L]
            tgts.append(tgt_norm[..., channel][:, :h])       # [B, h]
            fcsts.append(fc[:, :h])                          # [B, h]
            collected += ctx_raw.size(0)

    if not ctxs:
        raise ValueError(f"No batches collected for split={split!r}")

    return {
        "context": torch.cat(ctxs)[:n_samples].float(),
        "target_true": torch.cat(tgts)[:n_samples].float(),
        "forecast": torch.cat(fcsts)[:n_samples].float(),
        "channel": channel,
    }


def instrument_timesfm_shapes(
    checkpoint_name: str,
    cfg: JEPAConfig,
    *,
    channel: int = CLOSE_IDX,
    n_samples: int = 8,
    split: str = "val",
    device: Optional[str] = None,
) -> dict:
    """Run ONE TimesFM forward and report what its output actually looks like.

    Settles root-cause #5 from the plan: the encode path does
    ``last_hidden_state.mean(dim=1)`` assuming dim=1 is ~2 input patches (L=64,
    patch=32). If the HF wrapper instead pads the context up to the model's
    ``context_length`` (e.g. 512), dim=1 is dominated by pad positions and the
    mean-pool is corrupted. This prints the real shape so we stop guessing.

    Returns a dict of: ``last_hidden_state_shape`` (tuple), ``n_positions``,
    ``context_length`` (L), ``mean_pool_feat_norm`` / ``last_token_feat_norm``
    (mean L2 norm across the batch, to see if mean vs last differ materially),
    ``output_fields`` (tensor-valued attrs on the output), and
    ``mean_predictions_shape`` if the forecast head is present.
    """
    model, dev = _load_timesfm_for_prediction(checkpoint_name, device)
    loader = _build_split_loader(cfg, split, normalize=False)

    batch = next(iter(loader))
    ctx_raw = batch["context"][:n_samples]                 # [n, L, F]
    mu = ctx_raw.mean(dim=1, keepdim=True)
    sigma = ctx_raw.std(dim=1, keepdim=True) + 1e-6
    ctx_norm = (ctx_raw - mu) / sigma
    series = ctx_norm[..., channel].to(dev)                # [n, L]

    past_values = [series[b] for b in range(series.size(0))]
    freq = torch.zeros(series.size(0), dtype=torch.long, device=dev)
    with torch.no_grad():
        out = model(past_values=past_values, freq=freq, return_dict=True)

    # Discover the available output fields FIRST — this function's whole job is to
    # verify the real API, so it must report what's there even if our assumed
    # ``last_hidden_state`` attribute is wrong (rather than crash before printing).
    # HF outputs are ModelOutput (an OrderedDict subclass): prefer .items(), which
    # yields only the set (non-None) fields; fall back to __dict__ for plain objects.
    try:
        out_items = list(out.items())
    except AttributeError:  # pragma: no cover - non-dict output object
        out_items = list(vars(out).items())
    info: dict = {
        "context_length": int(series.size(1)),
        "output_fields": [k for k, v in out_items if isinstance(v, torch.Tensor)],
    }
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        info["last_hidden_state_shape"] = None
        info["WARNING"] = (
            "out.last_hidden_state is missing — timesfm_encode_fn's pooling assumption "
            f"is wrong. Pick a hidden-state field from output_fields={info['output_fields']}."
        )
    else:
        info["last_hidden_state_shape"] = tuple(hidden.shape)
        info["n_positions"] = int(hidden.size(1))
        info["mean_pool_feat_norm"] = float(hidden.mean(dim=1).norm(dim=-1).mean())
        info["last_token_feat_norm"] = float(hidden[:, -1, :].norm(dim=-1).mean())
    if getattr(out, "mean_predictions", None) is not None:
        info["mean_predictions_shape"] = tuple(out.mean_predictions.shape)
    return info


# ==========================================================================
# Plotting (matplotlib lazy-imported)
# ==========================================================================


def _plt():
    """Import pyplot lazily so the module (and the collectors) import without
    matplotlib in the test env. Raises a clear error only if a plot is requested.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Plotting requires matplotlib. `pip install matplotlib` (already in "
            "the nb05 Colab install cell)."
        ) from exc
    return plt


def plot_pred_vs_actual_grid(
    results: Sequence[dict],
    *,
    ncols: int = 2,
    point_size: float = 6.0,
):
    """Scatter predicted-vs-actual for each result from ``probe_regression_predictions``.

    One subplot per representation: x = true target, y = probe prediction, with the
    identity line and the val R² annotated. A representation that "sees" the target
    hugs the diagonal (positive R²); a useless one is a round cloud / negative slope.
    Returns the matplotlib Figure.
    """
    plt = _plt()
    n = len(results)
    nrows = (n + ncols - 1) // ncols
    # squeeze=False -> axes is always a 2-D ndarray, so .ravel() is uniform across
    # 1x1 / 1xN / NxM layouts (plain subplots returns an Axes or a 1-D array otherwise).
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), squeeze=False
    )
    axes = axes.ravel()

    for ax, res in zip(axes, results):
        y_true = res["y_true"].cpu().numpy()
        y_pred = res["y_pred"].cpu().numpy()
        ax.scatter(y_true, y_pred, s=point_size, alpha=0.35, edgecolors="none")
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.7)  # identity
        ax.set_title(
            f"{res['source_label']}\nval R²={res['val_r2']:+.3f}  "
            f"(train {res['train_r2']:+.3f}, α={res['best_alpha']:g})",
            fontsize=10,
        )
        ax.set_xlabel("actual")
        ax.set_ylabel("predicted")

    for ax in axes[n:]:  # hide unused panes
        ax.set_visible(False)
    fig.suptitle(
        f"Predicted vs actual — {results[0]['target_kind']}", fontsize=12
    )
    fig.tight_layout()
    return fig


def plot_forecast_overlays(
    forecasts: dict,
    *,
    n: int = 9,
    ncols: int = 3,
):
    """Overlay TimesFM's point forecast on the true target window (normalized space).

    Each subplot: the normalized context (close channel) up to t=L, then the true
    target and the TimesFM forecast over the horizon. A sensible forecaster tracks
    the target's level/shape; a flat or wildly-off forecast is direct evidence the
    normalized log-return input is out-of-distribution for TimesFM. Returns the Figure.
    """
    plt = _plt()
    context = forecasts["context"].cpu().numpy()       # [N, L]
    target = forecasts["target_true"].cpu().numpy()    # [N, H]
    forecast = forecasts["forecast"].cpu().numpy()      # [N, H]
    n = min(n, context.shape[0])
    L, H = context.shape[1], target.shape[1]
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.0 * nrows), squeeze=False
    )
    axes = axes.ravel()

    ctx_x = range(L)
    fut_x = range(L, L + H)
    for i in range(n):
        ax = axes[i]
        ax.plot(ctx_x, context[i], color="0.5", lw=1, label="context")
        ax.plot(fut_x, target[i], color="C0", lw=1.5, label="actual")
        ax.plot(fut_x, forecast[i], color="C3", lw=1.5, ls="--", label="TimesFM")
        ax.axvline(L - 0.5, color="0.8", lw=0.8)
        ax.set_title(f"val sample {i}", fontsize=9)
        if i == 0:
            ax.legend(fontsize=8, loc="best")

    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("TimesFM forecast vs actual (normalized close log-returns)", fontsize=12)
    fig.tight_layout()
    return fig
