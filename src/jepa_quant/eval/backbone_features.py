"""Feature extractors (``encode_fn``s) for the no-JEPA ceiling probe (nb04).

These build callables of the shape the linear-probe harness expects:
``encode_fn(ctx_norm[B, L, F]) -> [B, D]`` (see ``EncodeFn`` in linear_probe.py).
None of them involve JEPA, a projection head, or any training — they exist to
measure the *ceiling* of downstream signal that a frozen representation carries,
so we can decide whether a pretrained-backbone pivot is worth training units
BEFORE spending any.

Two extractors:

  * ``raw_feature_encode_fn``   — no encoder at all. Hand-crafted summary stats of
                                  the normalized context. Bounds "does ANY learned
                                  representation beat raw inputs?".
  * ``timesfm_encode_fn``       — RAW pretrained TimesFM hidden states (mean-pooled
                                  over patches). The ceiling JEPA inherits: a frozen
                                  backbone's representation is the most any frozen
                                  encoder + linear probe can decode.

TimesFM is loaded via the HuggingFace ``transformers`` integration
(``TimesFmModelForPrediction``), NOT the ``timesfm`` pip package. Rationale: the
pip package can drag jax/numpy constraints, whereas the transformers integration
rides Colab's already-installed numpy-2 stack (needs ``transformers>=4.48``). The
import is lazy (inside the factory) so ``import jepa_quant.eval.backbone_features``
works without transformers installed — mirrors ``MoiraiPriceEncoder``'s lazy
``uni2ts`` import in ``encoders/price_encoder.py``.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from .linear_probe import CLOSE_IDX

EncodeFn = Callable[[Tensor], Tensor]


def _load_timesfm_for_prediction(checkpoint_name: str, device: torch.device | str):
    """Lazy-load ``TimesFmModelForPrediction`` once, on ``device``, in eval mode.

    Shared by ``timesfm_encode_fn`` (taps hidden states), ``timesfm_forecast_fn``
    (taps the forecast head), and the nb05 shape instrumentation — so the HF
    integration import-guard and the ``from_pretrained`` call live in ONE place.
    The import is lazy (mirrors ``MoiraiPriceEncoder``'s ``uni2ts`` import) so
    ``import jepa_quant.eval.backbone_features`` works without transformers
    installed; it only fails when a TimesFM factory is actually called.

    Returns ``(model, device)`` with ``device`` resolved to a ``torch.device``.
    """
    try:
        from transformers import TimesFmModelForPrediction  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "TimesFM requires the HF transformers integration. Install with "
            "`pip install 'transformers>=4.48'` (do NOT pip install the `timesfm` "
            "package, which can pin numpy<2 and break Colab's numpy-2 stack)."
        ) from exc

    dev = torch.device(device) if isinstance(device, str) else device
    model = TimesFmModelForPrediction.from_pretrained(checkpoint_name)
    model.to(dev).eval()
    return model, dev


def raw_feature_encode_fn() -> EncodeFn:
    """No-encoder baseline: per-channel summary statistics of the context.

    For each of the F feature channels we take ``[mean, std, last, sum]`` over the
    L time steps, giving ``[B, F*4]`` (= [B, 24] for the 6 OHLCV channels). This is
    a deliberately simple, parameter-free, defensible bound: if a learned encoder
    cannot beat these four moments per channel, the representation adds nothing
    over the raw inputs. The context is already per-sample-normalized by the
    collector, so ``mean`` is ~0 and ``std`` is ~1 by construction — they still
    carry cross-channel and tail information that the probe can exploit.
    """

    def encode(ctx_norm: Tensor) -> Tensor:  # [B, L, F] -> [B, F*4]
        mean = ctx_norm.mean(dim=1)                       # [B, F]
        std = ctx_norm.std(dim=1, unbiased=False)         # [B, F]
        last = ctx_norm[:, -1, :]                         # [B, F]
        total = ctx_norm.sum(dim=1)                       # [B, F]
        return torch.cat([mean, std, last, total], dim=1)  # [B, 4F]

    return encode


def raw_sigma_encode_fn(channel: int = CLOSE_IDX) -> EncodeFn:
    """Single-feature probe: the std of one channel over the context window → [B, 1].

    On UN-normalized context (probe with ``normalize_context=False``) this single
    number IS the realized volatility of the window — the suspected strongest
    predictor of future volatility. nb05 uses it to test the hypothesis directly:
    if raw close-σ alone reproduces ~the +0.333 vol R² that the 24-d baseline got,
    then per-sample normalization (which divides this σ away) is the lever, not the
    backbone. On normalized context σ≈1 by construction, so this collapses to noise.
    """

    def encode(ctx: Tensor) -> Tensor:  # [B, L, F] -> [B, 1]
        return ctx[..., channel].std(dim=1, unbiased=False, keepdim=True)

    return encode


def timesfm_encode_fn(
    checkpoint_name: str,
    device: torch.device | str,
    *,
    channel: int = CLOSE_IDX,
    variant: str = "close",
    pool: str = "mean",
) -> EncodeFn:
    """RAW pretrained-TimesFM pooled hidden states as the probe representation.

    Loads ``TimesFmModelForPrediction`` ONCE (lazy import) and returns a closure
    that maps a normalized context window to mean-pooled transformer hidden
    states. We use ``last_hidden_state`` (the stacked-transformer output over input
    patches), NOT ``mean_predictions``/``full_predictions`` — we want the learned
    representation, not the forecast head.

    Args:
        checkpoint_name: HF id, e.g. ``"google/timesfm-2.0-500m-pytorch"``
            (patch_length=32, hidden_size=1280, 50 layers). Context L=64 → 2 patches.
        device: where to place the model and run inference.
        channel: feature column fed to TimesFM. Default ``CLOSE_IDX`` (close
            log-return) — TimesFM is univariate, so ``variant="close"`` probes the
            single most informative channel.
        variant: ``"close"`` (implemented) feeds one channel. ``"concat6"`` (hook
            for a stronger version) would run TimesFM per channel and concatenate
            the 6 pooled vectors → [B, 6*1280]; not implemented yet.
        pool: ``"mean"`` (default — averages over patch positions, the nb04
            behavior) or ``"last"`` (the last patch's hidden state). For a causal
            forecasting transformer the last position has attended over the whole
            window; nb05 compares the two to test whether mean-pooling dilutes
            the most-contextualized signal.

    Returns:
        ``encode_fn(ctx_norm[B, L, F]) -> [B, hidden_size]``.

    Note: TimesFM internally re-normalizes each series by its own loc/scale, so
    feeding already-normalized log-returns is harmless (idempotent rescale); the
    representation reflects the series *shape*, which is what we want to decode.
    """
    if variant not in ("close", "concat6"):
        raise ValueError(f"variant must be 'close' or 'concat6', got {variant!r}")
    if pool not in ("mean", "last"):
        raise ValueError(f"pool must be 'mean' or 'last', got {pool!r}")
    if variant == "concat6":
        # Intended approach (left as a hook for a stronger ceiling): loop over all
        # F channels, run each through TimesFM, mean-pool each, and concat to
        # [B, F*hidden_size]. Deferred — "close" first per the nb04 plan.
        raise NotImplementedError(
            "variant='concat6' (per-channel TimesFM, concatenated) is not yet "
            "implemented; use variant='close'."
        )

    model, dev = _load_timesfm_for_prediction(checkpoint_name, device)

    def encode(ctx_norm: Tensor) -> Tensor:  # [B, L, F] -> [B, hidden_size]
        series = ctx_norm[..., channel]  # [B, L]
        # TimesFmModelForPrediction.forward takes past_values as a sequence of 1-D
        # tensors (variable-length series) and freq as a long tensor of indices.
        past_values = [series[b] for b in range(series.size(0))]
        freq = torch.zeros(series.size(0), dtype=torch.long, device=dev)
        with torch.no_grad():
            out = model(past_values=past_values, freq=freq, return_dict=True)
        # last_hidden_state: [B, n_positions, hidden_size]. nb05 instrumentation
        # verifies n_positions on Colab (if the wrapper pads L=64 up to the model
        # context_length, mean-pool would average pad positions — see plan #5).
        hidden = out.last_hidden_state
        return hidden[:, -1, :] if pool == "last" else hidden.mean(dim=1)

    return encode


def timesfm_forecast_fn(
    checkpoint_name: str,
    device: torch.device | str,
    *,
    channel: int = CLOSE_IDX,
    horizon: int = 16,
) -> EncodeFn:
    """RAW pretrained-TimesFM point FORECAST (the model used as intended).

    Unlike ``timesfm_encode_fn`` (which taps internal hidden states for a probe),
    this returns the forecast head's mean point prediction — what TimesFM *thinks*
    the next ``horizon`` steps of the (normalized close log-return) series are. nb05
    overlays this against the true target window: if TimesFM cannot even forecast the
    continuation sensibly, that is direct visual evidence that our differenced /
    per-sample-normalized inputs are out-of-distribution for a level-pretrained
    forecaster — the leading explanation for the −0.134 probe result.

    Args:
        checkpoint_name: HF id, e.g. ``"google/timesfm-2.0-500m-pytorch"``.
        device: where to load the model and run inference.
        channel: feature column to forecast. Default ``CLOSE_IDX`` (close
            log-return), matching the probe's target channel.
        horizon: number of future steps to keep from the forecast head. The HF
            wrapper forecasts a fixed ``horizon_len`` (≥ our H=16); we slice the
            first ``horizon`` steps so the overlay aligns with the target window.

    Returns:
        ``forecast_fn(ctx_norm[B, L, F]) -> mean_forecast[B, horizon]``.

    NOTE: the output attribute (``mean_predictions``) is verified empirically by
    nb05's instrumentation cell before the overlay relies on it — same discipline
    as ``last_hidden_state`` in ``timesfm_encode_fn``.
    """
    model, dev = _load_timesfm_for_prediction(checkpoint_name, device)

    def forecast(ctx_norm: Tensor) -> Tensor:  # [B, L, F] -> [B, horizon]
        series = ctx_norm[..., channel]  # [B, L]
        past_values = [series[b] for b in range(series.size(0))]
        freq = torch.zeros(series.size(0), dtype=torch.long, device=dev)
        with torch.no_grad():
            out = model(past_values=past_values, freq=freq, return_dict=True)
        # mean_predictions: [B, horizon_len] point forecast. Slice to our H.
        return out.mean_predictions[:, :horizon]

    return forecast
