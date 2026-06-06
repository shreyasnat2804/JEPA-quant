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


def timesfm_encode_fn(
    checkpoint_name: str,
    device: torch.device | str,
    *,
    channel: int = CLOSE_IDX,
    variant: str = "close",
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

    Returns:
        ``encode_fn(ctx_norm[B, L, F]) -> [B, hidden_size]``.

    Note: TimesFM internally re-normalizes each series by its own loc/scale, so
    feeding already-normalized log-returns is harmless (idempotent rescale); the
    representation reflects the series *shape*, which is what we want to decode.
    """
    if variant not in ("close", "concat6"):
        raise ValueError(f"variant must be 'close' or 'concat6', got {variant!r}")
    if variant == "concat6":
        # Intended approach (left as a hook for a stronger ceiling): loop over all
        # F channels, run each through TimesFM, mean-pool each, and concat to
        # [B, F*hidden_size]. Deferred — "close" first per the nb04 plan.
        raise NotImplementedError(
            "variant='concat6' (per-channel TimesFM, concatenated) is not yet "
            "implemented; use variant='close'."
        )

    try:
        from transformers import TimesFmModelForPrediction  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "timesfm_encode_fn requires TimesFM from transformers. Install with "
            "`pip install 'transformers>=4.48'` (the HF integration — do NOT pip "
            "install the `timesfm` package, which can pin numpy<2)."
        ) from exc

    dev = torch.device(device) if isinstance(device, str) else device
    model = TimesFmModelForPrediction.from_pretrained(checkpoint_name)
    model.to(dev).eval()

    def encode(ctx_norm: Tensor) -> Tensor:  # [B, L, F] -> [B, hidden_size]
        series = ctx_norm[..., channel]  # [B, L]
        # TimesFmModelForPrediction.forward takes past_values as a sequence of 1-D
        # tensors (variable-length series) and freq as a long tensor of indices.
        past_values = [series[b] for b in range(series.size(0))]
        freq = torch.zeros(series.size(0), dtype=torch.long, device=dev)
        with torch.no_grad():
            out = model(past_values=past_values, freq=freq, return_dict=True)
        # last_hidden_state: [B, n_patches, hidden_size]. Mean-pool over patches.
        return out.last_hidden_state.mean(dim=1)

    return encode
