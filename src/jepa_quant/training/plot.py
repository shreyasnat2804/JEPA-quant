"""Loss-term diagnostics for a JEPA training run.

``Trainer.train()`` returns a ``history`` list of per-log-step dicts holding the
raw, *unweighted* loss components (``jepa``, ``reg/v``, ``reg/c``,
``reg/z_std_mean``) plus ``val_jepa`` on eval steps. The aggregate ``loss`` can
fall steadily while the actual objective (``jepa``/``val_jepa``) regresses,
because a regularizer term dominates the sum. These plots make that visible by
graphing each term's *weighted* contribution (λ·term) against the objective.

Usage (thin notebook driver)::

    history = trainer.train()
    from jepa_quant.training.plot import plot_loss_terms
    plot_loss_terms(history, cfg)            # cfg = the JEPAConfig used to train
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt

from ..config import JEPAConfig


def _series(history: Sequence[dict], key: str) -> tuple[list[int], list[float]]:
    """Steps/values for ``key``, skipping records that lack it (e.g. val_jepa
    only appears on eval steps)."""
    steps = [r["step"] for r in history if key in r]
    vals = [r[key] for r in history if key in r]
    return steps, vals


def _phase2_step(history: Sequence[dict]) -> int | None:
    """First step at phase 2 — drawn as a vertical marker so the LoRA/encoder
    unfreeze is visible against the curves."""
    for r in history:
        if r.get("phase") == 2:
            return r["step"]
    return None


def plot_loss_terms(
    history: Sequence[dict],
    cfg: JEPAConfig,
    *,
    save_path: str | None = None,
):
    """Three-panel diagnostic of a training ``history``.

    1. Weighted contributions (log y): ``jepa``, ``λ_v·v``, ``λ_c·c``. A term
       sitting far above ``jepa`` is eating the gradient budget.
    2. Share of total loss (%): stacked, the headline view — if one band fills
       most of the plot, the loss is mostly that term, not prediction.
    3. The actual objective: train ``jepa`` and ``val_jepa`` (left axis) vs
       ``z_std_mean`` (right axis). Rising jepa + flat z_std ≈ 1.0 = the
       regularizer is winning while prediction degrades.
    """
    lam_v = cfg.vicreg.lambda_v
    lam_c = cfg.vicreg.lambda_c

    steps, jepa = _series(history, "jepa")
    _, v_raw = _series(history, "reg/v")
    _, c_raw = _series(history, "reg/c")
    _, z_std = _series(history, "reg/z_std_mean")
    wv = [lam_v * x for x in v_raw]
    wc = [lam_c * x for x in c_raw]
    total = [j + a + b for j, a, b in zip(jepa, wv, wc)]

    val_steps, val_jepa = _series(history, "val_jepa")
    p2 = _phase2_step(history)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    # 1 — weighted contributions, log scale (terms span orders of magnitude).
    ax1.plot(steps, jepa, label="jepa", color="C3", lw=2)
    ax1.plot(steps, wv, label=f"λ_v·v  (λ_v={lam_v:g})", color="C0", lw=2)
    ax1.plot(steps, wc, label=f"λ_c·c  (λ_c={lam_c:g})", color="C1", lw=2)
    ax1.set_yscale("log")
    ax1.set_ylabel("weighted contribution")
    ax1.set_title("Loss terms — actual gradient budget (log scale)")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # 2 — percent share of total, stacked. The confirmation panel.
    share_j = [100 * j / t for j, t in zip(jepa, total)]
    share_v = [100 * a / t for a, t in zip(wv, total)]
    share_c = [100 * b / t for b, t in zip(wc, total)]
    ax2.stackplot(
        steps,
        share_j,
        share_v,
        share_c,
        labels=["jepa", "λ_v·v", "λ_c·c"],
        colors=["C3", "C0", "C1"],
        alpha=0.85,
    )
    ax2.set_ylabel("share of total loss (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("Where the loss actually goes")
    ax2.legend(loc="center right")

    # 3 — the objective vs the variance constraint.
    ax3.plot(steps, jepa, label="train jepa", color="C3", lw=2)
    if val_jepa:
        ax3.plot(val_steps, val_jepa, "o-", label="val_jepa", color="C4", lw=2)
    ax3.set_ylabel("JEPA loss (1 − cos)")
    ax3.set_xlabel("step")
    ax3.set_title("Objective (lower = better) vs z_std")
    ax3.grid(True, alpha=0.3)
    ax3b = ax3.twinx()
    ax3b.plot(steps, z_std, "--", label="z_std_mean", color="C2", alpha=0.7)
    ax3b.axhline(cfg.vicreg.gamma, color="C2", ls=":", alpha=0.4)
    ax3b.set_ylabel("z_std_mean (target = γ)")
    # Skip matplotlib's auto "_child"-labelled artists (e.g. the γ axhline).
    lines = [
        ln
        for ln in ax3.get_lines() + ax3b.get_lines()
        if not ln.get_label().startswith("_")
    ]
    ax3.legend(lines, [ln.get_label() for ln in lines], loc="best")

    if p2 is not None:
        for ax in (ax1, ax2, ax3):
            ax.axvline(p2, color="k", ls="--", alpha=0.4)
        ax1.text(p2, ax1.get_ylim()[1], " phase 2", va="top", fontsize=8, alpha=0.6)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
