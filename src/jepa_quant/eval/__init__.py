"""Evaluation utilities for JEPA-quant checkpoints.

Stage 0 diagnostics (decides whether the JEPA loss is non-degenerate):
shuffled-target control, baselines, collapse audit, level-dependence test,
regime clustering.

Stage 1 linear probe (PRIMARY progress metric — quantifies downstream utility
of z_price): ridge regression to future returns/volatility, logistic
regression to future return direction.
"""

from .diagnostics import (
    load_checkpoint,
    shuffled_target_control,
    compute_baselines,
    collapse_audit,
    level_dependence_test,
    regime_clustering,
)
from .linear_probe import (
    linear_probe_regression,
    linear_probe_direction,
)

__all__ = [
    "load_checkpoint",
    "shuffled_target_control",
    "compute_baselines",
    "collapse_audit",
    "level_dependence_test",
    "regime_clustering",
    "linear_probe_regression",
    "linear_probe_direction",
]
