"""Evaluation utilities for JEPA-quant checkpoints.

Stage 0 diagnostics: shuffled-target control, baselines, collapse audit,
level-dependence test, regime clustering.

Stage 1 (linear probe): in linear_probe.py.
"""

from .diagnostics import (
    load_checkpoint,
    shuffled_target_control,
    compute_baselines,
    collapse_audit,
    level_dependence_test,
    regime_clustering,
)

__all__ = [
    "load_checkpoint",
    "shuffled_target_control",
    "compute_baselines",
    "collapse_audit",
    "level_dependence_test",
    "regime_clustering",
]
