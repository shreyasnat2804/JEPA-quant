"""Regularization package.

Importing this package registers every built-in anti-collapse method (VICReg,
soft codebook) into the registry so they are selectable by name via
``build_regularizer`` / ``cfg.regularizer``.
"""

from __future__ import annotations

from .base import Regularizer, build_regularizer, register_regularizer, registered_regularizers
from . import vicreg  # noqa: F401  (registers "vicreg")
from . import codebook  # noqa: F401  (registers "codebook")

__all__ = [
    "Regularizer",
    "build_regularizer",
    "register_regularizer",
    "registered_regularizers",
]
