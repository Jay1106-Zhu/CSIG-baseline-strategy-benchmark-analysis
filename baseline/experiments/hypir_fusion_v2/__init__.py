"""HYPIR output-space residual-confidence fusion v2 (F1)."""

from .fusion import (
    MASK_GROUPS,
    UNIQUE_MASK_VARIANTS,
    combine_masks,
    compute_residual_confidence,
)

__all__ = [
    "MASK_GROUPS",
    "UNIQUE_MASK_VARIANTS",
    "combine_masks",
    "compute_residual_confidence",
]
