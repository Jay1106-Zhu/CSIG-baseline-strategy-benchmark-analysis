"""Residual-confidence HYPIR fusion primitives (F1 / fusion_v2).

Reuses v1 Sobel structure mask and Y-only fusion. Adds:

    conf = 1 - normalize(|H200 - LQ|)
    M_final = M_struct                         # groups A/B, original fusion_v1
    M_final = M_struct * (conf ** gamma)       # gamma in {0.5, 1, 2}

No new network, no GT in the mask, no HYPIR source edits.
"""
from __future__ import annotations

import numpy as np
import cv2


EPS = 1e-6
DEFAULT_RESIDUAL_BLUR_SIGMA = 8.0
DEFAULT_LOW_PERCENTILE = 1.0
DEFAULT_HIGH_PERCENTILE = 99.0

MASK_GROUPS: dict[str, dict[str, object]] = {
    "A": {
        "key": "struct",
        "gamma": None,
        "formula": "M_struct",
        "label": "fusion_v1",
        "method_a": "fusion_A",
        "method_b": "fusion_B",
    },
    "B": {
        "key": "struct",
        "gamma": None,
        "formula": "M_struct",
        "label": "M_struct",
        "method_a": "fusion_A",
        "method_b": "fusion_B",
    },
    "C": {
        "key": "conf",
        "gamma": 1.0,
        "formula": "M_struct * conf",
        "label": "M_struct * conf",
        "method_a": "fusion_A_conf",
        "method_b": "fusion_B_conf",
    },
    "D": {
        "key": "conf2",
        "gamma": 2.0,
        "formula": "M_struct * conf^2",
        "label": "M_struct * conf^2",
        "method_a": "fusion_A_conf2",
        "method_b": "fusion_B_conf2",
    },
}

# Unique masks actually written. A and B share `struct`. gamma=0.5 is extra.
UNIQUE_MASK_VARIANTS: tuple[dict[str, object], ...] = (
    {
        "key": "struct",
        "gamma": None,
        "formula": "M_struct",
        "method_a": "fusion_A",
        "method_b": "fusion_B",
        "folder_a": "A_struct",
        "folder_b": "B_struct",
        "final_name": "final_struct",
        "groups": ("A", "B"),
    },
    {
        "key": "conf05",
        "gamma": 0.5,
        "formula": "M_struct * conf^0.5",
        "method_a": "fusion_A_conf05",
        "method_b": "fusion_B_conf05",
        "folder_a": "A_conf05",
        "folder_b": "B_conf05",
        "final_name": "final_conf05",
        "groups": (),
    },
    {
        "key": "conf",
        "gamma": 1.0,
        "formula": "M_struct * conf",
        "method_a": "fusion_A_conf",
        "method_b": "fusion_B_conf",
        "folder_a": "A_conf",
        "folder_b": "B_conf",
        "final_name": "final_conf",
        "groups": ("C",),
    },
    {
        "key": "conf2",
        "gamma": 2.0,
        "formula": "M_struct * conf^2",
        "method_a": "fusion_A_conf2",
        "method_b": "fusion_B_conf2",
        "folder_a": "A_conf2",
        "folder_b": "B_conf2",
        "final_name": "final_conf2",
        "groups": ("D",),
    },
)


def compute_residual_confidence(
    lq_rgb: np.ndarray,
    h200_rgb: np.ndarray,
    *,
    blur_sigma: float = DEFAULT_RESIDUAL_BLUR_SIGMA,
    low_percentile: float = DEFAULT_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_HIGH_PERCENTILE,
    return_diagnostics: bool = False,
):
    """Build a 0-1 residual confidence map.

    conf = 1 - normalize(mean_c |H200 - LQ|)

    Large H200-LQ difference without supporting structure is treated as
    hallucination and receives low confidence. Uniform residual (including
    identical images) has no spatial distrust signal, so confidence stays 1.
    """
    lq = np.asarray(lq_rgb, dtype=np.float32)
    h200 = np.asarray(h200_rgb, dtype=np.float32)
    if lq.shape != h200.shape or lq.ndim != 3 or lq.shape[2] != 3:
        raise ValueError(f"LQ and H200 must be same-size RGB, got {lq.shape} vs {h200.shape}")
    residual = np.mean(np.abs(h200 - lq), axis=2).astype(np.float32)
    residual_blur = residual
    if blur_sigma and float(blur_sigma) > 0:
        residual_blur = cv2.GaussianBlur(residual, (0, 0), float(blur_sigma)).astype(np.float32)
    lo = float(np.percentile(residual_blur, low_percentile))
    hi = float(np.percentile(residual_blur, high_percentile))
    if hi - lo <= EPS:
        residual_norm = np.zeros_like(residual_blur, dtype=np.float32)
    else:
        residual_norm = np.clip((residual_blur - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    conf = (1.0 - residual_norm).astype(np.float32)
    if not return_diagnostics:
        return conf
    diagnostics = {
        "residual": residual,
        "residual_blur": residual_blur,
        "residual_norm": residual_norm,
        "lo": lo,
        "hi": hi,
    }
    return conf, diagnostics


def combine_masks(struct_mask: np.ndarray, conf: np.ndarray, gamma: float | None = None) -> np.ndarray:
    """M_final = M_struct if gamma is None, else M_struct * (conf ** gamma)."""
    struct = np.clip(np.asarray(struct_mask, dtype=np.float32), 0.0, 1.0)
    if struct.ndim != 2:
        raise ValueError(f"structure mask must be HxW, got {struct.shape}")
    if not np.isfinite(struct).all():
        raise ValueError("structure mask must be finite")
    if gamma is None:
        return struct.astype(np.float32)
    if not np.isfinite(gamma) or float(gamma) < 0:
        raise ValueError(f"gamma must be a non-negative finite scalar or None, got {gamma}")
    confidence = np.clip(np.asarray(conf, dtype=np.float32), 0.0, 1.0)
    if confidence.shape != struct.shape:
        raise ValueError(f"confidence shape {confidence.shape} != structure shape {struct.shape}")
    if not np.isfinite(confidence).all():
        raise ValueError("confidence mask must be finite")
    if float(gamma) == 0.0:
        return struct.astype(np.float32)
    out = struct * np.power(confidence, float(gamma))
    return np.clip(out, 0.0, 1.0).astype(np.float32)
