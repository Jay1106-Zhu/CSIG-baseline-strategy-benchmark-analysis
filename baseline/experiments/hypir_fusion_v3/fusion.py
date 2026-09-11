"""Laplacian-pyramid multi-band fusion on Y; Cb/Cr locked to LQ.

HYPIR's wavelet_reconstruction is 2-band (all high vs low) and only used
for color matching. This module uses a 3-band Laplacian split so mid-frequency
structure can be taken from a different source than fine detail.

Levels (OpenCV pyrDown, ~5x5 Gaussian, factor 2 each):
  high = L0 = G0 - expand(G1)          ~1-2 px
  mid  = expand(L1) + expand(L2)       ~4-8 px
  low  = expand(G3)                    coarser than ~8 px
"""
from __future__ import annotations

import cv2
import numpy as np

from baseline.experiments.hypir_fusion_v1.fusion import rgb_to_ycbcr, ycbcr_to_rgb


PYRAMID_LEVELS = 3
BAND_SCALES = {
    "high": "L0 = G0 - expand(G1); ~1-2 px (finest Laplacian)",
    "mid": "expand(L1)+expand(L2); ~4-8 px after two/three 2x blurs",
    "low": "expand(G3); content coarser than ~1/8 resolution",
}


def _as_y(y: np.ndarray) -> np.ndarray:
    array = np.asarray(y, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Y must be HxW, got {array.shape}")
    return array


def _expand_to(band: np.ndarray, gaussian: list[np.ndarray], start_level: int) -> np.ndarray:
    out = band
    for level in range(start_level, 0, -1):
        target = gaussian[level - 1]
        out = cv2.pyrUp(out, dstsize=(target.shape[1], target.shape[0]))
    return out


def decompose_y(y: np.ndarray, levels: int = PYRAMID_LEVELS) -> dict[str, np.ndarray]:
    """Split Y into full-resolution low / mid / high Laplacian bands."""
    if levels < 2:
        raise ValueError("need at least 2 pyramid levels for a mid band")
    current = _as_y(y)
    gaussian = [current]
    for _ in range(levels):
        gaussian.append(cv2.pyrDown(gaussian[-1]))
    laps = []
    for index in range(levels):
        up = cv2.pyrUp(gaussian[index + 1], dstsize=(gaussian[index].shape[1], gaussian[index].shape[0]))
        laps.append(gaussian[index] - up)
    high = laps[0]
    mid = np.zeros_like(current)
    for index in range(1, levels):
        mid = mid + _expand_to(laps[index], gaussian, index)
    low = _expand_to(gaussian[levels], gaussian, levels)
    return {"low": low.astype(np.float32), "mid": mid.astype(np.float32), "high": high.astype(np.float32)}


def reconstruct_y(bands: dict[str, np.ndarray]) -> np.ndarray:
    return (bands["low"] + bands["mid"] + bands["high"]).astype(np.float32)


def fuse_multiband(
    low_rgb: np.ndarray,
    mid_rgb: np.ndarray,
    high_rgb: np.ndarray,
    lq_rgb: np.ndarray,
) -> np.ndarray:
    """Y = low(low_src) + mid(mid_src) + high(high_src); Cb/Cr from LQ."""
    low_ycc = rgb_to_ycbcr(low_rgb)
    mid_ycc = rgb_to_ycbcr(mid_rgb)
    high_ycc = rgb_to_ycbcr(high_rgb)
    lq_ycc = rgb_to_ycbcr(lq_rgb)
    if len({low_ycc.shape, mid_ycc.shape, high_ycc.shape, lq_ycc.shape}) != 1:
        raise ValueError("all RGB inputs must be the same HxWx3 size")
    low_b = decompose_y(low_ycc[:, :, 0])
    mid_b = decompose_y(mid_ycc[:, :, 0])
    high_b = decompose_y(high_ycc[:, :, 0])
    out_y = np.clip(low_b["low"] + mid_b["mid"] + high_b["high"], 0.0, 255.0)
    out_ycc = np.stack((out_y, lq_ycc[:, :, 1], lq_ycc[:, :, 2]), axis=2)
    return ycbcr_to_rgb(out_ycc).astype(np.float32)


def blend_rgb(base: np.ndarray, detail: np.ndarray, alpha: float) -> np.ndarray:
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError(f"alpha must be a non-negative finite scalar, got {alpha}")
    base_y = rgb_to_ycbcr(base)[:, :, 0]
    detail_y = rgb_to_ycbcr(detail)[:, :, 0]
    mixed = np.clip(base_y + float(alpha) * (detail_y - base_y), 0.0, 255.0)
    ycc = rgb_to_ycbcr(base)
    ycc = np.stack((mixed, ycc[:, :, 1], ycc[:, :, 2]), axis=2)
    return ycbcr_to_rgb(ycc).astype(np.float32)


def band_gray(band: np.ndarray, *, residual: bool) -> np.ndarray:
    """Visualize a Y band as RGB. Residuals are offset to 128."""
    array = np.asarray(band, dtype=np.float32)
    if residual:
        pixels = np.clip(array + 128.0, 0.0, 255.0)
    else:
        pixels = np.clip(array, 0.0, 255.0)
    gray = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)
