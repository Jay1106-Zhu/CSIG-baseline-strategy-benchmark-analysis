"""Output-space HYPIR fusion primitives.

HYPIR-200 is treated as a detail candidate. LQ / HYPIR-50 are structure
anchors. The structure mask is a classical Sobel agreement map: it stays
near 1 when LQ and H200 share the same contour, and drops when H200
introduces a new edge. Fusion is performed on Y only; LQ chroma is kept.
"""
from __future__ import annotations

import numpy as np
import cv2


EPS = 1e-6
# Sobel magnitude on Y in [0, 1]. Below this, gradient direction is undefined
# and is treated as "no disagreement" rather than as a penalty.
COSINE_FLAT_TAU = 0.05
DEFAULT_BLUR_SIGMA = 1.5

DEFAULT_SCENE_ALPHA: dict[str, float] = {
    "text": 0.25,
    "book": 0.30,
    "bird": 0.12,
    "plant": 0.08,
    "clock": 0.30,
}

CASE_TO_SCENE: dict[str, str] = {
    "case1": "text",
    "case2": "book",
    "case3": "bird",
    "case4": "plant",
    "case5": "clock",
}

SCENE_NOTES: dict[str, str] = {
    "text": "中文文字; alpha=0.25",
    "book": "书脊文字; alpha=0.30",
    "bird": "鸟/水面; recommended 0.10-0.15, default 0.12",
    "plant": "密集绿植; recommended 0.05-0.10, default 0.08",
    "clock": "钟表; alpha=0.30",
}


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """OpenCV RGB→YCrCb, float32 in 0-255."""
    array = np.clip(np.rint(np.asarray(rgb)), 0, 255).astype(np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {array.shape}")
    return cv2.cvtColor(array, cv2.COLOR_RGB2YCrCb).astype(np.float32)


def ycbcr_to_rgb(ycrcb: np.ndarray) -> np.ndarray:
    """OpenCV YCrCb→RGB uint8."""
    array = np.clip(np.rint(np.asarray(ycrcb)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(array, cv2.COLOR_YCrCb2RGB)


def rgb_to_y(rgb: np.ndarray) -> np.ndarray:
    return rgb_to_ycbcr(rgb)[:, :, 0]


def _unit_y(y: np.ndarray) -> np.ndarray:
    array = np.asarray(y, dtype=np.float32)
    if array.ndim == 3:
        array = array[:, :, 0]
    peak = float(np.max(array)) if array.size else 0.0
    if peak > 1.5:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def _sobel_pair(y_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = cv2.Sobel(y_unit, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y_unit, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return gx, gy, mag


def compute_structure_mask(
    lq_y: np.ndarray,
    h200_y: np.ndarray,
    *,
    blur_sigma: float = DEFAULT_BLUR_SIGMA,
    return_diagnostics: bool = False,
):
    """Build a 0-1 structure agreement mask from LQ and H200 Y channels.

    Pipeline (all classical, no semantic model):
    1. resize Y to 1/4
    2. Sobel gradients
    3. direction cosine, clipped to [0, 1]
    4. gradient magnitude similarity
    5. mask = clip(cos, 0, 1) * gradient_similarity
    6. Gaussian blur, resize back to full resolution

    Consistent structure → mask near 1. A contour that appears only in
    H200 → mask decreases.
    """
    y1 = _unit_y(lq_y)
    y2 = _unit_y(h200_y)
    if y1.shape != y2.shape:
        raise ValueError(f"Y shapes differ: {y1.shape} vs {y2.shape}")
    height, width = y1.shape
    small_w = max(1, width // 4)
    small_h = max(1, height // 4)
    small1 = cv2.resize(y1, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small2 = cv2.resize(y2, (small_w, small_h), interpolation=cv2.INTER_AREA)
    gx1, gy1, mag1 = _sobel_pair(small1)
    gx2, gy2, mag2 = _sobel_pair(small2)

    dot = gx1 * gx2 + gy1 * gy2
    cos_raw = np.clip(dot / (mag1 * mag2 + EPS), 0.0, 1.0)
    # Direction is undefined in flat areas. Do not treat 0/eps as disagreement
    # when neither image has a contour; do treat it as disagreement when H200
    # suddenly grows a contour that LQ does not have.
    defined = np.maximum(mag1, mag2) / (np.maximum(mag1, mag2) + COSINE_FLAT_TAU)
    cos = defined * cos_raw + (1.0 - defined)
    gradient_similarity = 1.0 - np.abs(mag1 - mag2) / (mag1 + mag2 + EPS)
    mask_small = np.clip(cos * gradient_similarity, 0.0, 1.0).astype(np.float32)
    if blur_sigma > 0:
        mask_small = cv2.GaussianBlur(mask_small, (0, 0), float(blur_sigma))
    mask = cv2.resize(mask_small, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    if not return_diagnostics:
        return mask
    diagnostics = {
        "cos": cos.astype(np.float32),
        "gradient_similarity": gradient_similarity.astype(np.float32),
        "mag_lq": mag1.astype(np.float32),
        "mag_h200": mag2.astype(np.float32),
        "mask_small": mask_small.astype(np.float32),
    }
    return mask, diagnostics


def fuse_detail(
    base_rgb: np.ndarray,
    h200_rgb: np.ndarray,
    lq_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Y: base + alpha * mask * (H200 - base). Cb/Cr: LQ. Returns float RGB 0-255."""
    base = np.asarray(base_rgb)
    h200 = np.asarray(h200_rgb)
    lq = np.asarray(lq_rgb)
    weights = np.asarray(mask, dtype=np.float32)
    if base.shape != h200.shape or base.shape != lq.shape or base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("base, H200 and LQ must be same-size RGB arrays")
    if weights.shape != base.shape[:2]:
        raise ValueError("mask must match image height and width")
    if not np.isfinite(weights).all() or np.any((weights < -1e-6) | (weights > 1.0 + 1e-6)):
        raise ValueError("mask must be finite and in [0, 1]")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError(f"alpha must be a non-negative finite scalar, got {alpha}")

    base_ycc = rgb_to_ycbcr(base)
    h200_ycc = rgb_to_ycbcr(h200)
    lq_ycc = rgb_to_ycbcr(lq)
    gain = np.float32(alpha) * np.clip(weights, 0.0, 1.0)
    out_y = base_ycc[:, :, 0] + gain * (h200_ycc[:, :, 0] - base_ycc[:, :, 0])
    out_ycc = np.stack(
        (np.clip(out_y, 0.0, 255.0), lq_ycc[:, :, 1], lq_ycc[:, :, 2]),
        axis=2,
    )
    return ycbcr_to_rgb(out_ycc).astype(np.float32)


def fuse_scheme_a(lq: np.ndarray, h200: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """base=LQ, detail=H200-LQ."""
    return fuse_detail(lq, h200, lq, mask, alpha)


def fuse_scheme_b(lq: np.ndarray, h50: np.ndarray, h200: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """base=H50, detail=H200-H50. Chroma still from LQ."""
    return fuse_detail(h50, h200, lq, mask, alpha)


def scene_alpha_for_case(
    case: str,
    overrides: dict[str, float] | None = None,
    default_alpha: float = 0.15,
) -> tuple[str, float]:
    """Manual scene lookup. Not a classifier."""
    key = case.casefold()
    scene = CASE_TO_SCENE.get(key, "unknown")
    table = dict(DEFAULT_SCENE_ALPHA)
    if overrides:
        table.update({name.casefold(): float(value) for name, value in overrides.items()})
    if scene == "unknown":
        return scene, float(default_alpha)
    return scene, float(table[scene])
