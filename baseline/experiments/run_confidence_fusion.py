"""Phase-2 offline confidence-guided fusion prototype.

This script never imports or modifies HYPIR. It consumes already verified
coeff_t=200 and coeff_t=50 PNG outputs, computes an image-derived structural
reliability heuristic from each LQ image, and writes all artifacts to a new
experiment directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import platform
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


CASES = [f"case{i}" for i in range(1, 6)]
METHODS = ["LQ", "HYPIR-200", "HYPIR-50", "fixed_50_50", "confidence_200_50", "confidence_200_LQ"]


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode != "RGB":
            raise ValueError(f"{path}: expected RGB, got {im.mode}")
        arr = np.asarray(im, dtype=np.uint8).copy()
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{path}: expected 3 channels")
    return arr


def save_rgb(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{path}: non-RGB output shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: non-finite output")
    arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path, format="PNG")


def robust_norm(x: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(x, [low, high])
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def confidence_map(lq: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Compute a smooth, image-derived structural reliability heuristic.

    Stable local structure is favored by gradient/Laplacian responses and
    local contrast, while a local noise proxy (high-frequency residual) softly
    suppresses isolated speckle-like responses. Percentile normalization and
    Gaussian smoothing make the map comparable across the five images.
    """
    gray = cv2.cvtColor(lq, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    # Work at half resolution for a broad, stable reliability field.
    h, w = gray.shape
    small = cv2.resize(gray, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    lap = np.abs(cv2.Laplacian(small, cv2.CV_32F, ksize=3))
    mean = cv2.blur(small, (9, 9))
    sqmean = cv2.blur(small * small, (9, 9))
    var = np.maximum(sqmean - mean * mean, 0)
    # High-frequency residual is used as a noise penalty rather than treating
    # every high-frequency pixel as trustworthy detail.
    smooth = cv2.GaussianBlur(small, (0, 0), 1.2)
    residual = np.abs(small - smooth)
    g = robust_norm(cv2.GaussianBlur(grad, (0, 0), 1.0))
    l = robust_norm(cv2.GaussianBlur(lap, (0, 0), 1.0))
    v = robust_norm(np.sqrt(var))
    n = robust_norm(cv2.GaussianBlur(residual, (0, 0), 1.0))
    # Moderate contrast/edges indicate structure; isolated residuals reduce
    # confidence. The final blur is deliberately much wider than a pixel.
    raw = np.clip(0.45 * g + 0.30 * l + 0.25 * v - 0.18 * n, 0, 1)
    raw = cv2.GaussianBlur(raw, (0, 0), 3.0)
    conf = cv2.resize(raw, (w, h), interpolation=cv2.INTER_CUBIC)
    conf = np.clip(cv2.GaussianBlur(conf, (0, 0), 9.0), 0, 1).astype(np.float32)
    # Keep the map away from hard binary switching; this is a blend weight.
    conf = (0.08 + 0.84 * conf).astype(np.float32)
    return conf, {"mean": float(conf.mean()), "p10": float(np.percentile(conf, 10)), "p50": float(np.percentile(conf, 50)), "p90": float(np.percentile(conf, 90))}


def save_confidence_visual(path: Path, conf: np.ndarray) -> None:
    rgb = (cv2.applyColorMap(np.clip(conf * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[:, :, ::-1]).astype(np.uint8)
    save_rgb(path, rgb)


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path, format="PNG")


def metrics(ref: np.ndarray, gt: np.ndarray) -> tuple[float, float, float, float]:
    psnr = float(peak_signal_noise_ratio(gt, ref, data_range=255))
    ssim = float(structural_similarity(gt, ref, channel_axis=2, data_range=255))
    # LPIPS is computed in the companion evaluator to avoid loading a large
    # network during each per-method pass; this placeholder is overwritten.
    return psnr, ssim, float("nan"), float(np.mean(np.abs(ref.astype(np.float32) - gt.astype(np.float32))))


def make_panel(images: list[tuple[str, np.ndarray]], path: Path, max_width: int = 1024) -> None:
    scale = min(1.0, max_width / max(im.shape[1] for _, im in images))
    thumbs = []
    for name, im in images:
        h, w = im.shape[:2]
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        thumb = Image.fromarray(im).resize(size, Image.Resampling.LANCZOS)
        thumbs.append((name, thumb))
    header = 48
    gap = 8
    canvas = Image.new("RGB", (sum(im.width for _, im in thumbs) + gap * (len(thumbs) - 1), header + max(im.height for _, im in thumbs)), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for name, im in thumbs:
        draw.text((x + 4, 5), name, fill=(20, 20, 20))
        canvas.paste(im, (x, header))
        x += im.width + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--root", type=Path, default=root)
    args = parser.parse_args()
    root = args.root.resolve()
    out_root = root / "baseline" / "experiments" / "confidence_fusion_v1"
    if out_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing experiment directory: {out_root}")
    (out_root / "confidence_maps").mkdir(parents=True)
    (out_root / "masks").mkdir()
    for sub in ("fixed_50_50", "confidence_200_50", "confidence_200_lq"):
        (out_root / "fusion" / sub).mkdir(parents=True)
    (out_root / "comparison").mkdir()
    stats: dict[str, dict[str, float]] = {}
    for case in CASES:
        lq_path = root / "baseline" / "input" / f"{case}_lq.jpg"
        gt_path = root / "csig_dataset" / "验证集" / f"{case}_gt.jpg"
        h200_path = root / "baseline" / "experiments" / "coeff_t_200" / "output" / "result" / f"{case}_lq.png"
        h50_path = root / "baseline" / "experiments" / "coeff_t_50" / "output" / "result" / f"{case}_lq.png"
        lq, gt, h200, h50 = map(load_rgb, (lq_path, gt_path, h200_path, h50_path))
        if len({x.shape for x in (lq, gt, h200, h50)}) != 1:
            raise ValueError(f"{case}: mismatched image shapes")
        conf, cstats = confidence_map(lq)
        stats[case] = cstats
        save_confidence_visual(out_root / "confidence_maps" / f"{case}.png", conf)
        hi = conf >= float(np.median(conf))
        save_mask(out_root / "masks" / f"{case}_high.png", hi)
        save_mask(out_root / "masks" / f"{case}_low.png", ~hi)
        fixed = 0.5 * h200.astype(np.float32) + 0.5 * h50.astype(np.float32)
        c2050 = conf[..., None] * h200.astype(np.float32) + (1 - conf[..., None]) * h50.astype(np.float32)
        c20lq = conf[..., None] * h200.astype(np.float32) + (1 - conf[..., None]) * lq.astype(np.float32)
        save_rgb(out_root / "fusion" / "fixed_50_50" / f"{case}.png", fixed)
        save_rgb(out_root / "fusion" / "confidence_200_50" / f"{case}.png", c2050)
        save_rgb(out_root / "fusion" / "confidence_200_lq" / f"{case}.png", c20lq)
        make_panel([("LQ", lq), ("HYPIR-200", h200), ("HYPIR-50", h50), ("50/50", np.rint(fixed).astype(np.uint8)), ("Conf 200/50", np.rint(c2050).astype(np.uint8)), ("Conf 200/LQ", np.rint(c20lq).astype(np.uint8)), ("GT", gt)], out_root / "comparison" / f"{case}.png")
    # Save a compact machine-readable confidence summary.
    with (out_root / "confidence_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "mean", "p10", "p50", "p90"])
        writer.writeheader()
        for case, row in stats.items():
            writer.writerow({"case": case, **{k: f"{v:.6f}" for k, v in row.items()}})
    print(f"Wrote confidence fusion artifacts to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
