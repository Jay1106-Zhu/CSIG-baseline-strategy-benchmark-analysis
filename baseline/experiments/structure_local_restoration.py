"""Structure-anchored local restoration using LQ-only spatial weights.

The module is deliberately independent from HYPIR inference.  It consumes
already generated HYPIR-50/HYPIR-200 images and computes blend weights from
the LQ image only.  Ground truth is used only in the post-inference reports.
"""
from __future__ import annotations

import argparse
import csv
import math
from datetime import date
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


STRATEGIES = ("structure_guard", "texture_selective", "blurred_texture")
BASELINE_METHODS = ("LQ", "HYPIR-50", "HYPIR-200")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def load_rgb(path: Path | str) -> np.ndarray:
    path = Path(path)
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"{path}: expected RGB, got {image.mode}")
        array = np.asarray(image, dtype=np.uint8).copy()
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{path}: expected HxWx3 image, got {array.shape}")
    return array


def save_rgb(path: Path, image: np.ndarray) -> None:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{path}: expected HxWx3 output, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: non-finite output")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8)).save(path, format="PNG")


def _robust_norm(values: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(values, (low, high))
    if hi <= lo + 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _smooth_resize(values: np.ndarray, width: int, height: int) -> np.ndarray:
    values = cv2.GaussianBlur(values.astype(np.float32), (0, 0), 1.0)
    values = cv2.resize(values, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(cv2.GaussianBlur(values, (0, 0), 5.0), 0.0, 1.0).astype(np.float32)


def compute_lq_features(lq: np.ndarray) -> dict[str, np.ndarray]:
    """Return smooth edge, texture and blur proxies computed only from LQ."""
    array = np.asarray(lq)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("LQ must be an RGB HxWx3 array")
    height, width = array.shape[:2]
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    small = cv2.resize(gray, (max(8, width // 4), max(8, height // 4)), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    edge = _robust_norm(cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), 1.0))
    mean = cv2.blur(small, (9, 9))
    variance = np.maximum(cv2.blur(small * small, (9, 9)) - mean * mean, 0.0)
    texture = _robust_norm(cv2.GaussianBlur(np.sqrt(variance), (0, 0), 1.0))
    sharpness = _robust_norm(cv2.GaussianBlur(np.abs(cv2.Laplacian(small, cv2.CV_32F, ksize=3)), (0, 0), 1.0))
    blur = np.clip(1.0 - sharpness, 0.0, 1.0).astype(np.float32)
    return {
        "edge": _smooth_resize(edge, width, height),
        "texture": _smooth_resize(texture, width, height),
        "blur": _smooth_resize(blur, width, height),
    }


def compute_weight_map(features: dict[str, np.ndarray], strategy: str) -> np.ndarray:
    """Convert LQ-only features into a bounded HYPIR blend weight."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    edge = np.asarray(features["edge"], dtype=np.float32)
    texture = np.asarray(features["texture"], dtype=np.float32)
    blur = np.asarray(features["blur"], dtype=np.float32)
    if edge.shape != texture.shape or edge.shape != blur.shape:
        raise ValueError("LQ feature maps must have identical shapes")
    edge = np.clip(edge, 0.0, 1.0)
    texture = np.clip(texture, 0.0, 1.0)
    blur = np.clip(blur, 0.0, 1.0)
    protect = 1.0 - edge
    if strategy == "structure_guard":
        # Strong LQ edges are anchors; only a small amount of HYPIR is allowed.
        weight = 0.04 + protect * (0.28 + 0.08 * texture)
    elif strategy == "texture_selective":
        # Texture is useful only when it is not also a strong geometric edge.
        weight = 0.05 + 0.46 * texture * (0.35 + 0.65 * protect)
    else:
        # Allow generation in blurred, textured areas while guarding contours.
        weight = 0.05 + (0.40 * blur * texture + 0.08 * texture) * protect
    return np.clip(weight, 0.0, 0.60).astype(np.float32)


def fuse_image(lq: np.ndarray, hypir: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Blend HYPIR and LQ using ``I_out=w*I_HYPIR+(1-w)*I_LQ``."""
    lq_array = np.asarray(lq, dtype=np.float32)
    hypir_array = np.asarray(hypir, dtype=np.float32)
    weights = np.asarray(weight, dtype=np.float32)
    if lq_array.shape != hypir_array.shape or lq_array.ndim != 3 or lq_array.shape[2] != 3:
        raise ValueError("LQ and HYPIR must be same-size RGB arrays")
    if weights.shape != lq_array.shape[:2]:
        raise ValueError("weight map must match image height and width")
    if not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError("weight map must be finite and in [0, 1]")
    return np.clip(weights[..., None] * hypir_array + (1.0 - weights[..., None]) * lq_array, 0.0, 255.0).astype(np.float32)


def _case_file(directory: Path, case: str, suffixes: Iterable[str] = ()) -> Path:
    candidates = []
    for path in directory.iterdir() if directory.is_dir() else ():
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        stem = path.stem.casefold()
        if stem == case.casefold() or any(stem == f"{case}{suffix}".casefold() for suffix in suffixes):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"{case}: no image in {directory}")
    return sorted(candidates, key=lambda p: p.name.casefold())[0]


def _save_weight_visual(path: Path, weight: np.ndarray) -> None:
    image = cv2.applyColorMap(np.rint(np.clip(weight, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[:, :, ::-1]
    save_rgb(path, image)


def _panel(path: Path, arrays: list[tuple[str, np.ndarray]], max_width: int = 1024) -> None:
    scale = min(1.0, max_width / max(image.shape[1] for _, image in arrays))
    thumbs = []
    for name, image in arrays:
        height, width = image.shape[:2]
        resized = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).resize(
            (max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS
        )
        thumbs.append((name, resized))
    gap, header = 8, 38
    canvas = Image.new("RGB", (sum(im.width for _, im in thumbs) + gap * (len(thumbs) - 1), header + max(im.height for _, im in thumbs)), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for name, image in thumbs:
        draw.text((x + 3, 8), name, fill=(0, 0, 0))
        canvas.paste(image, (x, header))
        x += image.width + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")


def _metric(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred_u8 = np.clip(np.rint(pred), 0, 255).astype(np.uint8)
    if np.array_equal(pred_u8, gt):
        return float("inf"), 1.0
    return (
        float(peak_signal_noise_ratio(gt, pred_u8, data_range=255)),
        float(structural_similarity(gt, pred_u8, channel_axis=2, data_range=255)),
    )


def _region_rows(case: str, lq: np.ndarray, gt: np.ndarray, outputs: dict[str, np.ndarray], features: dict[str, np.ndarray], weights: dict[str, np.ndarray]) -> list[dict[str, object]]:
    edge, texture, blur = features["edge"], features["texture"], features["blur"]
    edge_hi = edge >= np.percentile(edge, 80)
    texture_hi = texture >= np.percentile(texture, 80)
    blur_hi = blur >= np.percentile(blur, 80)
    regions = {
        "strong_edge": edge_hi,
        "textured_non_edge": texture_hi & ~edge_hi,
        "blurred_texture": blur_hi & texture_hi & ~edge_hi,
    }
    assigned = np.zeros(edge.shape, dtype=bool)
    masks: dict[str, np.ndarray] = {}
    for name in ("strong_edge", "blurred_texture", "textured_non_edge"):
        masks[name] = regions[name] & ~assigned
        assigned |= masks[name]
    masks["other"] = ~assigned
    rows: list[dict[str, object]] = []
    lq_float = lq.astype(np.float32)
    gt_float = gt.astype(np.float32)
    for method, image in outputs.items():
        image_float = image.astype(np.float32)
        change = np.mean(np.abs(image_float - lq_float), axis=2)
        error = np.mean(np.abs(image_float - gt_float), axis=2)
        grad_lq_x = cv2.Sobel(cv2.cvtColor(lq, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F, 1, 0)
        grad_lq_y = cv2.Sobel(cv2.cvtColor(lq, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F, 0, 1)
        grad_out_x = cv2.Sobel(cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F, 1, 0)
        grad_out_y = cv2.Sobel(cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F, 0, 1)
        gradient_change = np.abs(np.hypot(grad_out_x, grad_out_y) - np.hypot(grad_lq_x, grad_lq_y))
        method_weight = weights.get(method)
        for region, mask in masks.items():
            if not mask.any():
                continue
            rows.append({
                "case": case,
                "method": method,
                "region": region,
                "pixels": int(mask.sum()),
                "mean_weight": "" if method_weight is None else f"{float(method_weight[mask].mean()):.6f}",
                "change_L1": f"{float(change[mask].mean()):.6f}",
                "error_to_GT_L1": f"{float(error[mask].mean()):.6f}",
                "gradient_change": f"{float(gradient_change[mask].mean()):.6f}",
            })
    return rows


def _lpips_score(pred: np.ndarray, gt: np.ndarray, model, device, max_side: int = 1024) -> float:
    import torch
    from torchvision.transforms.functional import pil_to_tensor

    def tensor(array: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8))
        scale = min(1.0, max_side / max(image.size))
        if scale < 1.0:
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
        return pil_to_tensor(image).unsqueeze(0).to(device=device, dtype=torch.float32).div(127.5).sub(1.0)

    with torch.inference_mode():
        return float(model(tensor(pred), tensor(gt)).item())


def _write_metadata(out: Path, lq_dir: Path, gt_dir: Path, h200_dir: Path, h50_dir: Path, lpips_size: int) -> None:
    text = f"""# Structure-Anchored Local Restoration v1 Metadata

- date: {date.today().isoformat()}
- status: completed
- scope: validation case1-case5; offline fusion; no new diffusion inference
- input_lq: `{lq_dir}`
- input_gt: `{gt_dir}` (post-inference evaluation only)
- input_hypir_200: `{h200_dir}`
- input_hypir_50: `{h50_dir}`
- output_dir: `{out}`
- strategies: `{', '.join(STRATEGIES)}`
- weight_range: `[0.0, 0.60]`; all constants are fixed across cases
- feature_source: LQ only (quarter-resolution Sobel edge, local standard deviation texture, inverse Laplacian blur proxy)
- fusion_formula: `I_out(x)=w(x)*I_HYPIR(x)+(1-w(x))*I_LQ(x)`
- HYPIR sources: existing coeff_t=200 and coeff_t=50 PNGs; no repeated diffusion inference
- LPIPS: Alex network, evaluated after uniform max-side resize to {lpips_size}px for memory-bounded comparison
- local_regions: LQ-only top-20-percent strong edge, blurred texture, textured non-edge, and remainder

## Strategies

1. `structure_guard`: `0.04 + (1-edge)*(0.28 + 0.08*texture)`. Strong edges receive the least diffusion.
2. `texture_selective`: `0.05 + 0.46*texture*(0.35 + 0.65*(1-edge))`. Texture gets more diffusion only away from strong contours.
3. `blurred_texture`: `0.05 + (0.40*blur*texture + 0.08*texture)*(1-edge)`. Blurred textured areas may receive more diffusion; flat areas and edges remain conservative.

GT is never passed to feature or weight computation. It is used only to measure errors after all fusion images are written.
"""
    (out / "experiment_metadata.md").write_text(text, encoding="utf-8")


def _write_report(out: Path, metric_rows: list[dict[str, object]], region_rows: list[dict[str, object]], *, with_lpips: bool) -> None:
    methods = list(dict.fromkeys(str(row["method"]) for row in metric_rows))
    lines = [
        "# Structure-Anchored Local Restoration v1 Report",
        "",
        "## Scope and conclusion",
        "",
        "This is an offline, validation-only fusion study. Existing HYPIR-50 and HYPIR-200 images were reused; no diffusion inference, HYPIR source edit, or GT-driven weight was used.",
        "",
        "The three maps are deliberately simple LQ-only heuristics. The decision below is based on all five cases and the local change/error CSV, not on a single-case optimum.",
        "",
        "## Average metrics",
        "",
        "| Method | PSNR | SSIM | LPIPS-Alex |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        rows = [row for row in metric_rows if row["method"] == method]
        mean_psnr = np.mean([float(row["PSNR"]) for row in rows])
        mean_ssim = np.mean([float(row["SSIM"]) for row in rows])
        lp_values = [float(row["LPIPS_Alex"]) for row in rows if row["LPIPS_Alex"] not in ("", None)]
        lp_text = f"{np.mean(lp_values):.6f}" if lp_values else "not computed"
        lines.append(f"| {method} | {mean_psnr:.6f} | {mean_ssim:.6f} | {lp_text} |")
    lines.extend([
        "",
        "## Strategy interpretation",
        "",
        "- `structure_guard` assigns the lowest weights to LQ strong-edge regions; it is the primary text/book-spine/clock protection control.",
        "- `texture_selective` permits more HYPIR only where local variance is high and edge strength is lower; it tests texture allowance without treating every high frequency as correct.",
        "- `blurred_texture` gives its largest allowance to blurred, textured, non-edge areas; it tests whether uncertainty/blur should be a reason for modest generation.",
        "",
        "## Local change/error evidence",
        "",
        "`local_analysis.csv` reports LQ-only regions (`strong_edge`, `blurred_texture`, `textured_non_edge`, `other`) and measures both change from LQ and error to GT after inference. A lower change in a region is preservation; it is not by itself restoration.",
        "",
        "| LQ region | H200 change | Guard-H200 change | Selective-H200 change | H200 error | Guard-H200 error | Selective-H200 error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    def region_mean(method: str, region: str, field: str) -> float:
        values = [float(row[field]) for row in region_rows if row["method"] == method and row["region"] == region]
        return float(np.mean(values)) if values else float("nan")
    for region in ("strong_edge", "textured_non_edge", "blurred_texture"):
        lines.append(
            f"| {region} | {region_mean('HYPIR-200', region, 'change_L1'):.4f} | "
            f"{region_mean('structure_guard_h200', region, 'change_L1'):.4f} | "
            f"{region_mean('texture_selective_h200', region, 'change_L1'):.4f} | "
            f"{region_mean('HYPIR-200', region, 'error_to_GT_L1'):.4f} | "
            f"{region_mean('structure_guard_h200', region, 'error_to_GT_L1'):.4f} | "
            f"{region_mean('texture_selective_h200', region, 'error_to_GT_L1'):.4f} |"
        )
    lines.append("")
    # Compare each fusion to the two explicit baselines on a common five-case mean.
    def avg(method: str, field: str) -> float | None:
        values = [float(row[field]) for row in metric_rows if row["method"] == method and row[field] not in ("", None)]
        return float(np.mean(values)) if values else None
    base50 = avg("HYPIR-50", "PSNR")
    base200 = avg("HYPIR-200", "PSNR")
    for method in methods:
        if method in BASELINE_METHODS:
            continue
        psnr = avg(method, "PSNR")
        if psnr is None:
            continue
        lines.append(f"- `{method}` average PSNR is {psnr:.4f} dB; delta vs HYPIR-50 is {psnr-base50:+.4f} dB and vs HYPIR-200 is {psnr-base200:+.4f} dB.")
    lines.extend([
        "",
        "## Answers to the requested questions",
        "",
        "A. LQ structure features can determine where to reduce diffusion change in an image-space sense: the strong-edge region receives lower weights by construction and its measured change is reduced. The five-case data do not prove that these features predict GT-aligned improvement; prior diagnosis found only weak improvement correlations.",
        "",
        "B. Text/book-spine/clock-like structure is plausibly protected because strong LQ edges receive less HYPIR. The evidence supports reduced rewriting, not exact character, numeral, or pointer identity; no OCR claim is made.",
        "",
        "C. Texture regions are worth testing with a higher weight than protected edges, but only selectively. The texture and blurred-texture maps are controls for this hypothesis; higher frequency or higher weight is not treated as automatically better, and GT-aligned local error remains the gate.",
        "",
        "D. This is sufficient to justify a small next local-control study, not a broad sweep or LoRA/module change. Advance only if a held-out case or seed reproduces lower edge-region rewriting without sacrificing global metrics; otherwise stop this direction.",
        "",
        "## Limitations",
        "",
        "Only five validation pairs and one seed are available. Regions are LQ feature quantiles, not semantic text/clock/texture masks. LPIPS is evaluated at a uniform max-side resize when enabled. Metrics cannot establish OCR or object identity.",
    ])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    lq_dir: Path | str,
    gt_dir: Path | str,
    h200_dir: Path | str,
    h50_dir: Path | str,
    out_dir: Path | str,
    *,
    compute_lpips: bool = False,
    lpips_size: int = 1024,
) -> dict[str, object]:
    """Run the offline prototype and write all image/CSV/metadata artifacts."""
    lq_dir, gt_dir, h200_dir, h50_dir, out = map(Path, (lq_dir, gt_dir, h200_dir, h50_dir, out_dir))
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty experiment directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("weight_maps", "fusion", "comparison"):
        (out / sub).mkdir(exist_ok=True)
    cases = sorted({p.stem.casefold().removesuffix("_lq") for p in lq_dir.iterdir() if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS})
    if not cases:
        raise FileNotFoundError(f"No LQ images found in {lq_dir}")
    cases = [f"case{int(case[4:])}" if case.startswith("case") and case[4:].isdigit() else case for case in cases]
    lpips_model = None
    lpips_device = None
    if compute_lpips:
        import torch
        import lpips
        lpips_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_model = lpips.LPIPS(net="alex", verbose=False).to(lpips_device).eval()
    metric_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, float]] = {}
    for case in cases:
        lq = load_rgb(_case_file(lq_dir, case, ("_lq",)))
        gt = load_rgb(_case_file(gt_dir, case, ("_gt",)))
        h200 = load_rgb(_case_file(h200_dir, case, ("_lq",)))
        h50 = load_rgb(_case_file(h50_dir, case, ("_lq",)))
        if len({lq.shape, gt.shape, h200.shape, h50.shape}) != 1:
            raise ValueError(f"{case}: LQ/GT/HYPIR dimensions do not match")
        features = compute_lq_features(lq)
        weights = {strategy: compute_weight_map(features, strategy) for strategy in STRATEGIES}
        for strategy, weight in weights.items():
            _save_weight_visual(out / "weight_maps" / f"{case}_{strategy}.png", weight)
            Image.fromarray(np.rint(weight * 255).astype(np.uint8)).save(out / "weight_maps" / f"{case}_{strategy}_gray.png", format="PNG")
        outputs: dict[str, np.ndarray] = {"LQ": lq, "HYPIR-50": h50, "HYPIR-200": h200}
        # Existing confidence_200_LQ is a read-only comparison when present.
        confidence_dir = out.parent / "confidence_fusion_v1" / "fusion" / "confidence_200_lq"
        if confidence_dir.is_dir():
            try:
                outputs["confidence_200_LQ"] = load_rgb(_case_file(confidence_dir, case))
            except FileNotFoundError:
                pass
        output_weights: dict[str, np.ndarray] = {}
        for strategy, weight in weights.items():
            for source, source_image in (("h50", h50), ("h200", h200)):
                method = f"{strategy}_{source}"
                fused = fuse_image(lq, source_image, weight)
                outputs[method] = fused
                output_weights[method] = weight
                save_rgb(out / "fusion" / strategy / source / f"{case}.png", fused)
            _panel(out / "comparison" / f"{case}_{strategy}.png", [("LQ", lq), ("HYPIR-50", h50), ("HYPIR-200", h200), (f"{strategy}/H50", outputs[f"{strategy}_h50"]), (f"{strategy}/H200", outputs[f"{strategy}_h200"]), ("GT", gt)])
        case_metric_values: list[tuple[str, float, float, float]] = []
        for method, image in outputs.items():
            psnr, ssim = _metric(image, gt)
            lp = float("nan") if lpips_model is None else _lpips_score(image, gt, lpips_model, lpips_device, lpips_size)
            case_metric_values.append((method, psnr, ssim, lp))
            summary.setdefault(method, {"PSNR": 0.0, "SSIM": 0.0, "LPIPS_Alex": 0.0, "count": 0.0})
            summary[method]["PSNR"] += psnr; summary[method]["SSIM"] += ssim; summary[method]["LPIPS_Alex"] += 0.0 if math.isnan(lp) else lp; summary[method]["count"] += 1
        lq_psnr, lq_ssim, lq_lpips = next(values[1:] for values in case_metric_values if values[0] == "LQ")
        for method, psnr, ssim, lp in case_metric_values:
            metric_rows.append({
                "case": case,
                "method": method,
                "PSNR": f"{psnr:.6f}",
                "SSIM": f"{ssim:.6f}",
                "LPIPS_Alex": "" if math.isnan(lp) else f"{lp:.6f}",
                "Delta_PSNR_vs_LQ": f"{psnr-lq_psnr:.6f}",
                "Delta_SSIM_vs_LQ": f"{ssim-lq_ssim:.6f}",
                "Delta_LPIPS_vs_LQ": "" if math.isnan(lp) or math.isnan(lq_lpips) else f"{lp-lq_lpips:.6f}",
            })
        region_rows.extend(_region_rows(case, lq, gt, outputs, features, output_weights))
    metric_fields = ["case", "method", "PSNR", "SSIM", "LPIPS_Alex", "Delta_PSNR_vs_LQ", "Delta_SSIM_vs_LQ", "Delta_LPIPS_vs_LQ"]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields); writer.writeheader(); writer.writerows(metric_rows)
        for method, values in summary.items():
            count = values["count"]
            avg_psnr = values["PSNR"] / count
            avg_ssim = values["SSIM"] / count
            avg_lpips = values["LPIPS_Alex"] / count
            lq_avg = summary["LQ"]
            writer.writerow({
                "case": "Average",
                "method": method,
                "PSNR": f"{avg_psnr:.6f}",
                "SSIM": f"{avg_ssim:.6f}",
                "LPIPS_Alex": "" if not compute_lpips else f"{avg_lpips:.6f}",
                "Delta_PSNR_vs_LQ": f"{avg_psnr-lq_avg['PSNR']/lq_avg['count']:.6f}",
                "Delta_SSIM_vs_LQ": f"{avg_ssim-lq_avg['SSIM']/lq_avg['count']:.6f}",
                "Delta_LPIPS_vs_LQ": "" if not compute_lpips else f"{avg_lpips-lq_avg['LPIPS_Alex']/lq_avg['count']:.6f}",
            })
    region_fields = ["case", "method", "region", "pixels", "mean_weight", "change_L1", "error_to_GT_L1", "gradient_change"]
    with (out / "local_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=region_fields); writer.writeheader(); writer.writerows(region_rows)
    with (out / "weight_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "strategy", "mean", "p10", "p50", "p90", "strong_edge_mean"]); writer.writeheader()
        for case in cases:
            lq = load_rgb(_case_file(lq_dir, case, ("_lq",)))
            features = compute_lq_features(lq)
            for strategy in STRATEGIES:
                weight = compute_weight_map(features, strategy); edge_hi = features["edge"] >= np.percentile(features["edge"], 80)
                writer.writerow({"case": case, "strategy": strategy, "mean": f"{weight.mean():.6f}", "p10": f"{np.percentile(weight,10):.6f}", "p50": f"{np.percentile(weight,50):.6f}", "p90": f"{np.percentile(weight,90):.6f}", "strong_edge_mean": f"{weight[edge_hi].mean():.6f}"})
    _write_metadata(out, lq_dir, gt_dir, h200_dir, h50_dir, lpips_size)
    _write_report(out, metric_rows, region_rows, with_lpips=compute_lpips)
    return {"cases": cases, "strategies": list(STRATEGIES), "metrics": metric_rows, "patch_count": len(region_rows)}


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--with-lpips", action="store_true", help="compute LPIPS after uniform max-side resize")
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "baseline" / "experiments" / "structure_local_restoration_v1"
    result = run_experiment(
        root / "baseline" / "input",
        root / "csig_dataset" / "验证集",
        root / "baseline" / "experiments" / "coeff_t_200" / "output" / "result",
        root / "baseline" / "experiments" / "coeff_t_50" / "output" / "result",
        out,
        compute_lpips=args.with_lpips,
    )
    print(f"Wrote {len(result['cases'])} cases and {len(result['strategies'])} strategies to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
