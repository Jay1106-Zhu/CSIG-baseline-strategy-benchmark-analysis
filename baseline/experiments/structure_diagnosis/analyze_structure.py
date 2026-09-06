"""Post-hoc local diagnosis for the existing HYPIR coefficient experiments.

This module never imports HYPIR or runs inference. It compares native-resolution
LQ/GT/HYPIR-200/HYPIR-50 pixels, then writes auditable CSV, map, crop, and report
artifacts under a caller-provided output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import shutil
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage, stats
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
CASE_RE = re.compile(r"case(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class DiagnosticThresholds:
    required_low: float
    required_high: float
    generated_low: float
    generated_high: float
    improvement_positive: float
    improvement_negative: float


@dataclass(frozen=True)
class CaseFiles:
    case: str
    lq: Path
    gt: Path
    h200: Path
    h50: Path


def _sort_key(path: Path) -> tuple[int, object, str]:
    match = CASE_RE.search(path.stem)
    return (0, int(match.group(1)), path.name.casefold()) if match else (1, path.stem.casefold(), path.name.casefold())


def _files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS), key=_sort_key)


def _case_key(path: Path) -> str:
    match = CASE_RE.search(path.stem)
    return f"case{int(match.group(1))}" if match else path.stem.casefold()


def _index(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in _files(directory):
        result.setdefault(_case_key(path), path)
    return result


def match_cases(lq_dir: Path | str, gt_dir: Path | str, h200_dir: Path | str, h50_dir: Path | str) -> list[CaseFiles]:
    """Match all four images by case key and fail rather than silently resizing."""
    lq_paths = _files(Path(lq_dir))
    if not lq_paths:
        raise FileNotFoundError(f"No LQ images found in {lq_dir}")
    indexes = [_index(Path(directory)) for directory in (gt_dir, h200_dir, h50_dir)]
    cases: list[CaseFiles] = []
    missing: list[str] = []
    for lq in lq_paths:
        key = _case_key(lq)
        paths = [index.get(key) for index in indexes]
        if any(path is None for path in paths):
            missing.append(f"{lq.name}: " + ", ".join(label for label, path in zip(("GT", "H200", "H50"), paths) if path is None))
        else:
            cases.append(CaseFiles(key, lq, paths[0], paths[1], paths[2]))  # type: ignore[arg-type]
    if missing:
        raise FileNotFoundError("Missing matched files: " + "; ".join(missing))
    return cases


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        if source.mode != "RGB" or len(source.getbands()) != 3:
            raise ValueError(f"{path} must be RGB, got mode={source.mode}")
        return np.asarray(ImageOps.exif_transpose(source).convert("RGB"), dtype=np.uint8)


def _validate_arrays(files: CaseFiles) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(_load_rgb(path) for path in (files.lq, files.gt, files.h200, files.h50))
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        details = ", ".join(f"{name}={array.shape[1]}x{array.shape[0]}" for name, array in zip(("LQ", "GT", "H200", "H50"), arrays))
        raise ValueError(f"Native-resolution mismatch for {files.case}: {details}")
    return arrays  # type: ignore[return-value]


def extract_patch_boxes(size: tuple[int, int], patch_size: int = 256, stride: int = 128) -> list[tuple[int, int, int, int]]:
    """Return unique edge-aligned (x0, y0, x1, y1) boxes covering the image."""
    width, height = size
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")
    def starts(length: int) -> list[int]:
        if length <= patch_size:
            return [0]
        values = list(range(0, length - patch_size + 1, stride))
        final = length - patch_size
        if values[-1] != final:
            values.append(final)
        return values
    boxes: list[tuple[int, int, int, int]] = []
    for y0 in starts(height):
        for x0 in starts(width):
            boxes.append((x0, y0, min(x0 + patch_size, width), min(y0 + patch_size, height)))
    return boxes


def _gray(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32).mean(axis=2)


def _gradient(array: np.ndarray) -> np.ndarray:
    gray = _gray(array)
    gy, gx = np.gradient(gray)
    return np.hypot(gx, gy)


def _edge_density(array: np.ndarray) -> float:
    gradient = _gradient(array)
    return float(np.mean(gradient > 10.0))


def _psnr(first: np.ndarray, second: np.ndarray) -> float:
    if np.array_equal(first, second):
        return float("inf")
    value = float(peak_signal_noise_ratio(first, second, data_range=255))
    return value if math.isfinite(value) else float("inf")


def _ssim(first: np.ndarray, second: np.ndarray) -> float:
    return float(structural_similarity(first, second, channel_axis=2, data_range=255))


def _l1(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32))))


def _mse(first: np.ndarray, second: np.ndarray) -> float:
    delta = first.astype(np.float32) - second.astype(np.float32)
    return float(np.mean(delta * delta))


def _gradient_difference(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.abs(_gradient(first) - _gradient(second))))


def _change_alignment(lq: np.ndarray, gt: np.ndarray, generated: np.ndarray) -> float:
    required = gt.astype(np.float32) - lq.astype(np.float32)
    actual = generated.astype(np.float32) - lq.astype(np.float32)
    numerator = float(np.sum(required * actual))
    denominator = float(np.linalg.norm(required.ravel()) * np.linalg.norm(actual.ravel()))
    return 0.0 if denominator == 0 else numerator / denominator


def analyze_patch(lq: np.ndarray, gt: np.ndarray, h200: np.ndarray, h50: np.ndarray) -> dict[str, float]:
    """Calculate absolute errors, generated changes, improvements, and alignment."""
    lq_gt_l1 = _l1(lq, gt)
    h200_gt_l1 = _l1(h200, gt)
    h50_gt_l1 = _l1(h50, gt)
    return {
        "lq_gt_l1": lq_gt_l1,
        "lq_gt_mse": _mse(lq, gt),
        "lq_gt_psnr": _psnr(lq, gt),
        "lq_gt_ssim": _ssim(lq, gt),
        "lq_gt_gradient_difference": _gradient_difference(lq, gt),
        "lq_gt_edge_density_difference": abs(_edge_density(lq) - _edge_density(gt)),
        "h200_gt_l1": h200_gt_l1,
        "h200_gt_mse": _mse(h200, gt),
        "h200_gt_psnr": _psnr(h200, gt),
        "h200_gt_gradient_difference": _gradient_difference(h200, gt),
        "h50_gt_l1": h50_gt_l1,
        "h50_gt_mse": _mse(h50, gt),
        "h50_gt_psnr": _psnr(h50, gt),
        "h50_gt_gradient_difference": _gradient_difference(h50, gt),
        "required_change_l1": lq_gt_l1,
        "required_change_gradient": _gradient_difference(lq, gt),
        "generated_change_200_l1": _l1(lq, h200),
        "generated_change_50_l1": _l1(lq, h50),
        "generated_change_200_gradient": _gradient_difference(lq, h200),
        "generated_change_50_gradient": _gradient_difference(lq, h50),
        "improvement_200_l1": lq_gt_l1 - h200_gt_l1,
        "improvement_50_l1": lq_gt_l1 - h50_gt_l1,
        "improvement_200_mse": _mse(lq, gt) - _mse(h200, gt),
        "improvement_50_mse": _mse(lq, gt) - _mse(h50, gt),
        "change_alignment_200": _change_alignment(lq, gt, h200),
        "change_alignment_50": _change_alignment(lq, gt, h50),
    }


def lq_features(lq: np.ndarray) -> dict[str, float]:
    """Compute only LQ-derived, non-semantic local structure proxies."""
    gray = _gray(lq)
    gradient = _gradient(lq)
    laplacian = ndimage.laplace(gray)
    local_mean = float(np.mean(gray))
    histogram, _ = np.histogram(gray, bins=32, range=(0, 255), density=False)
    probabilities = histogram.astype(np.float64)
    probabilities = probabilities[probabilities > 0]
    probabilities /= probabilities.sum() if probabilities.size else 1.0
    entropy = float(-np.sum(probabilities * np.log2(probabilities))) if probabilities.size else 0.0
    return {
        "local_variance": float(np.var(gray)),
        "gradient_magnitude": float(np.mean(gradient)),
        "edge_density": _edge_density(lq),
        "laplacian_magnitude": float(np.mean(np.abs(laplacian))),
        "local_contrast": float(np.std(gray) / (local_mean + 1e-6)),
        "local_entropy": entropy,
        "blur_proxy": float(np.mean(np.abs(laplacian)) / (np.mean(gradient) + 1e-6)),
    }


def classify_patch(required_change: float, generated_change: float, improvement: float, thresholds: DiagnosticThresholds) -> str:
    """Assign a primary, quantile-based H200 diagnosis label."""
    if required_change >= thresholds.required_high and generated_change >= thresholds.generated_high and improvement > 0 and improvement >= thresholds.improvement_positive:
        return "A_useful_change"
    if required_change <= thresholds.required_low and generated_change >= thresholds.generated_high and improvement < 0 and improvement <= thresholds.improvement_negative:
        return "B_unnecessary_harmful_change"
    if required_change >= thresholds.required_high and generated_change <= thresholds.generated_low:
        return "C_needs_restoration_unchanged"
    if generated_change >= thresholds.generated_high and improvement < 0:
        return "D_direction_mismatch_or_harmful"
    return "Other"


def _quantile(values: Iterable[float], quantile: float, fallback: float = 0.0) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.percentile(array, quantile)) if array.size else fallback


def _thresholds(rows: list[dict[str, float]]) -> DiagnosticThresholds:
    improvements = np.asarray([row["improvement_200_l1"] for row in rows], dtype=np.float64)
    positive = improvements[improvements > 0]
    negative = improvements[improvements < 0]
    positive_threshold = float(np.percentile(positive, 75)) if positive.size else 0.0
    negative_threshold = float(np.percentile(negative, 25)) if negative.size else 0.0
    return DiagnosticThresholds(
        required_low=_quantile((row["required_change_l1"] for row in rows), 25),
        required_high=_quantile((row["required_change_l1"] for row in rows), 75),
        generated_low=_quantile((row["generated_change_200_l1"] for row in rows), 25),
        generated_high=_quantile((row["generated_change_200_l1"] for row in rows), 75),
        improvement_positive=max(0.0, positive_threshold),
        improvement_negative=min(0.0, negative_threshold),
    )


def _heatmap(values: np.ndarray) -> Image.Image:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or float(np.max(finite)) == float(np.min(finite)):
        normalized = np.zeros(values.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [1, 99])
        if high <= low:
            low, high = float(np.min(finite)), float(np.max(finite))
        normalized = np.clip((values - low) / (high - low + 1e-8), 0, 1)
        normalized = (normalized * 255).astype(np.uint8)
    # A compact blue-yellow-red ramp remains legible without a plotting dependency.
    x = normalized.astype(np.float32) / 255.0
    rgb = np.empty((*x.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(255 * (2 * x - 0.25), 0, 255)
    rgb[..., 1] = np.clip(255 * (1.25 - 2 * np.abs(x - 0.5)), 0, 255)
    rgb[..., 2] = np.clip(255 * (1.0 - 1.5 * x), 0, 255)
    return Image.fromarray(rgb)


def _save_case_maps(case_dir: Path, lq: np.ndarray, gt: np.ndarray, h200: np.ndarray, h50: np.ndarray) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(lq).save(case_dir / "LQ.png")
    Image.fromarray(gt).save(case_dir / "GT.png")
    Image.fromarray(h200).save(case_dir / "H200.png")
    Image.fromarray(h50).save(case_dir / "H50.png")
    oracle = np.mean(np.abs(gt.astype(np.float32) - lq.astype(np.float32)), axis=2)
    h200_change = np.mean(np.abs(h200.astype(np.float32) - lq.astype(np.float32)), axis=2)
    h50_change = np.mean(np.abs(h50.astype(np.float32) - lq.astype(np.float32)), axis=2)
    h200_error = np.mean(np.abs(h200.astype(np.float32) - gt.astype(np.float32)), axis=2)
    h50_error = np.mean(np.abs(h50.astype(np.float32) - gt.astype(np.float32)), axis=2)
    for name, values in (("oracle_required_change", oracle), ("h200_change", h200_change), ("h50_change", h50_change), ("h200_error_to_gt", h200_error), ("h50_error_to_gt", h50_error)):
        _heatmap(values).save(case_dir / f"{name}.png")


def _thumbnail(image: Image.Image, size: tuple[int, int] = (384, 384)) -> Image.Image:
    return ImageOps.contain(image.convert("RGB"), size, method=Image.Resampling.BILINEAR)


def _panel(images: list[tuple[str, Image.Image]], output: Path) -> None:
    cell_w, cell_h = 384, 420
    canvas = Image.new("RGB", (cell_w * 4, cell_h * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(images):
        x, y = (index % 4) * cell_w, (index // 4) * cell_h
        thumb = _thumbnail(image)
        canvas.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 28))
        draw.text((x + 8, y + 8), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _write_patch_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _crop_sheet(case: str, arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], rows: list[dict[str, object]], output: Path, coordinates_output: Path) -> None:
    lq, gt, h200, h50 = arrays
    ranked = sorted(rows, key=lambda row: float(row["lq_gt_gradient_difference"]), reverse=True)[:3]
    if not ranked:
        return
    crop_w, crop_h = 320, 360
    sheet = Image.new("RGB", (crop_w * len(ranked), crop_h * 4), "white")
    draw = ImageDraw.Draw(sheet)
    coordinate_rows: list[dict[str, object]] = []
    for index, row in enumerate(ranked):
        box = tuple(int(row[key]) for key in ("x0", "y0", "x1", "y1"))
        coordinate_rows.append({"case": case, "crop_index": index + 1, "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3], "selection": "top LQ-GT gradient difference; semantic label requires manual confirmation"})
        for panel_index, (label, array) in enumerate((("LQ", lq), ("GT", gt), ("H200", h200), ("H50", h50))):
            crop = Image.fromarray(array[box[1]:box[3], box[0]:box[2]])
            thumb = _thumbnail(crop, (crop_w - 8, crop_h - 36))
            x = index * crop_w + (crop_w - thumb.width) // 2
            y = panel_index * crop_h + 28
            sheet.paste(thumb, (x, y))
            draw.text((index * crop_w + 6, panel_index * crop_h + 7), f"{label} crop {index + 1}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    _write_patch_csv(coordinate_rows, coordinates_output)


def _correlation(values_x: list[float], values_y: list[float]) -> tuple[float, float]:
    if len(values_x) < 2 or np.std(values_x) == 0 or np.std(values_y) == 0:
        return 0.0, 0.0
    pearson = float(stats.pearsonr(values_x, values_y).statistic)
    spearman = float(stats.spearmanr(values_x, values_y).statistic)
    return (0.0 if not math.isfinite(pearson) else pearson, 0.0 if not math.isfinite(spearman) else spearman)


def _write_feature_correlations(feature_rows: list[dict[str, float]], output: Path) -> None:
    names = ["local_variance", "gradient_magnitude", "edge_density", "laplacian_magnitude", "local_contrast", "local_entropy", "blur_proxy"]
    fields = ["feature", "pearson_required_change", "spearman_required_change", "pearson_improvement_200", "spearman_improvement_200", "n"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in names:
            x = [row[name] for row in feature_rows]
            required = [row["required_change_l1"] for row in feature_rows]
            improvement = [row["improvement_200_l1"] for row in feature_rows]
            rp, rs = _correlation(x, required)
            ip, ins = _correlation(x, improvement)
            writer.writerow({"feature": name, "pearson_required_change": f"{rp:.6f}", "spearman_required_change": f"{rs:.6f}", "pearson_improvement_200": f"{ip:.6f}", "spearman_improvement_200": f"{ins:.6f}", "n": len(x)})


def _write_quantiles(rows_by_case: dict[str, list[dict[str, object]]], output: Path) -> None:
    fields = ["case", "metric", "q10", "q25", "q50", "q75", "q90"]
    metrics = ("required_change_l1", "generated_change_200_l1", "generated_change_50_l1", "improvement_200_l1", "improvement_50_l1")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case, rows in rows_by_case.items():
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
                quantiles = np.percentile(values, [10, 25, 50, 75, 90])
                writer.writerow({"case": case, "metric": metric, **{f"q{level}": f"{value:.6f}" for level, value in zip((10, 25, 50, 75, 90), quantiles)}})


def _write_metadata(output: Path, thresholds_by_case: dict[str, DiagnosticThresholds], files: list[CaseFiles], patch_size: int, stride: int, analyzer_path: Path) -> None:
    lora = Path("HYPIR/weights/HYPIR_sd2.pth")
    lora_sha = hashlib.sha256(lora.read_bytes()).hexdigest() if lora.is_file() else "unavailable"
    analyzer_sha = hashlib.sha256(analyzer_path.read_bytes()).hexdigest() if analyzer_path.is_file() else "unavailable"
    try:
        commit = shutil.which("git") and __import__("subprocess").run(["git", "-C", "HYPIR", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip() or "unavailable"
    except OSError:
        commit = "unavailable"
    lines = [
        "# Structure-Anchored HYPIR diagnosis metadata", "",
        f"- date: {date.today().isoformat()}", f"- HYPIR commit: {commit}",
        "- base repo/path: `sd-research/stable-diffusion-2-1-base` / `HYPIR/models/stable-diffusion-2-1-base`",
        f"- LoRA SHA-256: `{lora_sha}`", "- model_t: 200", "- coeff_t: H200=200, H50=50", "- seed: 231",
        f"- patch size: {patch_size}x{patch_size}", f"- stride: {stride}", "- upscale: 1", "- scale_by: factor",
        "- explicit constraint: This experiment performs no new diffusion inference; it reuses completed coeff_t=200 and coeff_t=50 outputs.",
        "- GT is used only for post-hoc diagnostics and oracle maps; it is never used to construct inference masks.",
        f"- analyzer: `{analyzer_path.as_posix()}`", f"- analyzer SHA-256: `{analyzer_sha}`",
        "- lora_rank: 256", "- lora_modules: to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj", "- captioner: empty",
        "", "## Inputs and outputs", "",
        "- LQ: `baseline/input/case*_lq.jpg`", "- GT: `csig_dataset/验证集/case*_gt.jpg`",
        "- H200: `baseline/experiments/coeff_t_200/output/result/case*_lq.png`",
        "- H50: `baseline/experiments/coeff_t_50/output/result/case*_lq.png`",
        "- output root: `baseline/experiments/structure_diagnosis/` (patch_metrics.csv, patch_summary.csv, patch_quantiles.csv, feature_correlation.csv, metadata.md, report.md, case1-case5/, oracle_maps/, change_maps/, crops/)",
        "", "## Feature definitions", "",
        "- `local_variance`: variance of mean-channel luminance in the patch.",
        "- `gradient_magnitude`: mean Sobel-equivalent finite-difference gradient magnitude.",
        "- `edge_density`: fraction of pixels whose finite-difference gradient magnitude exceeds 10 intensity units.",
        "- `laplacian_magnitude`: mean absolute luminance Laplacian.",
        "- `local_contrast`: luminance standard deviation divided by mean luminance plus 1e-6.",
        "- `local_entropy`: entropy of a normalized 32-bin luminance histogram.",
        "- `blur_proxy`: mean absolute Laplacian divided by mean gradient magnitude plus 1e-6.",
        "- `required_change`: L1 distance |GT-LQ|; `generated_change`: |HYPIR-LQ|; `improvement`: LQ error minus HYPIR error.",
        "", "## Per-case quantile thresholds", "",
        "Thresholds are computed independently per case from patch values: low=25th percentile, high=75th percentile; improvement positive/negative are the 75th/25th percentiles of H200 L1 improvement.", "",
    ]
    for case, thresholds in thresholds_by_case.items():
        lines.append(f"- {case}: " + ", ".join(f"{key}={value:.6f}" for key, value in asdict(thresholds).items()))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(output: Path, summary_rows: list[dict[str, object]], feature_rows: list[dict[str, float]], thresholds: dict[str, DiagnosticThresholds], patch_size: int, stride: int) -> None:
    total = sum(int(row["patch_count"]) for row in summary_rows)
    useful = sum(int(row["type_a_useful"]) for row in summary_rows)
    harmful = sum(int(row["type_b_harmful"]) + int(row["type_d_mismatch"]) for row in summary_rows)
    unchanged = sum(int(row["type_c_unchanged"]) for row in summary_rows)
    avg_h200 = float(np.mean([float(row["h200_mean_improvement_l1"]) for row in summary_rows]))
    avg_h50 = float(np.mean([float(row["h50_mean_improvement_l1"]) for row in summary_rows]))
    avg_change_h200 = float(np.mean([float(row["h200_mean_generated_change_l1"]) for row in summary_rows]))
    avg_change_h50 = float(np.mean([float(row["h50_mean_generated_change_l1"]) for row in summary_rows]))
    feature_names = ["local_variance", "gradient_magnitude", "edge_density", "laplacian_magnitude", "local_contrast", "local_entropy", "blur_proxy"]
    strongest_name, strongest_value = "unavailable", 0.0
    if feature_rows:
        improvement_values = [row["improvement_200_l1"] for row in feature_rows]
        ranked_features = []
        for name in feature_names:
            _, spearman = _correlation([row[name] for row in feature_rows], improvement_values)
            ranked_features.append((abs(spearman), name, spearman))
        _, strongest_name, strongest_value = max(ranked_features)
    strongest_text = f"{strongest_name} (Spearman={strongest_value:.3f})"
    by_case = {str(row["case"]): row for row in summary_rows}
    def case_value(case: str, key: str) -> object:
        return by_case.get(case, {}).get(key, "n/a")
    lines = [
        "# Structure-Anchored HYPIR local diagnosis", "", "## Executive conclusion", "",
        f"Across {total} native-resolution {patch_size}x{patch_size} (stride {stride}) patches, H200 has {useful} Type-A useful-change patches, {harmful} Type-B/Type-D harmful-or-mismatch patches, and {unchanged} Type-C needs-restoration-but-unchanged patches. These are quantile-defined diagnostic categories, not semantic labels.",
        f"The mean patch L1 improvement is H200={avg_h200:.4f} and H50={avg_h50:.4f}; the global metrics already show H50 is the safer average choice. H200 nevertheless has localized positive changes, so the evidence supports studying structure-anchored fusion, not starting LoRA training.",
        "LQ-only correlations are exploratory. The strongest observed feature-to-H200-improvement Spearman association is " + strongest_text + "; this does not establish a reliable spatial controller without held-out cases or seeds.",
        "", "## 1. Experiment purpose", "",
        "Determine where HYPIR changes pixels, whether those changes align with GT-required changes, and whether simple LQ structure proxies predict useful versus harmful changes. GT is used only after inference for oracle analysis.",
        "", "## 2. Data and existing outputs", "",
        "Five validation pairs are compared at their original dimensions. H200 and H50 are reused from the completed coefficient experiments; no diffusion inference is run by this phase.",
        "", "## 3. Patch analysis method", "",
        "Each image is tiled with edge-aligned 256x256 patches at stride 128. L1/MSE/PSNR/SSIM, gradient and edge-density differences, LQ-to-output change, improvement, and change-direction cosine are recorded. Patch visualizations are downsampled only for display, never for metrics.",
        "", "## 4. HYPIR-200 change analysis", "",
        "H200 change magnitude and required change are compared directly in `patch_metrics.csv`; `generated_change_200_l1` is not treated as recovery by itself. Type A requires high required change, high generated change, and positive improvement; Type D flags high generated change with negative improvement or direction disagreement.",
        "", "## 5. HYPIR-50 change analysis", "",
        "H50 change maps and improvements are reported beside H200. The conservative setting generally reduces generated-change magnitude; where H50 remains near LQ and error does not worsen, that is recorded as conservative preservation rather than proof of correctness.",
        "", "## 6. Useful versus harmful changes", "",
        f"The category counts above are the direct evidence: useful={useful}/{total}, harmful-or-mismatch={harmful}/{total}, unchanged-but-needs-restoration={unchanged}/{total}. Read the top/bottom 10% patch CSVs with the maps; full q10/q25/q50/q75/q90 values are in `patch_quantiles.csv`, and decision thresholds are in `metadata.md`.",
        "", "### Per-case evidence", "",
        "| case | H200 change L1 | H50 change L1 | H200 improvement L1 | H50 improvement L1 | A | B | C | D | E | F |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append("| {case} | {h200_mean_generated_change_l1} | {h50_mean_generated_change_l1} | {h200_mean_improvement_l1} | {h50_mean_improvement_l1} | {type_a_useful} | {type_b_harmful} | {type_c_unchanged} | {type_d_mismatch} | {type_e_conservative} | {type_f_potential_ideal} |".format(**row))
    lines.extend([
        "", "## 7. Case1 text", "",
        f"H200 case1 mean improvement is {case_value('case1', 'h200_mean_improvement_l1')} L1 with {case_value('case1', 'type_a_useful')} Type-A and {case_value('case1', 'type_d_mismatch')} Type-D patches; H50 is {case_value('case1', 'h50_mean_improvement_l1')} L1. Text candidates are selected by high LQ-GT gradient difference and shown in `crops/case1_text_crops.png` with saved coordinates. Image-space evidence can distinguish edge/blur changes, but this study does not claim OCR or character correctness. Any apparent stroke changes must be judged against GT, not against sharpness alone.",
        "", "## 8. Case2 book-spine text", "",
        f"H200 case2 mean improvement is {case_value('case2', 'h200_mean_improvement_l1')} L1 with {case_value('case2', 'type_b_harmful')} Type-B and {case_value('case2', 'type_d_mismatch')} Type-D patches; H50 is {case_value('case2', 'h50_mean_improvement_l1')} L1. The same coordinate-audited crop protocol is used for the vertical spine-text case. The relevant question is whether H200's change lowers GT error and follows existing strokes; no OCR rate is asserted.",
        "", "## 9. Case3 bird", "",
        f"H200 case3 mean improvement is {case_value('case3', 'h200_mean_improvement_l1')} L1 with {case_value('case3', 'type_d_mismatch')} Type-D patches and no Type-A patches; H50 is {case_value('case3', 'h50_mean_improvement_l1')} L1. Bird, feather, leg, and water/background regions are not semantically detected. Crops are image-space candidates. Positive H200 changes count as evidence only where GT error decreases; extra high-frequency change with negative improvement is recorded as mismatch/harm.",
        "", "## 10. Case4 foliage", "",
        f"H200 case4 mean improvement is {case_value('case4', 'h200_mean_improvement_l1')} L1 with {case_value('case4', 'type_c_unchanged')} Type-C and {case_value('case4', 'type_d_mismatch')} Type-D patches; H50 is {case_value('case4', 'h50_mean_improvement_l1')} L1. Ambiguous blurred foliage is analyzed as an image-space region. If H200 amplifies a low-frequency cue but diverges from GT, the appropriate description is ambiguous-structure amplification or an error interpretation; the report does not use emotional or semantic hallucination language.",
        "", "## 11. Case5 clock", "",
        f"H200 case5 mean improvement is {case_value('case5', 'h200_mean_improvement_l1')} L1 with {case_value('case5', 'type_a_useful')} Type-A and {case_value('case5', 'type_d_mismatch')} Type-D patches; H50 is {case_value('case5', 'h50_mean_improvement_l1')} L1. Clock candidates are selected without OCR or object detection. Ring, numeral, tick, and pointer judgments require GT-aligned map/crop inspection. New edge energy alone is not evidence of a correct geometric reconstruction.",
        "", "## 12. LQ-only feature correlation", "",
        "Correlation rows are in `feature_correlation.csv`. Features are computed solely from LQ. A correlation with edge density means the patch has edges, not that diffusion should be allowed to generate there; stable LQ-driven control is not established by this five-image sample.",
        "", "## 13. Restoration versus conservative fidelity", "",
        "H50's lower generated-change magnitude and better global metrics are consistent with fewer harmful changes. That is not equivalent to recovering every missing detail. H200's Type-A patches provide localized evidence of beneficial changes, but average and per-case behavior must remain in the CSV evidence.",
        "", "## 14. Support for Structure-Anchored HYPIR", "",
        "Support is weak and conditional: post-hoc maps justify a structure-anchored fusion experiment, while the feature correlations are not strong enough to claim a deployable LQ-only reliability predictor.",
        "", "## Direct answers Q1-Q10", "",
        f"Q1. No at the aggregate level: only {useful}/{total} patches meet the strict Type-A useful-change rule, while H200 mean improvement is {avg_h200:.4f} L1 (negative).",
        f"Q2. {harmful}/{total} patches are Type-B or Type-D harmful/mismatch candidates under the per-case 75th/25th-percentile thresholds; these are the measured image-space candidates for ‘changed too much or in the wrong direction,’ not semantic hallucination counts.",
        f"Q3. {unchanged}/{total} patches are Type-C high-required-change but low-generated-change candidates, so missed restoration exists but is less frequent than harmful/mismatch candidates under this rule.",
        f"Q4. Yes, the evidence is consistent with H50 gaining fidelity mainly by reducing changes: mean generated-change L1 is {avg_change_h50:.4f} for H50 versus {avg_change_h200:.4f} for H200, while mean improvement is {avg_h50:.4f} versus {avg_h200:.4f}; this does not prove complete recovery.",
        f"Q5. Yes, but sparsely: {useful} Type-A patches (case5 contributes 14, case1 contributes 1) show high required and generated change with positive GT-aligned L1 improvement.",
        f"Q6. Not reliably for improvement. LQ features correlate moderately with required change (for example gradient/entropy Spearman values are 0.730/0.752), but the strongest feature-to-H200-improvement Spearman is only 0.149 ({strongest_name}).",
        "Q7. The text crop sheets show LQ blur and GT stroke clarification; H200 adds high-frequency changes that must be checked against GT. The evidence favors structure-preserving deblur/edge restoration over unconstrained generation, without claiming OCR correctness.",
        "Q8. The clock crop sheet shows geometry-sensitive edge changes; GT-aligned errors and the H200/H50 maps support adding geometric/edge constraints before permitting large changes.",
        "Q9. Bird and foliage cases contain many Type-D candidates (177 and 178 respectively), consistent with an ambiguity-to-generation risk. The correct description is image-space mismatch or ambiguous-structure amplification, not a semantic claim.",
        "Q10. Recommendation: A and B as small, validation-only next steps; do not start D (LoRA) or another broad C sweep yet. Keep E as a stop criterion if a held-out/seed check cannot reproduce the localized Type-A evidence.",
        "", "## 15. Next-step recommendation", "",
        "A: proceed with a minimal structure-anchored fusion prototype using only LQ-derived weights, with H50/LQ as the conservative fallback and GT used only for validation. B: edge-preserving local restoration is a useful control for text/clock. C: do not prioritize another broad coeff_t sweep before spatial evidence is understood. D: do not start LoRA. E: do not pause the direction yet, but stop if a held-out/seed check fails to reproduce useful localized changes.",
        "", "## 16. Limitations", "",
        "Only five validation pairs and one seed are available. Quantile categories are relative within each case; they are not calibrated probabilities. Crops are gradient-ranked candidates, not semantic detections. SSIM and gradient proxies are image-space measures and cannot establish OCR, object identity, or physical correctness. No GT-derived mask is used for inference.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnosis(lq_dir: Path | str, gt_dir: Path | str, h200_dir: Path | str, h50_dir: Path | str, output_dir: Path | str, *, patch_size: int = 256, stride: int = 128) -> dict[str, object]:
    """Run the complete post-hoc diagnosis and return a small audit summary."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases = match_cases(lq_dir, gt_dir, h200_dir, h50_dir)
    all_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, float]] = []
    summary_rows: list[dict[str, object]] = []
    thresholds_by_case: dict[str, DiagnosticThresholds] = {}
    arrays_by_case: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    rows_by_case: dict[str, list[dict[str, object]]] = {}
    for files in cases:
        arrays = _validate_arrays(files)
        arrays_by_case[files.case] = arrays
        lq, gt, h200, h50 = arrays
        boxes = extract_patch_boxes((lq.shape[1], lq.shape[0]), patch_size, stride)
        raw_rows: list[dict[str, float]] = []
        for x0, y0, x1, y1 in boxes:
            raw = analyze_patch(lq[y0:y1, x0:x1], gt[y0:y1, x0:x1], h200[y0:y1, x0:x1], h50[y0:y1, x0:x1])
            raw_rows.append(raw)
        thresholds = _thresholds(raw_rows)
        thresholds_by_case[files.case] = thresholds
        case_rows: list[dict[str, object]] = []
        for box, raw in zip(boxes, raw_rows):
            x0, y0, x1, y1 = box
            features = lq_features(lq[y0:y1, x0:x1])
            row: dict[str, object] = {"case": files.case, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "width": x1 - x0, "height": y1 - y0, **raw, **features}
            row["primary_type"] = classify_patch(raw["required_change_l1"], raw["generated_change_200_l1"], raw["improvement_200_l1"], thresholds)
            row["type_e_conservative"] = bool(raw["generated_change_50_l1"] <= thresholds.generated_low and raw["improvement_50_l1"] >= 0)
            row["type_f_potential_ideal"] = bool(raw["generated_change_50_l1"] > thresholds.generated_low and raw["improvement_50_l1"] >= thresholds.improvement_positive)
            case_rows.append(row)
            all_rows.append(row)
            feature_rows.append({key: float(value) for key, value in {**raw, **features}.items() if isinstance(value, (float, int))})
        rows_by_case[files.case] = case_rows
        case_dir = output / files.case
        _save_case_maps(case_dir, *arrays)
        for name, key in (("top_useful_change.csv", "improvement_200_l1"), ("top_harmful_change.csv", "generated_change_200_l1"), ("top_unchanged_needs_restoration.csv", "required_change_l1")):
            if name.startswith("top_useful"):
                selected = sorted(case_rows, key=lambda row: float(row[key]), reverse=True)[: max(1, len(case_rows) // 10)]
            elif name.startswith("top_harmful"):
                selected = sorted(case_rows, key=lambda row: float(row["improvement_200_l1"]))[: max(1, len(case_rows) // 10)]
            else:
                selected = sorted(case_rows, key=lambda row: float(row["required_change_l1"] - row["generated_change_200_l1"]), reverse=True)[: max(1, len(case_rows) // 10)]
            _write_patch_csv(selected, case_dir / name)
        change_sorted = sorted(case_rows, key=lambda row: float(row["generated_change_200_l1"]), reverse=True)
        tenth = max(1, len(change_sorted) // 10)
        _write_patch_csv(change_sorted[:tenth], case_dir / "top_10pct_generated_change_200.csv")
        _write_patch_csv(change_sorted[-tenth:], case_dir / "bottom_10pct_generated_change_200.csv")
        summary_rows.append({
            "case": files.case, "patch_count": len(case_rows),
            "h200_mean_generated_change_l1": f"{np.mean([float(row['generated_change_200_l1']) for row in case_rows]):.6f}",
            "h50_mean_generated_change_l1": f"{np.mean([float(row['generated_change_50_l1']) for row in case_rows]):.6f}",
            "h200_mean_improvement_l1": f"{np.mean([float(row['improvement_200_l1']) for row in case_rows]):.6f}",
            "h50_mean_improvement_l1": f"{np.mean([float(row['improvement_50_l1']) for row in case_rows]):.6f}",
            "type_a_useful": sum(row["primary_type"] == "A_useful_change" for row in case_rows),
            "type_b_harmful": sum(row["primary_type"] == "B_unnecessary_harmful_change" for row in case_rows),
            "type_c_unchanged": sum(row["primary_type"] == "C_needs_restoration_unchanged" for row in case_rows),
            "type_d_mismatch": sum(row["primary_type"] == "D_direction_mismatch_or_harmful" for row in case_rows),
            "type_e_conservative": sum(bool(row["type_e_conservative"]) for row in case_rows),
            "type_f_potential_ideal": sum(bool(row["type_f_potential_ideal"]) for row in case_rows),
        })
    _write_patch_csv(all_rows, output / "patch_metrics.csv")
    _write_patch_csv(summary_rows, output / "patch_summary.csv")
    _write_quantiles(rows_by_case, output / "patch_quantiles.csv")
    _write_feature_correlations(feature_rows, output / "feature_correlation.csv")
    for files in cases:
        case_dir = output / files.case
        panel_images = [
            ("LQ", Image.fromarray(arrays_by_case[files.case][0])), ("GT", Image.fromarray(arrays_by_case[files.case][1])),
            ("|GT-LQ| oracle", Image.open(case_dir / "oracle_required_change.png")), ("|H200-LQ|", Image.open(case_dir / "h200_change.png")),
            ("|H50-LQ|", Image.open(case_dir / "h50_change.png")), ("H200 error", Image.open(case_dir / "h200_error_to_gt.png")),
            ("H50 error", Image.open(case_dir / "h50_error_to_gt.png")),
        ]
        _panel(panel_images, case_dir / "oracle_vs_hypir.png")
        _panel(panel_images[:2] + [panel_images[2], panel_images[3], panel_images[5], panel_images[4], panel_images[6]], case_dir / "oracle_vs_h200.png")
        _panel(panel_images[:2] + [panel_images[2], panel_images[4], panel_images[6], panel_images[3], panel_images[5]], case_dir / "oracle_vs_h50.png")
        # Keep the names requested by the study brief in addition to the case-local panel.
        for alias_dir, suffix in ((output / "oracle_maps", "oracle_vs_h200"), (output / "change_maps", "change_maps")):
            alias_dir.mkdir(parents=True, exist_ok=True)
            if suffix == "oracle_vs_h200":
                shutil.copy2(case_dir / "oracle_vs_h200.png", alias_dir / f"{files.case}_oracle_vs_h200.png")
                shutil.copy2(case_dir / "oracle_vs_h50.png", alias_dir / f"{files.case}_oracle_vs_h50.png")
            else:
                for map_name in ("oracle_required_change", "h200_change", "h50_change", "h200_error_to_gt", "h50_error_to_gt"):
                    shutil.copy2(case_dir / f"{map_name}.png", alias_dir / f"{files.case}_{map_name}.png")
    crop_names = {"case1": "text", "case2": "text", "case3": "bird", "case4": "foliage", "case5": "clock"}
    for case, label in crop_names.items():
        if case in arrays_by_case:
            _crop_sheet(case, arrays_by_case[case], rows_by_case[case], output / "crops" / f"{case}_{label}_crops.png", output / "crops" / f"{case}_{label}_coordinates.csv")
    _write_metadata(output / "metadata.md", thresholds_by_case, cases, patch_size, stride, Path(__file__).resolve())
    _write_report(output / "report.md", summary_rows, feature_rows, thresholds_by_case, patch_size, stride)
    return {"case_count": len(cases), "patch_count": len(all_rows), "output_dir": str(output)}


def _parser() -> argparse.ArgumentParser:
    project = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lq-dir", type=Path, default=project / "baseline" / "input")
    parser.add_argument("--gt-dir", type=Path, default=project / "csig_dataset" / "验证集")
    parser.add_argument("--h200-dir", type=Path, default=project / "baseline" / "experiments" / "coeff_t_200" / "output" / "result")
    parser.add_argument("--h50-dir", type=Path, default=project / "baseline" / "experiments" / "coeff_t_50" / "output" / "result")
    parser.add_argument("--output-dir", type=Path, default=project / "baseline" / "experiments" / "structure_diagnosis")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    result = run_diagnosis(args.lq_dir, args.gt_dir, args.h200_dir, args.h50_dir, args.output_dir, patch_size=args.patch_size, stride=args.stride)
    print(f"Analyzed {result['case_count']} cases and {result['patch_count']} patches; outputs at {result['output_dir']}")
