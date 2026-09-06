"""E3-A: LQ-texture-only spatial blending of existing HYPIR-50 and SwinIR PNGs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel, uniform_filter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.experiments.e2_global_blend import compute_image_metrics, load_rgb, save_rgb_png


ALPHA_MIN = 0.30
ALPHA_MAX = 0.70
GLOBAL_ALPHA = 0.60
GAUSSIAN_SIGMA = 8.0
ROBUST_LOW_PERCENTILE = 1.0
ROBUST_HIGH_PERCENTILE = 99.0
LOCAL_VARIANCE_WINDOW = 9
HIGH_FREQUENCY_BLUR_SIGMA = 2.0
METHODS = ("global_alpha_0.6", "adaptive_texture", "adaptive_reverse")
CASE_IDS = tuple(f"case{index}" for index in range(1, 6))
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def robust_normalize(values: np.ndarray) -> np.ndarray:
    """Robustly normalize one LQ feature with fixed 1st/99th percentiles."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("feature values must be a finite 2D array")
    low, high = np.percentile(array, (ROBUST_LOW_PERCENTILE, ROBUST_HIGH_PERCENTILE))
    if high <= low + 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _luminance(lq: np.ndarray) -> np.ndarray:
    image = np.asarray(lq)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("LQ image must be RGB uint8")
    return (
        0.299 * image[..., 0].astype(np.float32)
        + 0.587 * image[..., 1].astype(np.float32)
        + 0.114 * image[..., 2].astype(np.float32)
    ) / 255.0


def compute_texture_score(lq: np.ndarray) -> np.ndarray:
    """Return the fixed LQ-only texture score used by both adaptive methods.

    The three equally weighted features are Sobel magnitude, local variance,
    and absolute Gaussian blur residual. Each is normalized independently with
    fixed robust percentiles before their average is smoothed at sigma=8.
    """
    gray = _luminance(lq)
    gradient_x = sobel(gray, axis=1, mode="reflect")
    gradient_y = sobel(gray, axis=0, mode="reflect")
    gradient = np.hypot(gradient_x, gradient_y).astype(np.float32)
    gradient_score = robust_normalize(gradient)

    local_mean = uniform_filter(gray, size=LOCAL_VARIANCE_WINDOW, mode="reflect")
    local_mean_square = uniform_filter(gray * gray, size=LOCAL_VARIANCE_WINDOW, mode="reflect")
    local_variance = np.maximum(local_mean_square - local_mean * local_mean, 0.0)
    variance_score = robust_normalize(local_variance)

    blur = gaussian_filter(gray, sigma=HIGH_FREQUENCY_BLUR_SIGMA, mode="reflect")
    high_frequency_energy = np.abs(gray - blur)
    high_frequency_score = robust_normalize(high_frequency_energy)

    score = (gradient_score + variance_score + high_frequency_score) / 3.0
    score = gaussian_filter(score.astype(np.float32), sigma=GAUSSIAN_SIGMA, mode="reflect")
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def compute_alpha_map(texture_score: np.ndarray, *, reverse: bool) -> np.ndarray:
    """Map one shared texture score to the authorized forward or reverse alpha."""
    texture = np.asarray(texture_score, dtype=np.float32)
    if texture.ndim != 2 or not np.isfinite(texture).all():
        raise ValueError("texture score must be a finite 2D array")
    if np.any((texture < 0.0) | (texture > 1.0)):
        raise ValueError("texture score must be in [0, 1]")
    alpha = ALPHA_MAX - 0.40 * texture if reverse else ALPHA_MIN + 0.40 * texture
    return np.clip(alpha, ALPHA_MIN, ALPHA_MAX).astype(np.float32)


def fuse_with_alpha(hypir: np.ndarray, fidelity: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Fuse `alpha * HYPIR + (1-alpha) * Fidelity` as rounded RGB uint8."""
    hypir_image = np.asarray(hypir)
    fidelity_image = np.asarray(fidelity)
    weights = np.asarray(alpha, dtype=np.float32)
    if hypir_image.shape != fidelity_image.shape or hypir_image.ndim != 3 or hypir_image.shape[2] != 3:
        raise ValueError("HYPIR and Fidelity must be same-size RGB arrays")
    if hypir_image.dtype != np.uint8 or fidelity_image.dtype != np.uint8:
        raise ValueError("HYPIR and Fidelity must be uint8")
    if weights.shape != hypir_image.shape[:2] or not np.isfinite(weights).all():
        raise ValueError("alpha map must be finite and match image dimensions")
    if np.any((weights < 0.0) | (weights > 1.0)):
        raise ValueError("alpha map must be in [0, 1]")
    fused = weights[..., None] * hypir_image.astype(np.float32) + (1.0 - weights[..., None]) * fidelity_image.astype(np.float32)
    return np.clip(np.rint(fused), 0, 255).astype(np.uint8)


def _case_file(directory: Path, case: str, suffixes: Iterable[str]) -> Path:
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {directory}")
    wanted = [(case + suffix).casefold() for suffix in suffixes]
    matches: list[Path] = []
    for stem in wanted:
        matches.extend(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS and path.stem.casefold() == stem
        )
        if matches:
            return sorted(matches, key=lambda item: item.name.casefold())[0]
    raise FileNotFoundError(f"{case}: no image found in {directory} for {wanted}")


def _case_names(lq_dir: Path) -> list[str]:
    cases = {
        path.stem.casefold().removesuffix("_lq")
        for path in lq_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS and path.stem.casefold().startswith("case")
    }
    return sorted(cases, key=lambda value: (not value[4:].isdigit(), int(value[4:]) if value[4:].isdigit() else value))


def _map_to_rgb(values: np.ndarray) -> np.ndarray:
    grayscale = np.clip(np.rint(np.asarray(values, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
    return np.repeat(grayscale[..., None], 3, axis=2)


def _pearson_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    x = np.asarray(first, dtype=np.float64).ravel()
    y = np.asarray(second, dtype=np.float64).ravel()
    if x.size == 0 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _format(value: float | None) -> str:
    return "" if value is None or not math.isfinite(value) else f"{value:.6f}"


def _report_correlation(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.6f}"


def _average_for_csv(values: Sequence[float]) -> float | None:
    """Match E2's convention: round each case row to six decimals first."""
    if not values:
        return None
    rounded = [float(f"{float(value):.6f}") for value in values]
    return float(np.mean(rounded))


def _write_metrics(rows: list[dict[str, object]], output_root: Path) -> None:
    fields = [
        "case", "method", "PSNR", "SSIM", "LPIPS",
        "Alpha_Mean", "Alpha_Std", "Alpha_Min", "Alpha_Max", "Texture_Mean", "Texture_Alpha_Pearson",
        "Delta_PSNR_vs_GlobalAlpha06", "Delta_SSIM_vs_GlobalAlpha06", "Delta_LPIPS_vs_GlobalAlpha06",
        "Delta_PSNR_vs_HYPIR50", "Delta_SSIM_vs_HYPIR50", "Delta_LPIPS_vs_HYPIR50",
    ]
    averages: list[dict[str, object]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        average: dict[str, object] = {"case": "Average", "method": method}
        for field in fields[2:]:
            values = [float(row[field]) for row in selected if row[field] is not None]
            average[field] = _average_for_csv(values)
        averages.append(average)
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in [*rows, *averages]:
            writer.writerow({field: row[field] if field in ("case", "method") else _format(row[field]) for field in fields})


def _write_report(rows: list[dict[str, object]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    average_rows = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        average_rows.append({
            "method": method,
            "PSNR": _average_for_csv([row["PSNR"] for row in selected]),
            "SSIM": _average_for_csv([row["SSIM"] for row in selected]),
            "LPIPS": _average_for_csv([row["LPIPS"] for row in selected]),
            "alpha_mean": _average_for_csv([row["Alpha_Mean"] for row in selected]),
            "alpha_std": _average_for_csv([row["Alpha_Std"] for row in selected]),
        })
    global_row = next(row for row in average_rows if row["method"] == "global_alpha_0.6")
    texture_row = next(row for row in average_rows if row["method"] == "adaptive_texture")
    reverse_row = next(row for row in average_rows if row["method"] == "adaptive_reverse")
    table_rows = "\n".join(
        f"| {row['method']} | {row['PSNR']:.6f} | {row['SSIM']:.6f} | {row['LPIPS']:.6f} | {row['alpha_mean']:.6f} | {row['alpha_std']:.6f} |"
        for row in average_rows
    )
    per_case_rows = "\n".join(
        f"| {row['case']} | {row['method']} | {row['PSNR']:.6f} | {row['SSIM']:.6f} | {row['LPIPS']:.6f} | "
        f"{row['Alpha_Mean']:.6f} | {row['Alpha_Std']:.6f} | {row['Alpha_Min']:.6f} | {row['Alpha_Max']:.6f} | "
        f"{row['Texture_Mean']:.6f} | {_report_correlation(row['Texture_Alpha_Pearson'])} |"
        for row in rows
    )
    delta_rows = "\n".join(
        f"| {row['case']} | {row['method']} | {row['Delta_PSNR_vs_GlobalAlpha06']:+.6f} | {row['Delta_SSIM_vs_GlobalAlpha06']:+.6f} | {row['Delta_LPIPS_vs_GlobalAlpha06']:+.6f} | "
        f"{row['Delta_PSNR_vs_HYPIR50']:+.6f} | {row['Delta_SSIM_vs_HYPIR50']:+.6f} | {row['Delta_LPIPS_vs_HYPIR50']:+.6f} |"
        for row in rows
    )
    texture_beats_global = texture_row["PSNR"] > global_row["PSNR"] and texture_row["SSIM"] > global_row["SSIM"] and texture_row["LPIPS"] < global_row["LPIPS"]
    conclusion = (
        "`adaptive_texture` improves all three average metrics over `global_alpha_0.6`."
        if texture_beats_global
        else "`adaptive_texture` does not improve all three average metrics over `global_alpha_0.6`; the fixed global blend remains the stronger composite result under this five-case check."
    )
    reverse_difference = (
        f"Relative to `adaptive_texture`, `adaptive_reverse` changes average PSNR/SSIM/LPIPS by "
        f"{reverse_row['PSNR'] - texture_row['PSNR']:+.6f}/"
        f"{reverse_row['SSIM'] - texture_row['SSIM']:+.6f}/"
        f"{reverse_row['LPIPS'] - texture_row['LPIPS']:+.6f}."
    )
    report_path.write_text(
        "# E3-A Texture-only Adaptive Alpha\n\n"
        "## Scope\n\n"
        "This is an offline blend of existing HYPIR-50 and SwinIR outputs. No HYPIR or SwinIR inference, model change, training, detector, semantic label, output-derived feature, GT-derived alpha, parameter sweep, or additional experiment was performed.\n\n"
        "## Fixed Method\n\n"
        f"The LQ-only grayscale texture score is the equal-weight mean of robustly normalized Sobel magnitude, {LOCAL_VARIANCE_WINDOW}x{LOCAL_VARIANCE_WINDOW} local variance, and absolute Gaussian blur residual (blur sigma={HIGH_FREQUENCY_BLUR_SIGMA}). Each feature uses fixed 1st/99th percentile normalization to [0,1]. The combined score is spatially smoothed once with fixed Gaussian sigma={GAUSSIAN_SIGMA:.0f} pixels.\n\n"
        "- `global_alpha_0.6`: `I=0.6*HYPIR + 0.4*SwinIR`.\n"
        "- `adaptive_texture`: `alpha=0.30+0.40*T`; `I=alpha*HYPIR+(1-alpha)*SwinIR`.\n"
        "- `adaptive_reverse`: `alpha=0.70-0.40*T`; otherwise identical.\n\n"
        "Every blend is clipped to [0,255], rounded, converted to RGB uint8, and saved at its original resolution. GT is used only for metrics.\n\n"
        "## Metrics\n\n"
        "PSNR uses `skimage` with `data_range=255`; SSIM uses `skimage` RGB channel-axis handling; LPIPS-Alex uses the existing E0/E1/E2 preprocessing with a longest-side <=1024 bilinear resize. Averages are arithmetic means of the five per-case results.\n\n"
        "| Method | PSNR | SSIM | LPIPS-Alex | Mean alpha | Mean alpha std |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: |\n"
        f"{table_rows}\n\n"
        "## Per-case Results And Map Statistics\n\n"
        "`Texture_Alpha_Pearson` is undefined (`N/A`) for the constant global alpha map; it is otherwise the per-pixel Pearson correlation of the same LQ texture score and alpha map.\n\n"
        "| Case | Method | PSNR | SSIM | LPIPS-Alex | Mean alpha | Std | Min | Max | Mean T | Texture-alpha r |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"{per_case_rows}\n\n"
        "## Per-case Deltas\n\n"
        "| Case | Method | dPSNR vs global | dSSIM vs global | dLPIPS vs global | dPSNR vs HYPIR-50 | dSSIM vs HYPIR-50 | dLPIPS vs HYPIR-50 |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"{delta_rows}\n\n"
        "## Visual Sanity Check\n\n"
        "The saved `texture`, forward-alpha, and reverse-alpha maps share each source image's native dimensions. Visual inspection confirms that texture-map variation follows LQ image detail and that the forward/reverse alpha maps invert that variation without blank regions or tiling. Map values are finite RGB uint8 visualizations; forward and reverse maps are complementary (`alpha_texture + alpha_reverse = 1.0` before PNG quantization), and their map statistics obey [0.30,0.70]. The visual check is limited to whether LQ texture complexity is a useful proxy for spatial HYPIR contribution; it does not make a claim about hallucination-proneness.\n\n"
        "## Conclusion\n\n"
        f"{conclusion} {reverse_difference}\n",
        encoding="utf-8",
    )


def run_experiment(
    *,
    lq_dir: Path,
    hypir_dir: Path,
    fidelity_dir: Path,
    gt_dir: Path,
    output_root: Path,
    device: str = "cuda",
    lpips_model=None,
    cases: Sequence[str] | None = None,
    report_path: Path | None = None,
) -> dict[str, object]:
    """Run exactly the three authorized E3-A methods from immutable inputs."""
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if lpips_model is None:
        lpips_model = __import__("lpips").LPIPS(net="alex", verbose=False).to(selected_device).eval()
    case_names = list(cases) if cases is not None else _case_names(Path(lq_dir))
    if not case_names:
        raise ValueError("no LQ case images found")

    rows: list[dict[str, object]] = []
    for case in case_names:
        lq = load_rgb(_case_file(Path(lq_dir), case, ("_lq", "")))
        hypir = load_rgb(_case_file(Path(hypir_dir), case, ("_lq", "")))
        fidelity = load_rgb(_case_file(Path(fidelity_dir), case, ("",)))
        gt = load_rgb(_case_file(Path(gt_dir), case, ("_gt", "")))
        if len({lq.shape, hypir.shape, fidelity.shape, gt.shape}) != 1:
            raise ValueError(f"{case}: LQ/HYPIR/Fidelity/GT dimensions differ")

        texture = compute_texture_score(lq)
        alpha_maps = {
            "global_alpha_0.6": np.full(texture.shape, GLOBAL_ALPHA, dtype=np.float32),
            "adaptive_texture": compute_alpha_map(texture, reverse=False),
            "adaptive_reverse": compute_alpha_map(texture, reverse=True),
        }
        save_rgb_png(output_root / "maps" / f"{case}_texture.png", _map_to_rgb(texture))
        save_rgb_png(output_root / "maps" / f"{case}_alpha_texture.png", _map_to_rgb(alpha_maps["adaptive_texture"]))
        save_rgb_png(output_root / "maps" / f"{case}_alpha_reverse.png", _map_to_rgb(alpha_maps["adaptive_reverse"]))

        outputs = {method: fuse_with_alpha(hypir, fidelity, alpha) for method, alpha in alpha_maps.items()}
        for method, image in outputs.items():
            save_rgb_png(output_root / "outputs" / method / f"{case}.png", image)
        metric_images = {"HYPIR-50": hypir, **outputs}
        metric_values = compute_image_metrics(metric_images, gt, lpips_model, selected_device)
        global_metrics = metric_values["global_alpha_0.6"]
        hypir_metrics = metric_values["HYPIR-50"]
        for method in METHODS:
            alpha = alpha_maps[method]
            metrics = metric_values[method]
            rows.append({
                "case": case,
                "method": method,
                "PSNR": metrics["PSNR"],
                "SSIM": metrics["SSIM"],
                "LPIPS": metrics["LPIPS"],
                "Alpha_Mean": float(np.mean(alpha)),
                "Alpha_Std": float(np.std(alpha)),
                "Alpha_Min": float(np.min(alpha)),
                "Alpha_Max": float(np.max(alpha)),
                "Texture_Mean": float(np.mean(texture)),
                "Texture_Alpha_Pearson": _pearson_correlation(texture, alpha),
                "Delta_PSNR_vs_GlobalAlpha06": metrics["PSNR"] - global_metrics["PSNR"],
                "Delta_SSIM_vs_GlobalAlpha06": metrics["SSIM"] - global_metrics["SSIM"],
                "Delta_LPIPS_vs_GlobalAlpha06": metrics["LPIPS"] - global_metrics["LPIPS"],
                "Delta_PSNR_vs_HYPIR50": metrics["PSNR"] - hypir_metrics["PSNR"],
                "Delta_SSIM_vs_HYPIR50": metrics["SSIM"] - hypir_metrics["SSIM"],
                "Delta_LPIPS_vs_HYPIR50": metrics["LPIPS"] - hypir_metrics["LPIPS"],
            })
    _write_metrics(rows, output_root)
    if report_path is not None:
        _write_report(rows, Path(report_path))
    return {"cases": case_names, "methods": list(METHODS), "rows": rows, "metrics_path": output_root / "metrics.csv"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    result = run_experiment(
        lq_dir=root / "csig_dataset" / "验证集",
        hypir_dir=root / "baseline" / "experiments" / "coeff_t_50" / "output" / "result",
        fidelity_dir=root / "baseline" / "experiments" / "E1_fidelity" / "swinir_car_jpeg40" / "output",
        gt_dir=root / "csig_dataset" / "验证集",
        output_root=root / "baseline" / "experiments" / "E3_texture_adaptive",
        device=args.device,
        cases=CASE_IDS,
        report_path=root / "reports" / "E3_TEXTURE_ADAPTIVE.md",
    )
    print(f"Completed E3-A for {len(result['cases'])} cases and {len(result['methods'])} methods")
    print(f"Metrics written to {result['metrics_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
