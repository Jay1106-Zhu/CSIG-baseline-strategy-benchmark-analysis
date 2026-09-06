"""E2 offline global alpha blend of existing HYPIR-50 and SwinIR outputs."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control_conditioned_v1.control_conditioned_hypir_v1 import _lpips_scores, _metric


ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CASES = tuple(f"case{i}" for i in range(1, 6))


def load_rgb(path: Path | str) -> np.ndarray:
    path = Path(path)
    with Image.open(path) as image:
        if image.mode != "RGB" or len(image.getbands()) != 3:
            raise ValueError(f"{path}: expected RGB, got mode={image.mode}")
        array = np.asarray(image, dtype=np.uint8).copy()
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{path}: expected HxWx3, got {array.shape}")
    if not np.isfinite(array).all() or int(array.min()) < 0 or int(array.max()) > 255:
        raise ValueError(f"{path}: invalid uint8 value range")
    return array


def save_rgb_png(path: Path | str, array: np.ndarray) -> None:
    value = np.asarray(array)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 HxWx3 output, got {value.shape} {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError("output contains non-finite values")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value).save(path, format="PNG")


def blend_arrays(hypir: np.ndarray, fidelity: np.ndarray, alpha: float) -> np.ndarray:
    """Blend HYPIR and Fidelity in image space using the requested global alpha."""
    if alpha not in ALPHAS:
        raise ValueError(f"alpha must be one of {ALPHAS}, got {alpha}")
    first = np.asarray(hypir)
    second = np.asarray(fidelity)
    if first.shape != second.shape or first.ndim != 3 or first.shape[2] != 3:
        raise ValueError(f"HYPIR/Fidelity shape mismatch: {first.shape} vs {second.shape}")
    if first.dtype != np.uint8 or second.dtype != np.uint8:
        raise ValueError("HYPIR and Fidelity inputs must be uint8")
    blended = alpha * first.astype(np.float32) + (1.0 - alpha) * second.astype(np.float32)
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def _case_file(directory: Path, case: str, suffixes: Iterable[str] = ()) -> Path:
    candidates: list[Path] = []
    allowed = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    expected = {case.casefold(), *(f"{case}{suffix}".casefold() for suffix in suffixes)}
    for path in directory.iterdir() if directory.is_dir() else ():
        if path.is_file() and path.suffix.casefold() in allowed and path.stem.casefold() in expected:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"{case}: no image in {directory}")
    return sorted(candidates, key=lambda item: item.name.casefold())[0]


def _ensure_empty_output(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)


def compute_image_metrics(
    images: dict[str, np.ndarray],
    gt: np.ndarray,
    lpips_model,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Compute E0 metrics, batching all image LPIPS predictions for one case."""
    lpips_values = _lpips_scores(images, gt, lpips_model, device, 1024)
    result: dict[str, dict[str, float]] = {}
    for name, image in images.items():
        psnr, ssim = _metric(image, gt)
        result[name] = {"PSNR": psnr, "SSIM": ssim, "LPIPS": lpips_values[name]}
    return result


def run_e2(
    *,
    hypir_dir: Path,
    fidelity_dir: Path,
    gt_dir: Path,
    output_root: Path,
    device: str = "cuda",
    reuse_outputs: bool = False,
) -> dict[str, object]:
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not reuse_outputs:
        _ensure_empty_output(output_root)
    elif not output_root.is_dir():
        raise FileNotFoundError(f"cannot reuse missing output directory: {output_root}")

    records: list[dict[str, object]] = []
    baseline_by_case: dict[str, dict[str, float]] = {}
    start_total = time.perf_counter()
    lpips_model = __import__("lpips").LPIPS(net="alex", verbose=False).to(selected_device).eval()

    for case in CASES:
        hypir = load_rgb(_case_file(hypir_dir, case, ("_lq",)))
        fidelity = load_rgb(_case_file(fidelity_dir, case))
        gt = load_rgb(_case_file(gt_dir, case, ("_gt",)))
        if len({hypir.shape, fidelity.shape, gt.shape}) != 1:
            raise ValueError(
                f"{case}: resolution mismatch HYPIR={hypir.shape[:2]}, "
                f"Fidelity={fidelity.shape[:2]}, GT={gt.shape[:2]}"
            )

        blend_images: dict[str, np.ndarray] = {}
        for alpha in ALPHAS:
            blended = blend_arrays(hypir, fidelity, alpha)
            output_path = output_root / "outputs" / f"alpha_{alpha:.1f}" / f"{case}.png"
            if reuse_outputs:
                if not output_path.is_file():
                    raise FileNotFoundError(f"missing existing blend output: {output_path}")
            else:
                save_rgb_png(output_path, blended)
            blend_images[f"{alpha:.1f}"] = blended

        metric_images = {"HYPIR-50": hypir, "Fidelity": fidelity, **blend_images}
        case_metrics = compute_image_metrics(metric_images, gt, lpips_model, selected_device)
        base = {
            "hypir_psnr": case_metrics["HYPIR-50"]["PSNR"],
            "hypir_ssim": case_metrics["HYPIR-50"]["SSIM"],
            "hypir_lpips": case_metrics["HYPIR-50"]["LPIPS"],
            "fidelity_psnr": case_metrics["Fidelity"]["PSNR"],
            "fidelity_ssim": case_metrics["Fidelity"]["SSIM"],
            "fidelity_lpips": case_metrics["Fidelity"]["LPIPS"],
        }
        baseline_by_case[case] = base
        for alpha in ALPHAS:
            metrics = case_metrics[f"{alpha:.1f}"]
            psnr, ssim, lpips_value = metrics["PSNR"], metrics["SSIM"], metrics["LPIPS"]
            records.append({
                "alpha": f"{alpha:.1f}", "case": case,
                "PSNR": f"{psnr:.6f}", "SSIM": f"{ssim:.6f}", "LPIPS": f"{lpips_value:.6f}",
                "Delta_PSNR_vs_HYPIR50": f"{psnr - base['hypir_psnr']:.6f}",
                "Delta_SSIM_vs_HYPIR50": f"{ssim - base['hypir_ssim']:.6f}",
                "Delta_LPIPS_vs_HYPIR50": f"{lpips_value - base['hypir_lpips']:.6f}",
                "Delta_PSNR_vs_Fidelity": f"{psnr - base['fidelity_psnr']:.6f}",
                "Delta_SSIM_vs_Fidelity": f"{ssim - base['fidelity_ssim']:.6f}",
                "Delta_LPIPS_vs_Fidelity": f"{lpips_value - base['fidelity_lpips']:.6f}",
            })

    fields = [
        "alpha", "case", "PSNR", "SSIM", "LPIPS",
        "Delta_PSNR_vs_HYPIR50", "Delta_SSIM_vs_HYPIR50", "Delta_LPIPS_vs_HYPIR50",
        "Delta_PSNR_vs_Fidelity", "Delta_SSIM_vs_Fidelity", "Delta_LPIPS_vs_Fidelity",
    ]
    averages: list[dict[str, object]] = []
    for alpha in ALPHAS:
        selected = [row for row in records if row["alpha"] == f"{alpha:.1f}"]
        average: dict[str, object] = {"alpha": f"{alpha:.1f}", "case": "Average"}
        for field in fields[2:]:
            average[field] = f"{float(np.mean([float(row[field]) for row in selected])):.6f}"
        averages.append(average)
    metrics_path = output_root / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([*records, *averages])

    total_seconds = time.perf_counter() - start_total
    metadata = output_root / "experiment_metadata.md"
    metadata.write_text(
        "# E2 Global Alpha Blend metadata\n\n"
        f"- cases: {', '.join(CASES)}\n- alphas: {', '.join(f'{a:.1f}' for a in ALPHAS)}\n"
        f"- hypir_dir: `{hypir_dir}`\n- fidelity_dir: `{fidelity_dir}`\n- gt_dir: `{gt_dir}`\n"
        f"- output_root: `{output_root}`\n- device: `{selected_device}`\n"
        f"- total_wall_time_seconds: {total_seconds:.3f}\n"
        "- formula: I = alpha * HYPIR-50 + (1-alpha) * Fidelity\n"
        "- processing: clip to [0,255], round, cast uint8, save RGB PNG\n"
        f"- reuse_outputs: {reuse_outputs}\n"
        "- inference: offline pixel blend only; no model inference, training, TTA, ensemble, or parameter changes\n"
        "- metrics: E0-compatible PSNR, SSIM, LPIPS-Alex (max-side 1024)\n",
        encoding="utf-8",
    )
    return {"records": records, "averages": averages, "metrics_path": metrics_path, "total_wall_time": total_seconds}


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--hypir-dir", type=Path, default=None)
    parser.add_argument("--fidelity-dir", type=Path, default=None)
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--recompute-metrics", action="store_true", help="reuse existing 30 PNGs and rewrite metrics only")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    base = root / "baseline" / "experiments" / "E2_global_blend"
    result = run_e2(
        hypir_dir=(args.hypir_dir or root / "baseline" / "experiments" / "coeff_t_50" / "output" / "result").resolve(),
        fidelity_dir=(args.fidelity_dir or root / "baseline" / "experiments" / "E1_fidelity" / "swinir_car_jpeg40" / "output").resolve(),
        gt_dir=(args.gt_dir or root / "csig_dataset" / "验证集").resolve(),
        output_root=(args.output_root or base).resolve(),
        device=args.device,
        reuse_outputs=args.recompute_metrics,
    )
    print(f"Completed {len(result['records'])} blend rows; wall_time={result['total_wall_time']:.3f}s")
    print(f"Metrics written to {result['metrics_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
