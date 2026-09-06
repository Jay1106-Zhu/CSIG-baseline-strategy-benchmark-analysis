"""Offline Adaptive H50/H200 Fusion v1.

The alpha map is computed from LQ only. Existing HYPIR-50, HYPIR-200 and
texture-selective images are consumed as immutable inputs; no model inference
or training happens here.
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
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    from baseline.experiments.structure_local_restoration import _case_file, _panel, load_rgb, save_rgb
except ModuleNotFoundError:  # direct script execution
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from baseline.experiments.structure_local_restoration import _case_file, _panel, load_rgb, save_rgb


ALPHA_MAX = 0.30
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
METHODS = ("LQ", "HYPIR-50", "HYPIR-200", "texture_selective_h200", "adaptive_h50_h200")


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
    """Compute LQ-only gradient, texture, blur, and strong-edge maps."""
    array = np.asarray(lq)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("LQ must be an RGB HxWx3 array")
    height, width = array.shape[:2]
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    small = cv2.resize(gray, (max(8, width // 4), max(8, height // 4)), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    edge_small = _robust_norm(cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), 1.0))
    mean = cv2.blur(small, (9, 9))
    variance = np.maximum(cv2.blur(small * small, (9, 9)) - mean * mean, 0.0)
    texture_small = _robust_norm(cv2.GaussianBlur(np.sqrt(variance), (0, 0), 1.0))
    sharpness_small = _robust_norm(cv2.GaussianBlur(np.abs(cv2.Laplacian(small, cv2.CV_32F, ksize=3)), (0, 0), 1.0))
    blur_small = np.clip(1.0 - sharpness_small, 0.0, 1.0).astype(np.float32)
    edge = _smooth_resize(edge_small, width, height)
    texture = _smooth_resize(texture_small, width, height)
    blur = _smooth_resize(blur_small, width, height)
    if float(edge.max()) <= 1e-6:
        strong_edge = np.zeros(edge.shape, dtype=bool)
    else:
        strong_edge = edge >= np.percentile(edge, 80.0)
    return {"edge": edge, "texture": texture, "blur": blur, "strong_edge": strong_edge}


def compute_alpha_map(features: dict[str, np.ndarray]) -> np.ndarray:
    """Build one fixed, interpretable alpha map from LQ feature maps only.

    ``alpha`` controls movement from H50 toward H200. Strong edges receive a
    large fixed penalty; texture and moderate structure receive allowance;
    blur contributes only when texture is present, so blur alone cannot cause
    unbounded generation.
    """
    required = ("edge", "texture", "blur", "strong_edge")
    if any(key not in features for key in required):
        raise ValueError(f"features must contain {required}")
    edge = np.clip(np.asarray(features["edge"], dtype=np.float32), 0.0, 1.0)
    texture = np.clip(np.asarray(features["texture"], dtype=np.float32), 0.0, 1.0)
    blur = np.clip(np.asarray(features["blur"], dtype=np.float32), 0.0, 1.0)
    strong_edge = np.asarray(features["strong_edge"], dtype=bool)
    if edge.shape != texture.shape or edge.shape != blur.shape or edge.shape != strong_edge.shape:
        raise ValueError("LQ feature maps must have identical shapes")
    non_edge = (~strong_edge).astype(np.float32)
    base = 0.035
    texture_term = 0.11 * texture
    structure_term = 0.045 * edge * non_edge
    blur_term = 0.035 * blur * texture * non_edge
    strong_edge_penalty = 0.18 * strong_edge.astype(np.float32)
    alpha = base + texture_term + structure_term + blur_term - strong_edge_penalty
    return np.clip(alpha, 0.0, ALPHA_MAX).astype(np.float32)


def fuse_h50_h200(h50: np.ndarray, h200: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Return ``(1-alpha)*H50 + alpha*H200`` as a float image."""
    first = np.asarray(h50, dtype=np.float32)
    second = np.asarray(h200, dtype=np.float32)
    weights = np.asarray(alpha, dtype=np.float32)
    if first.shape != second.shape or first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("H50 and H200 must be same-size RGB arrays")
    if weights.shape != first.shape[:2]:
        raise ValueError("alpha map must match image height and width")
    if not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError("alpha map must be finite and in [0, 1]")
    return np.clip((1.0 - weights[..., None]) * first + weights[..., None] * second, 0.0, 255.0).astype(np.float32)


def _metric(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred_u8 = np.clip(np.rint(pred), 0, 255).astype(np.uint8)
    if np.array_equal(pred_u8, gt):
        return float("inf"), 1.0
    return (
        float(peak_signal_noise_ratio(gt, pred_u8, data_range=255)),
        float(structural_similarity(gt, pred_u8, channel_axis=2, data_range=255)),
    )


def _lpips_scores(images: dict[str, np.ndarray], gt: np.ndarray, model, device, max_side: int) -> dict[str, float]:
    import torch
    from torchvision.transforms.functional import pil_to_tensor

    def tensor(array: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8))
        scale = min(1.0, max_side / max(image.size))
        if scale < 1.0:
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
        return pil_to_tensor(image).unsqueeze(0).to(device=device, dtype=torch.float32).div(127.5).sub(1.0)

    names = list(images)
    with torch.inference_mode():
        pred = torch.cat([tensor(images[name]) for name in names], dim=0)
        target = tensor(gt).expand(len(names), -1, -1, -1)
        values = model(pred, target).reshape(-1).detach().cpu().tolist()
    return {name: float(value) for name, value in zip(names, values)}


def _case_names(directory: Path) -> list[str]:
    cases = sorted({p.stem.casefold().removesuffix("_lq") for p in directory.iterdir() if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS})
    return [f"case{int(case[4:])}" if case.startswith("case") and case[4:].isdigit() else case for case in cases]


def _region_masks(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    edge = features["edge"]
    texture = features["texture"]
    blur = features["blur"]
    strong = features["strong_edge"]
    texture_hi = texture >= np.percentile(texture, 80.0)
    blur_hi = blur >= np.percentile(blur, 80.0)
    masks = {
        "strong_edge": strong.copy(),
        "blurred_texture": blur_hi & texture_hi & ~strong,
        "textured_non_edge": texture_hi & ~strong,
    }
    assigned = np.zeros(edge.shape, dtype=bool)
    ordered: dict[str, np.ndarray] = {}
    for name in ("strong_edge", "blurred_texture", "textured_non_edge"):
        ordered[name] = masks[name] & ~assigned
        assigned |= ordered[name]
    ordered["other"] = ~assigned
    return ordered


def _write_alpha_visual(path: Path, alpha: np.ndarray) -> None:
    color = cv2.applyColorMap(np.rint(np.clip(alpha / ALPHA_MAX, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[:, :, ::-1]
    save_rgb(path, color)


def _write_summary(out: Path, metric_rows: list[dict[str, object]], region_rows: list[dict[str, object]], lpips_enabled: bool) -> None:
    def avg(method: str, field: str) -> float:
        values = [float(row[field]) for row in metric_rows if row["method"] == method and row["case"] != "Average" and row[field] not in ("", None)]
        return float(np.mean(values)) if values else float("nan")

    adaptive_psnr, adaptive_ssim, adaptive_lp = avg("adaptive_h50_h200", "PSNR"), avg("adaptive_h50_h200", "SSIM"), avg("adaptive_h50_h200", "LPIPS_Alex")
    best_psnr, best_ssim, best_lp = 28.480280, 0.781421, 0.164546
    exceeds = adaptive_psnr > best_psnr and adaptive_ssim >= best_ssim and adaptive_lp <= best_lp
    case_lines = []
    for case in sorted({str(row["case"]) for row in metric_rows if row["case"] != "Average"}):
        row = next(row for row in metric_rows if row["case"] == case and row["method"] == "adaptive_h50_h200")
        base = next(row for row in metric_rows if row["case"] == case and row["method"] == "texture_selective_h200")
        delta = float(row["PSNR"]) - float(base["PSNR"])
        case_lines.append(f"| {case} | {float(base['PSNR']):.6f} | {float(row['PSNR']):.6f} | {delta:+.6f} | {'获益' if delta > 0 else '受损' if delta < 0 else '持平'} |")

    lines = [
        "# Adaptive H50/H200 Fusion v1 Summary", "",
        "## Decision", "",
        f"Adaptive average metrics are PSNR {adaptive_psnr:.6f}, SSIM {adaptive_ssim:.6f}, LPIPS-Alex {adaptive_lp:.6f} (LPIPS computed={lpips_enabled}).",
        f"The current best `texture_selective_h200` reference is PSNR {best_psnr:.6f}, SSIM {best_ssim:.6f}, LPIPS-Alex {best_lp:.6f}.",
        f"**1. 是否超过当前最佳？ {'是' if exceeds else '否'}。** {'三项指标均达到或超过参考，才视为明确超过。' if exceeds else '未同时明确超过参考，按决策标准停止 adaptive fusion 方向。'}",
        "",
        "## Per-case PSNR", "",
        "| Case | texture_selective_h200 | adaptive | Delta | 结果 |", "|---|---:|---:|---:|---|",
        *case_lines, "",
        "**2. 哪个 case 获益/受损？** 以上表格按 PSNR 列出；正值为获益，负值为受损。",
        "",
        "## Alpha interpretation", "",
        "Alpha uses only LQ Sobel gradient magnitude, local variance texture strength, inverse-Laplacian blur proxy, and an LQ strong-edge mask (top 20% gradient). The fixed formula is `clip(0.035 + 0.11*texture + 0.045*edge*(1-strong_edge) + 0.035*blur*texture*(1-strong_edge) - 0.18*strong_edge, 0, 0.30)`. Fusion is `(1-alpha)*H50 + alpha*H200`.",
        "**3. α(x) 是否比之前 texture mask 更有效？** 只能以全图和区域 GT error 判断；alpha 的连续性与边缘抑制是设计性质，不等于恢复正确。若本实验未超过参考，则没有证据表明它比 texture mask 更有效。",
        "",
        "## Regional evidence", "",
        "`region_metrics.csv` records `change_L1` (deviation from LQ) and `error_to_GT_L1` for LQ-defined strong_edge, textured_non_edge, and blurred_texture regions. Lower GT error is the restoration criterion; lower change only indicates preservation.",
        "",
        "| Region | H50 change | H50 error | texture-selective change | texture-selective error | adaptive change | adaptive error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    def region_avg(method: str, region: str, field: str) -> float:
        values = [float(row[field]) for row in region_rows if row["method"] == method and row["region"] == region]
        return float(np.mean(values)) if values else float("nan")
    for region in ("strong_edge", "textured_non_edge", "blurred_texture"):
        lines.append(
            f"| {region} | {region_avg('HYPIR-50', region, 'change_L1'):.4f} | {region_avg('HYPIR-50', region, 'error_to_GT_L1'):.4f} | "
            f"{region_avg('texture_selective_h200', region, 'change_L1'):.4f} | {region_avg('texture_selective_h200', region, 'error_to_GT_L1'):.4f} | "
            f"{region_avg('adaptive_h50_h200', region, 'change_L1'):.4f} | {region_avg('adaptive_h50_h200', region, 'error_to_GT_L1'):.4f} |"
        )
    lines.extend([
        "",
        "## Scope and stop", "",
        "No GT value entered alpha construction or threshold selection. No diffusion inference, training, LoRA, or parameter sweep was run. H50/H200/LQ/texture-selective files were reused read-only.",
        f"**4. {'继续' if exceeds else '失败，明确停止'} adaptive fusion 方向。** {'结果达到继续条件。' if exceeds else 'Adaptive v1 没有明确超过当前最佳，按要求不再 sweep 或扩展该方向。'}",
    ])
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(lq_dir: Path | str, gt_dir: Path | str, h50_dir: Path | str, h200_dir: Path | str, texture_dir: Path | str, out_dir: Path | str, *, compute_lpips: bool = False, lpips_size: int = 1024) -> dict[str, object]:
    lq_dir, gt_dir, h50_dir, h200_dir, texture_dir, out = map(Path, (lq_dir, gt_dir, h50_dir, h200_dir, texture_dir, out_dir))
    # The default output directory also contains this package's source files.
    # Refuse prior artifacts while allowing only those known source entries.
    source_entries = {"__init__.py", "experiment.py", "__pycache__"}
    artifact_entries = {"alpha_maps", "fusion", "comparison", "metrics.csv", "region_metrics.csv", "summary.md", "experiment_metadata.md", "README.md"}
    if out.exists() and any(path.name not in source_entries | artifact_entries for path in out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty experiment directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("alpha_maps", "fusion", "comparison"):
        (out / sub).mkdir(exist_ok=True)
    cases = _case_names(lq_dir)
    if not cases:
        raise FileNotFoundError(f"No LQ images found in {lq_dir}")
    lpips_model = lpips_device = None
    if compute_lpips:
        import torch
        import lpips
        lpips_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_model = lpips.LPIPS(net="alex", verbose=False).to(lpips_device).eval()
    metric_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    totals: dict[str, list[float]] = {method: [] for method in METHODS}
    totals_ssim: dict[str, list[float]] = {method: [] for method in METHODS}
    totals_lp: dict[str, list[float]] = {method: [] for method in METHODS}
    for case in cases:
        lq = load_rgb(_case_file(lq_dir, case, ("_lq",)))
        gt = load_rgb(_case_file(gt_dir, case, ("_gt",)))
        h50 = load_rgb(_case_file(h50_dir, case, ("_lq",)))
        h200 = load_rgb(_case_file(h200_dir, case, ("_lq",)))
        texture = load_rgb(_case_file(texture_dir, case, ("_lq",)))
        if len({lq.shape, gt.shape, h50.shape, h200.shape, texture.shape}) != 1:
            raise ValueError(f"{case}: LQ/GT/H50/H200/texture dimensions do not match")
        features = compute_lq_features(lq)
        alpha = compute_alpha_map(features)
        adaptive = fuse_h50_h200(h50, h200, alpha)
        outputs = {"LQ": lq, "HYPIR-50": h50, "HYPIR-200": h200, "texture_selective_h200": texture, "adaptive_h50_h200": adaptive}
        _write_alpha_visual(out / "alpha_maps" / f"{case}.png", alpha)
        Image.fromarray(np.rint(np.clip(alpha / ALPHA_MAX, 0, 1) * 255).astype(np.uint8)).save(out / "alpha_maps" / f"{case}_gray.png", format="PNG")
        save_rgb(out / "fusion" / f"{case}.png", adaptive)
        _panel(out / "comparison" / f"{case}.png", [("LQ", lq), ("H50", h50), ("H200", h200), ("texture_selective", texture), ("adaptive", adaptive), ("GT", gt)], max_width=768)
        lp_values = {} if lpips_model is None else _lpips_scores(outputs, gt, lpips_model, lpips_device, lpips_size)
        lq_psnr, lq_ssim = _metric(lq, gt)
        lq_lp = float("nan") if lpips_model is None else lp_values["LQ"]
        masks = _region_masks(features)
        for method, image in outputs.items():
            psnr, ssim = _metric(image, gt)
            lp = float("nan") if lpips_model is None else lp_values[method]
            metric_rows.append({"case": case, "method": method, "PSNR": f"{psnr:.6f}", "SSIM": f"{ssim:.6f}", "LPIPS_Alex": "" if math.isnan(lp) else f"{lp:.6f}", "Delta_PSNR_vs_LQ": f"{psnr-lq_psnr:.6f}", "Delta_SSIM_vs_LQ": f"{ssim-lq_ssim:.6f}", "Delta_LPIPS_vs_LQ": "" if math.isnan(lp) else f"{lp-lq_lp:.6f}"})
            totals[method].append(psnr); totals_ssim[method].append(ssim)
            if not math.isnan(lp): totals_lp[method].append(lp)
            change = np.mean(np.abs(image.astype(np.float32) - lq.astype(np.float32)), axis=2)
            error = np.mean(np.abs(image.astype(np.float32) - gt.astype(np.float32)), axis=2)
            for region, mask in masks.items():
                if mask.any():
                    region_rows.append({"case": case, "method": method, "region": region, "pixels": int(mask.sum()), "mean_alpha": f"{float(alpha[mask].mean()):.6f}", "change_L1": f"{float(change[mask].mean()):.6f}", "error_to_GT_L1": f"{float(error[mask].mean()):.6f}"})
    metric_fields = ["case", "method", "PSNR", "SSIM", "LPIPS_Alex", "Delta_PSNR_vs_LQ", "Delta_SSIM_vs_LQ", "Delta_LPIPS_vs_LQ"]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields); writer.writeheader(); writer.writerows(metric_rows)
        lq_psnr = float(np.mean(totals["LQ"])); lq_ssim = float(np.mean(totals_ssim["LQ"])); lq_lp = float(np.mean(totals_lp["LQ"])) if totals_lp["LQ"] else None
        for method in METHODS:
            psnr = float(np.mean(totals[method])); ssim = float(np.mean(totals_ssim[method])); lp = float(np.mean(totals_lp[method])) if totals_lp[method] else None
            writer.writerow({"case": "Average", "method": method, "PSNR": f"{psnr:.6f}", "SSIM": f"{ssim:.6f}", "LPIPS_Alex": "" if lp is None else f"{lp:.6f}", "Delta_PSNR_vs_LQ": f"{psnr-lq_psnr:.6f}", "Delta_SSIM_vs_LQ": f"{ssim-lq_ssim:.6f}", "Delta_LPIPS_vs_LQ": "" if lp is None or lq_lp is None else f"{lp-lq_lp:.6f}"})
    with (out / "region_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "method", "region", "pixels", "mean_alpha", "change_L1", "error_to_GT_L1"]); writer.writeheader(); writer.writerows(region_rows)
    (out / "experiment_metadata.md").write_text(f"# Adaptive H50/H200 Fusion v1 Metadata\n\n- date: {date.today().isoformat()}\n- scope: validation case1-case5; offline fusion only\n- lq_dir: `{lq_dir}`\n- gt_dir: `{gt_dir}` (post-fusion metrics only)\n- h50_dir: `{h50_dir}`\n- h200_dir: `{h200_dir}`\n- texture_selective_dir: `{texture_dir}`\n- formula: `F=(1-alpha)*H50+alpha*H200`\n- alpha_formula: `clip(0.035 + 0.11*texture + 0.045*edge*(1-strong_edge) + 0.035*blur*texture*(1-strong_edge) - 0.18*strong_edge, 0, 0.30)`\n- alpha_source: LQ-only Sobel gradient, local variance, blur proxy, and top-20-percent strong-edge mask\n- gt_usage: evaluation only; never used in alpha or thresholds\n- inference: no diffusion inference, training, or LoRA\n- lpips: Alex with max-side {lpips_size}px resize, matching prior local-restoration metrics\n", encoding="utf-8")
    _write_summary(out, metric_rows, region_rows, compute_lpips)
    (out / "README.md").write_text("# Adaptive H50/H200 Fusion v1\n\nOffline fusion of existing LQ, HYPIR-50, HYPIR-200, and texture-selective outputs. See `summary.md`.\n", encoding="utf-8")
    return {"cases": cases, "metrics": metric_rows, "regions": region_rows, "output_dir": out}


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--with-lpips", action="store_true")
    parser.add_argument("--lpips-size", type=int, default=1024)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = (args.output_dir or root / "baseline" / "experiments" / "adaptive_fusion_v1").resolve()
    result = run_experiment(root / "baseline" / "input", root / "csig_dataset" / "验证集", root / "baseline" / "experiments" / "coeff_t_50" / "output" / "result", root / "baseline" / "experiments" / "coeff_t_200" / "output" / "result", root / "baseline" / "experiments" / "texture_weight_sweep_v2" / "fusion" / "texture_selective", out, compute_lpips=args.with_lpips, lpips_size=args.lpips_size)
    print(f"Wrote {len(result['cases'])} cases to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
