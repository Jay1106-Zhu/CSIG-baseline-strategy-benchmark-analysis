"""Offline Structure-Anchored Residual Fusion v2.

The experiment reuses existing LQ, HYPIR-50, HYPIR-200 and validation GT
images.  The v2 gate is computed from LQ structure and the HYPIR-200 residual;
GT is only used after fusion for evaluation.  No diffusion inference occurs.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from datetime import date
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


METHOD_V2 = "Structure-Anchored Residual Fusion v2"
METHOD_TEXTURE = "texture_selective_h200"
METHODS = ("LQ", "HYPIR-50", "HYPIR-200", METHOD_TEXTURE, METHOD_V2)
REGIONS = ("strong_edge", "textured_non_edge", "blurred_texture")
IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
_CASE_RE = re.compile(r"^(case\d+)(?:_(?:lq|gt|input|output))?$", re.IGNORECASE)


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
    if array.ndim != 3 or array.shape[2] != 3 or not np.isfinite(array).all():
        raise ValueError(f"{path}: expected finite RGB output, got {array.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8)).save(path, format="PNG")


def _robust_norm(values: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lo, hi = np.percentile(values, (low, high))
    if hi <= lo + 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _small_shape(height: int, width: int) -> tuple[int, int]:
    return max(32, height // 4), max(32, width // 4)


def _resize_map(values: np.ndarray, width: int, height: int) -> np.ndarray:
    values = cv2.GaussianBlur(values.astype(np.float32), (0, 0), 0.8)
    values = cv2.resize(values, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.asarray(values, dtype=np.float32)


def _gray_gradient(image: np.ndarray, small_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    small_h, small_w = small_size
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    return _resize_map(gx, width, height), _resize_map(gy, width, height), _resize_map(cv2.magnitude(gx, gy), width, height)


def compute_lq_structure(lq: np.ndarray) -> dict[str, np.ndarray]:
    """Extract LQ-only structure bands and local texture/blur proxies."""
    array = np.asarray(lq)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("LQ must be an RGB HxWx3 array")
    height, width = array.shape[:2]
    small_h, small_w = _small_shape(height, width)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    gx_small = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy_small = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    magnitude_small = cv2.GaussianBlur(cv2.magnitude(gx_small, gy_small), (0, 0), 0.8)
    gradient = _resize_map(_robust_norm(magnitude_small), width, height)
    gradient_x = _resize_map(gx_small, width, height)
    gradient_y = _resize_map(gy_small, width, height)

    mean = cv2.blur(small, (9, 9))
    variance = np.maximum(cv2.blur(small * small, (9, 9)) - mean * mean, 0.0)
    texture = _resize_map(_robust_norm(cv2.GaussianBlur(np.sqrt(variance), (0, 0), 0.8)), width, height)
    lap = np.abs(cv2.Laplacian(small, cv2.CV_32F, ksize=3))
    sharpness = _resize_map(_robust_norm(cv2.GaussianBlur(lap, (0, 0), 0.8)), width, height)
    blur_proxy = np.clip(1.0 - sharpness, 0.0, 1.0).astype(np.float32)

    canny_small = cv2.Canny(np.clip(np.rint(small * 255.0), 0, 255).astype(np.uint8), 40, 100).astype(np.float32) / 255.0
    canny_edge = np.clip(_resize_map(cv2.GaussianBlur(canny_small, (0, 0), 1.0), width, height), 0.0, 1.0)
    strong_cut = max(0.45, float(np.percentile(gradient, 80)))
    weak_cut = max(0.12, float(np.percentile(gradient, 45)))
    if strong_cut <= weak_cut:
        strong_cut = weak_cut + 0.10
    strong_edge = gradient >= strong_cut
    weak_edge = (gradient >= weak_cut) & ~strong_edge
    non_edge = ~(strong_edge | weak_edge)
    support = (strong_edge | weak_edge | (canny_edge >= 0.25)).astype(np.uint8)
    near_edge = cv2.dilate(support, np.ones((9, 9), np.uint8), iterations=1).astype(bool) & ~strong_edge
    return {
        "gradient_magnitude": np.clip(gradient, 0.0, 1.0).astype(np.float32),
        "gradient_x": gradient_x.astype(np.float32),
        "gradient_y": gradient_y.astype(np.float32),
        "texture": np.clip(texture, 0.0, 1.0).astype(np.float32),
        "blur_proxy": blur_proxy,
        "canny_edge": canny_edge.astype(np.float32),
        "strong_edge": strong_edge,
        "weak_edge": weak_edge,
        "non_edge": non_edge,
        "near_edge": near_edge,
    }


def compute_residual_gate(lq: np.ndarray, h200: np.ndarray, structure: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build the fixed v2 gate from LQ structure and H200 residual compatibility."""
    lq_array = np.asarray(lq)
    h200_array = np.asarray(h200)
    if lq_array.shape != h200_array.shape or lq_array.ndim != 3 or lq_array.shape[2] != 3:
        raise ValueError("LQ and HYPIR-200 must be same-size RGB arrays")
    structure = compute_lq_structure(lq_array) if structure is None else structure
    small_size = _small_shape(lq_array.shape[0], lq_array.shape[1])
    lq_gx, lq_gy, _ = _gray_gradient(lq_array, small_size)
    h_gx, h_gy, _ = _gray_gradient(h200_array, small_size)
    lq_mag_raw = np.hypot(lq_gx, lq_gy)
    h_mag_raw = np.hypot(h_gx, h_gy)
    residual = h200_array.astype(np.float32) - lq_array.astype(np.float32)
    residual_gray = cv2.cvtColor(residual, cv2.COLOR_RGB2GRAY)
    residual_small = cv2.resize(residual_gray, (small_size[1], small_size[0]), interpolation=cv2.INTER_AREA)
    residual_gx = cv2.Sobel(residual_small, cv2.CV_32F, 1, 0, ksize=3)
    residual_gy = cv2.Sobel(residual_small, cv2.CV_32F, 0, 1, ksize=3)
    residual_edge = _resize_map(_robust_norm(cv2.magnitude(residual_gx, residual_gy)), lq_array.shape[1], lq_array.shape[0])

    lq_edge = np.clip(structure["gradient_magnitude"], 0.0, 1.0)
    h_edge = _robust_norm(h_mag_raw)
    outside_support = np.clip(1.0 - np.maximum(lq_edge, structure["canny_edge"]), 0.0, 1.0)
    novelty = np.clip(residual_edge * outside_support * (1.0 - lq_edge), 0.0, 1.0)
    novelty = cv2.GaussianBlur(novelty.astype(np.float32), (0, 0), 1.0)
    new_edge = (novelty >= 0.40) & (outside_support >= 0.35) & (h_edge >= 0.35)

    denominator = np.maximum(lq_mag_raw * h_mag_raw, 1e-6)
    cosine = np.clip((lq_gx * h_gx + lq_gy * h_gy) / denominator, -1.0, 1.0)
    alignment = np.clip((cosine - 0.35) / 0.65, 0.0, 1.0) * lq_edge
    strong, weak, near = structure["strong_edge"], structure["weak_edge"], structure["near_edge"]
    gate = np.full(lq_edge.shape, 0.035, dtype=np.float32)
    gate[near & ~strong] = 0.065
    gate[weak] = 0.10
    gate[strong] = 0.020
    gate += 0.035 * alignment.astype(np.float32)
    gate *= 1.0 - 0.85 * novelty.astype(np.float32)
    gate = np.clip(gate, 0.0, 0.16).astype(np.float32)
    gate[strong] = np.minimum(gate[strong], 0.020)
    diagnostics = dict(structure)
    diagnostics.update({
        "residual_edge": np.clip(residual_edge, 0.0, 1.0).astype(np.float32),
        "new_edge_score": novelty.astype(np.float32),
        "new_edge": new_edge,
        "alignment": np.clip(alignment, 0.0, 1.0).astype(np.float32),
        "h200_edge": np.clip(h_edge, 0.0, 1.0).astype(np.float32),
    })
    return gate, diagnostics


def fuse_residual(lq: np.ndarray, h200: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Apply F = LQ + gate * (HYPIR-200 - LQ)."""
    lq_float = np.asarray(lq, dtype=np.float32)
    h200_float = np.asarray(h200, dtype=np.float32)
    gate_float = np.asarray(gate, dtype=np.float32)
    if lq_float.shape != h200_float.shape or lq_float.ndim != 3 or lq_float.shape[2] != 3:
        raise ValueError("LQ and HYPIR-200 must be same-size RGB arrays")
    if gate_float.shape != lq_float.shape[:2] or not np.isfinite(gate_float).all() or np.any((gate_float < 0) | (gate_float > 1)):
        raise ValueError("gate must match image dimensions and be finite in [0, 1]")
    return np.clip(lq_float + gate_float[..., None] * (h200_float - lq_float), 0.0, 255.0).astype(np.float32)


def _case_key(stem: str) -> str:
    match = _CASE_RE.fullmatch(stem)
    return match.group(1).casefold() if match else stem.casefold()


def _image_files(directory: Path) -> list[Path]:
    return sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS), key=lambda p: (int(re.search(r"\d+", p.stem).group()) if re.search(r"\d+", p.stem) else 10**9, p.name.casefold()))


def _case_file(directory: Path, case: str, suffixes: Iterable[str] = ()) -> Path:
    candidates = []
    expected = {case.casefold(), *(f"{case}{suffix}".casefold() for suffix in suffixes)}
    for path in _image_files(directory) if directory.is_dir() else ():
        if path.stem.casefold() in expected:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"{case}: no image in {directory}")
    return candidates[0]


def _metric(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred_u8 = np.clip(np.rint(pred), 0, 255).astype(np.uint8)
    if np.array_equal(pred_u8, gt):
        return float("inf"), 1.0
    return float(peak_signal_noise_ratio(gt, pred_u8, data_range=255)), float(structural_similarity(gt, pred_u8, channel_axis=2, data_range=255))


def _metric_delta(value: float, baseline: float) -> float:
    if math.isinf(value) and math.isinf(baseline):
        return 0.0
    return value - baseline


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


def _panel(path: Path, arrays: list[tuple[str, np.ndarray]], max_width: int = 1024) -> None:
    scale = min(1.0, max_width / max(image.shape[1] for _, image in arrays))
    thumbs = []
    for name, image in arrays:
        height, width = image.shape[:2]
        resized = Image.fromarray(np.clip(np.rint(image), 0, 255).astype(np.uint8)).resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
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


def _map_image(values: np.ndarray, color: bool = True) -> np.ndarray:
    values_u8 = np.clip(np.rint(np.asarray(values) * 255.0), 0, 255).astype(np.uint8)
    if not color:
        return np.repeat(values_u8[..., None], 3, axis=2)
    return cv2.applyColorMap(values_u8, cv2.COLORMAP_VIRIDIS)[:, :, ::-1]


def _region_masks(structure: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    strong = structure["strong_edge"]
    texture_hi = structure["texture"] >= np.percentile(structure["texture"], 80)
    blur_hi = structure["blur_proxy"] >= np.percentile(structure["blur_proxy"], 80)
    blurred = texture_hi & blur_hi & ~strong
    textured = texture_hi & ~strong & ~blurred
    return {"strong_edge": strong, "textured_non_edge": textured, "blurred_texture": blurred}


def _region_rows(case: str, lq: np.ndarray, gt: np.ndarray, outputs: dict[str, np.ndarray], structure: dict[str, np.ndarray], gate: np.ndarray, diagnostics: dict[str, np.ndarray]) -> list[dict[str, object]]:
    masks = _region_masks(structure)
    rows: list[dict[str, object]] = []
    lq_float, gt_float = lq.astype(np.float32), gt.astype(np.float32)
    for method, image in outputs.items():
        image_float = image.astype(np.float32)
        change = np.mean(np.abs(image_float - lq_float), axis=2)
        error = np.mean(np.abs(image_float - gt_float), axis=2)
        lq_gray = cv2.cvtColor(lq, cv2.COLOR_RGB2GRAY).astype(np.float32)
        out_gray = cv2.cvtColor(np.clip(np.rint(image), 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        grad_lq = cv2.magnitude(cv2.Sobel(lq_gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(lq_gray, cv2.CV_32F, 0, 1, ksize=3))
        grad_out = cv2.magnitude(cv2.Sobel(out_gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(out_gray, cv2.CV_32F, 0, 1, ksize=3))
        gradient_change = np.abs(grad_out - grad_lq)
        method_gate = gate if method == METHOD_V2 else None
        for region, mask in masks.items():
            if not mask.any():
                continue
            rows.append({
                "case": case,
                "method": method,
                "region": region,
                "pixels": int(mask.sum()),
                "mean_gate": "" if method_gate is None else f"{float(method_gate[mask].mean()):.6f}",
                "change_L1": f"{float(change[mask].mean()):.6f}",
                "error_to_GT_L1": f"{float(error[mask].mean()):.6f}",
                "gradient_change": f"{float(gradient_change[mask].mean()):.6f}",
                "new_edge_score": "" if method_gate is None else f"{float(diagnostics['new_edge_score'][mask].mean()):.6f}",
                "alignment": "" if method_gate is None else f"{float(diagnostics['alignment'][mask].mean()):.6f}",
            })
    return rows


def _write_summary(path: Path, metric_rows: list[dict[str, object]], region_rows: list[dict[str, object]], source_dirs: tuple[Path, Path, Path, Path, Path], *, lpips_enabled: bool) -> None:
    def mean(method: str, field: str) -> float:
        values = [float(row[field]) for row in metric_rows if row["case"] != "Average" and row["method"] == method and row[field] not in ("", "inf")]
        return float(np.mean(values)) if values else float("nan")

    avg_psnr = {method: mean(method, "PSNR") for method in METHODS}
    avg_ssim = {method: mean(method, "SSIM") for method in METHODS}
    v2_better = avg_psnr[METHOD_V2] >= max(avg_psnr["HYPIR-50"], avg_psnr[METHOD_TEXTURE]) + 0.05 and avg_ssim[METHOD_V2] >= max(avg_ssim["HYPIR-50"], avg_ssim[METHOD_TEXTURE]) - 0.001
    decision = "继续小规模验证" if v2_better else "停止该方向"
    lines = [
        "# Structure-Anchored Residual Fusion v2",
        "",
        f"结论：**{decision}**。这是一个只使用已有图像的离线验证；没有训练、LoRA 更新或新增 HYPIR/SD 推理。",
        "",
        "## 方法",
        "",
        "`R200 = HYPIR-200 - LQ`，最终输出为 `F = LQ + gate * R200`。gate 只由 LQ 的 Sobel 梯度、Canny 邻域、局部纹理/模糊代理，以及 H200 residual 的边缘新颖性和梯度方向一致性构成。GT 从未进入 gate 或任何阈值。",
        "",
        "固定门控：strong_edge=0.020；weak_edge=0.100；existing-structure-nearby=0.065；其他 non-edge=0.035。新边缘 novelty 会乘以 `(1 - 0.85 * novelty)`，方向一致最多增加 0.035，strong_edge 最终仍封顶 0.020。",
        "",
        "## 输入与产物",
        "",
        f"- LQ: `{source_dirs[0]}`；GT（仅评价）: `{source_dirs[1]}`",
        f"- HYPIR-200: `{source_dirs[2]}`；HYPIR-50: `{source_dirs[3]}`；current texture_selective_h200: `{source_dirs[4]}`",
        "- v2 只写入本目录的 `fusion/v2/`；旧实验目录不修改。",
        "",
        "## 全图指标平均",
        "",
        "| Method | PSNR | SSIM | LPIPS-Alex |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        lp = mean(method, "LPIPS_Alex") if lpips_enabled else float("nan")
        lines.append(f"| {method} | {avg_psnr[method]:.6f} | {avg_ssim[method]:.6f} | {'%.6f' % lp if np.isfinite(lp) else 'not computed'} |")
    lines.extend(["", "## Per-case PSNR (SSIM/LPIPS per case are in metrics.csv)", "", "| Case | LQ | H50 | H200 | texture_selective_h200 | v2 |", "|---|---:|---:|---:|---:|---:|"])
    cases = sorted({str(row["case"]) for row in metric_rows if row["case"] != "Average"}, key=lambda x: int(re.search(r"\d+", x).group()))
    for case in cases:
        values = {str(row["method"]): row for row in metric_rows if row["case"] == case}
        lines.append(f"| {case} | {float(values['LQ']['PSNR']):.4f} | {float(values['HYPIR-50']['PSNR']):.4f} | {float(values['HYPIR-200']['PSNR']):.4f} | {float(values[METHOD_TEXTURE]['PSNR']):.4f} | {float(values[METHOD_V2]['PSNR']):.4f} |")
    lines.extend(["", "## 分区域证据", "", "`region_metrics.csv` 同时记录 change_L1 与 error_to_GT_L1。change 变小只代表更少改写，error 变小才是对 GT 的恢复证据。", ""])
    for region in REGIONS:
        lines.append(f"### {region}")
        lines.append("")
        lines.append("| Method | change_L1 | error_to_GT_L1 |")
        lines.append("|---|---:|---:|")
        for method in METHODS:
            values = [row for row in region_rows if row["case"] != "Average" and row["method"] == method and row["region"] == region]
            if values:
                lines.append(f"| {method} | {np.mean([float(row['change_L1']) for row in values]):.6f} | {np.mean([float(row['error_to_GT_L1']) for row in values]):.6f} |")
        lines.append("")
    lines.extend([
        "## 重点 case 检查",
        "",
        "- case1 text：strong_edge gate 用于降低 H200 对文字笔画的改写；需以 panel 和 error_to_GT_L1 判定，不把锐度变化当作 OCR 正确。",
        "- case2 book spine：同样检查竖向书脊结构是否少改写；没有使用文字检测或 GT mask。",
        "- case3 bird：检查羽毛/轮廓与背景交界，方向一致只允许小幅残差。",
        "- case4 foliage：检查 blurred_texture 是否获得有限恢复，同时防止 H200 新边缘扩散。",
        "- case5 clock：检查圆环、刻度和指针的边缘保护；没有宣称数字或几何身份恢复。",
        "",
        "## 决策规则与限制",
        "",
        "本次不做参数 sweep。只有当 v2 平均 PSNR 比 H50 和 current texture_selective_h200 都至少高 0.05 dB，且平均 SSIM 不低于两者最优值 0.001 以上，才视为值得继续；否则停止这个方向。验证集只有 case1-case5，结论不代表未见测试集。",
        "",
        "详见 `metrics.csv`、`region_metrics.csv`、`gate_maps/` 和每个 case 的 `comparison/case*.png`。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(lq_dir: Path | str, gt_dir: Path | str, h200_dir: Path | str, h50_dir: Path | str, texture_dir: Path | str, out_dir: Path | str, *, compute_lpips: bool = True, lpips_size: int = 1024) -> dict[str, object]:
    lq_dir, gt_dir, h200_dir, h50_dir, texture_dir, out = map(Path, (lq_dir, gt_dir, h200_dir, h50_dir, texture_dir, out_dir))
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty experiment directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    lq_paths = _image_files(lq_dir)
    if not lq_paths:
        raise FileNotFoundError(f"No LQ images found in {lq_dir}")
    cases = [_case_key(path.stem) for path in lq_paths]
    lpips_model = None
    lpips_device = None
    if compute_lpips:
        import torch
        import lpips
        lpips_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_model = lpips.LPIPS(net="alex", verbose=False).to(lpips_device).eval()
    metric_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    for case in cases:
        lq = load_rgb(_case_file(lq_dir, case, ("_lq",)))
        gt = load_rgb(_case_file(gt_dir, case, ("_gt",)))
        h200 = load_rgb(_case_file(h200_dir, case, ("_lq",)))
        h50 = load_rgb(_case_file(h50_dir, case, ("_lq",)))
        texture = load_rgb(_case_file(texture_dir, case, ("_lq",)))
        if len({lq.shape, gt.shape, h200.shape, h50.shape, texture.shape}) != 1:
            raise ValueError(f"{case}: LQ/GT/H50/H200/texture dimensions do not match")
        structure = compute_lq_structure(lq)
        gate, diagnostics = compute_residual_gate(lq, h200, structure)
        v2 = fuse_residual(lq, h200, gate)
        outputs = {"LQ": lq, "HYPIR-50": h50, "HYPIR-200": h200, METHOD_TEXTURE: texture, METHOD_V2: v2}
        save_rgb(out / "fusion" / "v2" / f"{case}.png", v2)
        save_rgb(out / "gate_maps" / f"{case}_gate.png", _map_image(gate))
        save_rgb(out / "gate_maps" / f"{case}_new_edge.png", _map_image(diagnostics["new_edge_score"], color=False))
        save_rgb(out / "gate_maps" / f"{case}_alignment.png", _map_image(diagnostics["alignment"], color=False))
        bands = np.zeros(lq.shape[:2], dtype=np.float32)
        bands[structure["weak_edge"]] = 0.5
        bands[structure["strong_edge"]] = 1.0
        save_rgb(out / "gate_maps" / f"{case}_edge_bands.png", _map_image(bands, color=False))
        _panel(out / "comparison" / f"{case}.png", [("LQ", lq), ("H50", h50), ("H200", h200), ("v2", v2), ("GT", gt)])
        for method, image in outputs.items():
            psnr, ssim = _metric(image, gt)
            lp = "" if lpips_model is None else f"{_lpips_score(image, gt, lpips_model, lpips_device, lpips_size):.6f}"
            metric_rows.append({"case": case, "method": method, "PSNR": f"{psnr:.6f}", "SSIM": f"{ssim:.6f}", "LPIPS_Alex": lp})
        region_rows.extend(_region_rows(case, lq, gt, outputs, structure, gate, diagnostics))

    metric_fields = ["case", "method", "PSNR", "SSIM", "LPIPS_Alex", "Delta_PSNR_vs_LQ", "Delta_SSIM_vs_LQ", "Delta_LPIPS_vs_LQ"]
    lq_metrics = {row["case"]: row for row in metric_rows if row["method"] == "LQ"}
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        for row in metric_rows:
            base = lq_metrics[row["case"]]
            lp_delta = "" if not row["LPIPS_Alex"] else f"{float(row['LPIPS_Alex']) - float(base['LPIPS_Alex']):.6f}"
            writer.writerow({**row, "Delta_PSNR_vs_LQ": f"{_metric_delta(float(row['PSNR']), float(base['PSNR'])):.6f}", "Delta_SSIM_vs_LQ": f"{_metric_delta(float(row['SSIM']), float(base['SSIM'])):.6f}", "Delta_LPIPS_vs_LQ": lp_delta})
        for method in METHODS:
            values = [row for row in metric_rows if row["method"] == method]
            lp_values = [float(row["LPIPS_Alex"]) for row in values if row["LPIPS_Alex"]]
            lq_mean = float(np.mean([float(row["PSNR"]) for row in metric_rows if row["method"] == "LQ"]))
            lq_ssim = float(np.mean([float(row["SSIM"]) for row in metric_rows if row["method"] == "LQ"]))
            lq_lp = float(np.mean([float(row["LPIPS_Alex"]) for row in metric_rows if row["method"] == "LQ"])) if compute_lpips else None
            method_psnr = float(np.mean([float(row["PSNR"]) for row in values]))
            method_ssim = float(np.mean([float(row["SSIM"]) for row in values]))
            writer.writerow({"case": "Average", "method": method, "PSNR": f"{method_psnr:.6f}", "SSIM": f"{method_ssim:.6f}", "LPIPS_Alex": "" if not lp_values else f"{np.mean(lp_values):.6f}", "Delta_PSNR_vs_LQ": f"{_metric_delta(method_psnr, lq_mean):.6f}", "Delta_SSIM_vs_LQ": f"{_metric_delta(method_ssim, lq_ssim):.6f}", "Delta_LPIPS_vs_LQ": "" if not lp_values or lq_lp is None else f"{np.mean(lp_values) - lq_lp:.6f}"})
    region_fields = ["case", "method", "region", "pixels", "mean_gate", "change_L1", "error_to_GT_L1", "gradient_change", "new_edge_score", "alignment"]
    with (out / "region_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=region_fields)
        writer.writeheader()
        writer.writerows(region_rows)
    metadata = [
        "# Structure-Anchored Residual Fusion v2 metadata", "", f"- date: {date.today().isoformat()}", "- status: completed", "- scope: validation case1-case5; offline fusion only", f"- lq_dir: `{lq_dir}`", f"- gt_dir: `{gt_dir}` (evaluation only)", f"- h200_dir: `{h200_dir}`", f"- h50_dir: `{h50_dir}`", f"- texture_selective_dir: `{texture_dir}`", f"- output_dir: `{out}`", "- formula: `F = LQ + gate * (HYPIR-200 - LQ)`", "- feature_source: LQ Sobel gradient, Canny edge proximity, local variance, inverse Laplacian blur proxy", "- residual checks: H200 residual edge novelty and LQ/H200 gradient direction alignment", "- fixed gates: strong=0.020, weak=0.100, near=0.065, non-edge=0.035; max=0.160", "- GT usage: post-fusion PSNR/SSIM/LPIPS and regional error only; never used for masks or weights", "- inference: no model or LoRA training; no new HYPIR/SD inference", "",
    ]
    (out / "experiment_metadata.md").write_text("\n".join(metadata), encoding="utf-8")
    _write_summary(out / "summary.md", metric_rows, region_rows, (lq_dir, gt_dir, h200_dir, h50_dir, texture_dir), lpips_enabled=compute_lpips)
    (out / "README.md").write_text("# Structure-Anchored Residual Fusion v2\n\nRun with `..\\..\\..\\.conda\\python.exe baseline\\experiments\\structure_anchored_residual_fusion_v2.py`. Outputs are offline fusions of existing images. See `summary.md`.\n", encoding="utf-8")
    return {"cases": cases, "metrics": metric_rows, "regions": region_rows, "output_dir": out}


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-lpips", action="store_true", help="skip LPIPS for a quick smoke run")
    parser.add_argument("--lpips-size", type=int, default=1024)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = (args.output_dir if args.output_dir is not None else root / "baseline" / "experiments" / "structure_anchored_residual_fusion_v2").resolve()
    result = run_experiment(root / "baseline" / "input", root / "csig_dataset" / "验证集", root / "baseline" / "experiments" / "coeff_t_200" / "output" / "result", root / "baseline" / "experiments" / "coeff_t_50" / "output" / "result", root / "baseline" / "experiments" / "texture_weight_sweep_v2" / "fusion" / "texture_selective", out, compute_lpips=not args.no_lpips, lpips_size=args.lpips_size)
    print(f"Wrote {len(result['cases'])} cases to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
