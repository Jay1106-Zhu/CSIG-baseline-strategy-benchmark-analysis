"""Evaluate confidence-fusion artifacts, including high/low-map regions."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms.functional import pil_to_tensor
import lpips

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "baseline" / "experiments" / "confidence_fusion_v1"
CASES = [f"case{i}" for i in range(1, 6)]
METHODS = ["LQ", "HYPIR-200", "HYPIR-50", "fixed_50_50", "confidence_200_50", "confidence_200_LQ"]


def load(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode != "RGB":
            raise ValueError(f"{path}: mode {im.mode} is not RGB")
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode != "L":
            raise ValueError(f"{path}: expected grayscale mask, got {im.mode}")
        return np.asarray(im.convert("L"), dtype=np.uint8).copy()


def path_for(case: str, method: str) -> Path:
    if method == "LQ":
        return ROOT / "baseline" / "input" / f"{case}_lq.jpg"
    if method == "HYPIR-200":
        return ROOT / "baseline" / "experiments" / "coeff_t_200" / "output" / "result" / f"{case}_lq.png"
    if method == "HYPIR-50":
        return ROOT / "baseline" / "experiments" / "coeff_t_50" / "output" / "result" / f"{case}_lq.png"
    sub = {"fixed_50_50": "fixed_50_50", "confidence_200_50": "confidence_200_50", "confidence_200_LQ": "confidence_200_lq"}[method]
    return OUT / "fusion" / sub / f"{case}.png"


def tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return pil_to_tensor(Image.fromarray(arr)).unsqueeze(0).to(device=device, dtype=torch.float32).div(127.5).sub(1)


def gradient(arr: np.ndarray) -> np.ndarray:
    gray = arr.astype(np.float32).mean(axis=2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
    return np.sqrt(gx * gx + gy * gy)


def region_stats(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float]:
    if not mask.any():
        return float("nan"), float("nan"), float("nan"), float("nan")
    diff = pred.astype(np.float32) - gt.astype(np.float32)
    pix = diff[mask]
    l1 = float(np.mean(np.abs(pix)))
    l2 = float(np.sqrt(np.mean(pix * pix)))
    psnr = float(20 * np.log10(255.0 / max(l2, 1e-12)))
    gd = float(np.mean(np.abs(gradient(pred)[mask] - gradient(gt)[mask])))
    return l1, l2, psnr, gd


def save_error_panel(case: str, arrays: dict[str, np.ndarray], gt: np.ndarray) -> None:
    names = ["LQ", "HYPIR-200", "HYPIR-50", "50/50", "Conf 200/50", "Conf 200/LQ"]
    keys = ["LQ", "HYPIR-200", "HYPIR-50", "fixed_50_50", "confidence_200_50", "confidence_200_LQ"]
    thumbs = []
    for name, key in zip(names, keys):
        d = np.abs(arrays[key].astype(np.int16) - gt.astype(np.int16)).astype(np.uint8)
        d = np.clip(d * 3, 0, 255).astype(np.uint8)
        im = Image.fromarray(d).resize((512, int(round(d.shape[0] * 512 / d.shape[1]))), Image.Resampling.LANCZOS)
        thumbs.append((name, im))
    gap, header = 8, 34
    w = sum(im.width for _, im in thumbs) + gap * (len(thumbs) - 1)
    h = header + max(im.height for _, im in thumbs)
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for name, im in thumbs:
        draw.text((x + 3, 6), name, fill=(0, 0, 0))
        canvas.paste(im, (x, header))
        x += im.width + gap
    canvas.save(OUT / "comparison" / f"{case}_error_maps.png", format="PNG")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    gt_dirs = [p for p in (ROOT / "csig_dataset").iterdir() if p.is_dir() and any(p.glob("case1_gt.*"))]
    if len(gt_dirs) != 1:
        raise FileNotFoundError("Could not uniquely locate validation GT directory")
    gt_dir = gt_dirs[0]
    for case in CASES:
        gt = load(gt_dir / f"{case}_gt.jpg")
        arrays = {method: load(path_for(case, method)) for method in METHODS}
        if len({arr.shape for arr in [gt, *arrays.values()]}) != 1:
            raise ValueError(f"{case}: shape mismatch")
        conf = load(OUT / "confidence_maps" / f"{case}.png")
        # Use the stored grayscale mask; threshold was the confidence median.
        high = load_mask(OUT / "masks" / f"{case}_high.png") > 127
        low = ~high
        lq = arrays["LQ"]
        with torch.inference_mode():
            lq_lp = float(model(tensor(lq, device), tensor(gt, device)).item())
        base_metrics = {}
        for method, arr in arrays.items():
            psnr = float(peak_signal_noise_ratio(gt, arr, data_range=255))
            ssim = float(structural_similarity(gt, arr, channel_axis=2, data_range=255))
            lp = lq_lp if method == "LQ" else float(model(tensor(arr, device), tensor(gt, device)).item())
            base_metrics[method] = (psnr, ssim, lp)
            lq_psnr, lq_ssim, lq_lpips = base_metrics["LQ"]
            rows.append({"case": case, "method": method, "PSNR": f"{psnr:.6f}", "SSIM": f"{ssim:.6f}", "LPIPS": f"{lp:.6f}", "Delta PSNR vs LQ": f"{psnr-lq_psnr:.6f}", "Delta SSIM vs LQ": f"{ssim-lq_ssim:.6f}", "Delta LPIPS vs LQ": f"{lp-lq_lpips:.6f}"})
            for region_name, mask in (("high_confidence", high), ("low_confidence", low)):
                l1, l2, rpsnr, gd = region_stats(arr, gt, mask)
                region_rows.append({"case": case, "method": method, "region": region_name, "pixels": int(mask.sum()), "L1": f"{l1:.6f}", "L2_RMSE": f"{l2:.6f}", "PSNR_equiv": f"{rpsnr:.6f}", "gradient_difference": f"{gd:.6f}"})
        save_error_panel(case, arrays, gt)
    fields = ["case", "method", "PSNR", "SSIM", "LPIPS", "Delta PSNR vs LQ", "Delta SSIM vs LQ", "Delta LPIPS vs LQ"]
    with (OUT / "evaluation_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        for method in METHODS:
            subset = [r for r in rows if r["method"] == method]
            avg = {"case": "Average", "method": method}
            for field in fields[2:]:
                avg[field] = f"{np.mean([float(r[field]) for r in subset]):.6f}"
            writer.writerow(avg)
    region_fields = ["case", "method", "region", "pixels", "L1", "L2_RMSE", "PSNR_equiv", "gradient_difference"]
    with (OUT / "region_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=region_fields); writer.writeheader(); writer.writerows(region_rows)
        for method in METHODS:
            for region in ("high_confidence", "low_confidence"):
                subset = [r for r in region_rows if r["method"] == method and r["region"] == region]
                avg = {"case": "Average", "method": method, "region": region, "pixels": int(np.mean([int(r["pixels"]) for r in subset]))}
                for field in region_fields[4:]: avg[field] = f"{np.mean([float(r[field]) for r in subset]):.6f}"
                writer.writerow(avg)
    print(f"Wrote {OUT / 'evaluation_metrics.csv'} and {OUT / 'region_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
