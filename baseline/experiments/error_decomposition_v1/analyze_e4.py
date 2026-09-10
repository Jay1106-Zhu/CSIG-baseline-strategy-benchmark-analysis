"""Analyze case4 multi-seed H200 outputs. No new HYPIR inference."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "baseline" / "experiments" / "error_decomposition_v1"
GT = ROOT / "csig_dataset" / "验证集" / "case4_gt.jpg"
LQ = ROOT / "baseline" / "input" / "case4_lq.jpg"
H50 = ROOT / "baseline" / "experiments" / "coeff_t_50" / "output" / "result" / "case4_lq.png"
SEEDS = {
    "231": ROOT / "baseline" / "experiments" / "coeff_t_200" / "output" / "result" / "case4_lq.png",
    "17": OUT / "e4_seeds" / "seed_17" / "output" / "result" / "case4_lq.png",
    "89": OUT / "e4_seeds" / "seed_89" / "output" / "result" / "case4_lq.png",
    "401": OUT / "e4_seeds" / "seed_401" / "output" / "result" / "case4_lq.png",
}
CROPS = {
    "high02": (3072, 768, 3328, 1024),
    "high06": (2560, 1792, 2816, 2048),
    "mid04": (2816, 1024, 3072, 1280),
    "high04": (256, 1280, 512, 1536),
}


def load(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(a, b, data_range=255))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


def l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def font(size: int) -> ImageFont.ImageFont:
    path = Path(r"C:\Windows\Fonts\segoeui.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def caption(array: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(array)
    canvas = Image.new("RGB", (image.width, image.height + 26), (16, 16, 16))
    canvas.paste(image, (0, 26))
    ImageDraw.Draw(canvas).text((6, 4), text, fill=(240, 240, 240), font=font(15))
    return np.asarray(canvas, dtype=np.uint8)


def hstack(images: list[np.ndarray], gap: int = 6) -> np.ndarray:
    h = max(im.shape[0] for im in images)
    w = sum(im.shape[1] for im in images) + gap * (len(images) - 1)
    out = np.full((h, w, 3), 12, dtype=np.uint8)
    x = 0
    for im in images:
        out[: im.shape[0], x : x + im.shape[1]] = im
        x += im.shape[1] + gap
    return out


def vstack(images: list[np.ndarray], gap: int = 6) -> np.ndarray:
    w = max(im.shape[1] for im in images)
    h = sum(im.shape[0] for im in images) + gap * (len(images) - 1)
    out = np.full((h, w, 3), 12, dtype=np.uint8)
    y = 0
    for im in images:
        out[y : y + im.shape[0], : im.shape[1]] = im
        y += im.shape[0] + gap
    return out


def crop(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return array[y0:y1, x0:x1].copy()


def main() -> None:
    gt = load(GT)
    lq = load(LQ)
    h50 = load(H50)
    seeds = {name: load(path) for name, path in SEEDS.items()}
    stack = np.stack([seeds[name].astype(np.float32) for name in ("231", "17", "89", "401")], axis=0)
    mean_img = np.clip(np.rint(stack.mean(axis=0)), 0, 255).astype(np.uint8)
    median_img = np.clip(np.rint(np.median(stack, axis=0)), 0, 255).astype(np.uint8)
    var_map = stack.var(axis=0).mean(axis=2)
    var_norm = (var_map - var_map.min()) / (var_map.max() - var_map.min() + 1e-8)
    var_u8 = np.clip(np.rint(var_norm * 255), 0, 255).astype(np.uint8)
    vis_dir = OUT / "e4_analysis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mean_img).save(vis_dir / "mean.png")
    Image.fromarray(median_img).save(vis_dir / "median.png")
    Image.fromarray(var_u8).save(vis_dir / "variance_gray.png")

    rows = []
    for name, img in {"LQ": lq, "H50": h50, **seeds, "mean": mean_img, "median": median_img}.items():
        rows.append({"method": name, "PSNR": psnr(gt, img), "SSIM": ssim(gt, img), "L1_vs_LQ": l1(img, lq), "L1_vs_231": l1(img, seeds["231"])})
    names = ["231", "17", "89", "401"]
    pair_rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pair_rows.append({"a": a, "b": b, "PSNR": psnr(seeds[a], seeds[b]), "L1": l1(seeds[a], seeds[b])})

    with (vis_dir / "e4_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (vis_dir / "e4_pairwise.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)

    crop_rows = []
    sheets = []
    for cid, box in CROPS.items():
        tiles = [caption(crop(lq, box), f"{cid} LQ"), caption(crop(gt, box), "GT")]
        for name in names:
            patch = crop(seeds[name], box)
            tiles.append(caption(patch, f"seed {name}"))
            crop_rows.append({"crop": cid, "seed": name, "PSNR": psnr(crop(gt, box), patch), "L1_vs_GT": l1(patch, crop(gt, box))})
        tiles.append(caption(crop(mean_img, box), "mean"))
        sheets.append(hstack(tiles))
    Image.fromarray(vstack(sheets)).save(vis_dir / "key_crops.png")
    with (vis_dir / "e4_crop_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(crop_rows[0].keys()))
        writer.writeheader()
        writer.writerows(crop_rows)

    preview = Image.fromarray(mean_img).resize((1024, 768), Image.Resampling.BILINEAR)
    preview.save(vis_dir / "mean_preview.jpg", quality=90)
    var_preview = Image.fromarray(var_u8).resize((1024, 768), Image.Resampling.BILINEAR)
    var_preview.save(vis_dir / "variance_preview.jpg", quality=90)
    print("metrics")
    for row in rows:
        print(row)
    print("pairwise")
    for row in pair_rows:
        print(row)


if __name__ == "__main__":
    main()
