"""Offline error-decomposition experiments E0-E3. No HYPIR inference or source edits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms.functional import pil_to_tensor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_SEED = 20260909
LPIPS_MAX_SIDE = 1024
SCALES = (1, 2, 4, 8, 16)
ALPHAS = tuple(round(i * 0.1, 1) for i in range(11))
CASES = tuple(f"case{i}" for i in range(1, 6))

PATHS = {
    "lq": PROJECT_ROOT / "baseline" / "input",
    "gt": PROJECT_ROOT / "csig_dataset" / "验证集",
    "h200": PROJECT_ROOT / "baseline" / "experiments" / "coeff_t_200" / "output" / "result",
    "h50": PROJECT_ROOT / "baseline" / "experiments" / "coeff_t_50" / "output" / "result",
    "patch_metrics": PROJECT_ROOT / "baseline" / "experiments" / "structure_diagnosis" / "patch_metrics.csv",
}


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"{path}: expected RGB, got {image.mode}")
        array = np.asarray(image, dtype=np.uint8).copy()
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{path}: expected HxWx3, got {array.shape}")
    return array


def save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8)).save(path, format="PNG")


def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(gt, pred, data_range=255))


def ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(structural_similarity(gt, pred, channel_axis=2, data_range=255))


def mean_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    scale = min(1.0, max_side / max(image.size))
    if scale >= 1.0:
        return image
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.BILINEAR,
    )


def downscale(array: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return array
    image = Image.fromarray(array)
    size = (max(1, image.width // factor), max(1, image.height // factor))
    return np.asarray(image.resize(size, Image.Resampling.BICUBIC), dtype=np.uint8)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def caption(array: np.ndarray, text: str, bar: int = 28) -> np.ndarray:
    image = Image.fromarray(array)
    canvas = Image.new("RGB", (image.width, image.height + bar), (18, 18, 18))
    canvas.paste(image, (0, bar))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), text, fill=(240, 240, 240), font=font(16, bold=True))
    return np.asarray(canvas, dtype=np.uint8)


def hstack(images: list[np.ndarray], gap: int = 8, bg: int = 12) -> np.ndarray:
    height = max(im.shape[0] for im in images)
    width = sum(im.shape[1] for im in images) + gap * (len(images) - 1)
    out = np.full((height, width, 3), bg, dtype=np.uint8)
    x = 0
    for im in images:
        out[: im.shape[0], x : x + im.shape[1]] = im
        x += im.shape[1] + gap
    return out


def vstack(images: list[np.ndarray], gap: int = 8, bg: int = 12) -> np.ndarray:
    width = max(im.shape[1] for im in images)
    height = sum(im.shape[0] for im in images) + gap * (len(images) - 1)
    out = np.full((height, width, 3), bg, dtype=np.uint8)
    y = 0
    for im in images:
        out[y : y + im.shape[0], : im.shape[1]] = im
        y += im.shape[0] + gap
    return out


def case_paths(case: str) -> dict[str, Path]:
    return {
        "lq": PATHS["lq"] / f"{case}_lq.jpg",
        "gt": PATHS["gt"] / f"{case}_gt.jpg",
        "h200": PATHS["h200"] / f"{case}_lq.png",
        "h50": PATHS["h50"] / f"{case}_lq.png",
    }


def load_case(case: str) -> dict[str, np.ndarray]:
    paths = case_paths(case)
    images = {name: load_rgb(path) for name, path in paths.items()}
    shapes = {name: im.shape for name, im in images.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"{case} shape mismatch: {shapes}")
    return images


class LpipsAlex:
    def __init__(self, max_side: int | None) -> None:
        self.max_side = max_side
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips_model(self.device)

    def score(self, pred: np.ndarray, gt: np.ndarray) -> float:
        def tensor(array: np.ndarray) -> torch.Tensor:
            image = Image.fromarray(array)
            if self.max_side is not None:
                image = resize_max_side(image, self.max_side)
            return pil_to_tensor(image).unsqueeze(0).to(device=self.device, dtype=torch.float32).div(127.5).sub(1.0)

        with torch.inference_mode():
            value = self.model(tensor(pred), tensor(gt))
        return float(value.item())


def lpips_model(device: torch.device):
    import lpips

    model = lpips.LPIPS(net="alex")
    model.to(device)
    model.eval()
    return model


def write_phase0(out_dir: Path) -> None:
    """Freeze already-computed FR numbers. Do not recompute."""
    rows = [
        # PSNR/SSIM from coeff_t CSVs; LPIPS from structure_local (max-side 1024).
        {"case": "case1", "method": "LQ", "PSNR": 32.028801, "SSIM": 0.948453, "LPIPS_1024": 0.078794, "LPIPS_native_coeff": 0.072517},
        {"case": "case1", "method": "H50", "PSNR": 32.143319, "SSIM": 0.945300, "LPIPS_1024": 0.064659, "LPIPS_native_coeff": 0.135358},
        {"case": "case1", "method": "H200", "PSNR": 29.335323, "SSIM": 0.846634, "LPIPS_1024": 0.071112, "LPIPS_native_coeff": 0.452658},
        {"case": "case2", "method": "LQ", "PSNR": 28.189381, "SSIM": 0.831443, "LPIPS_1024": 0.103027, "LPIPS_native_coeff": 0.373972},
        {"case": "case2", "method": "H50", "PSNR": 28.976278, "SSIM": 0.825432, "LPIPS_1024": 0.079462, "LPIPS_native_coeff": 0.331810},
        {"case": "case2", "method": "H200", "PSNR": 24.213958, "SSIM": 0.731045, "LPIPS_1024": 0.108835, "LPIPS_native_coeff": 0.374753},
        {"case": "case3", "method": "LQ", "PSNR": 35.622082, "SSIM": 0.934958, "LPIPS_1024": 0.058367, "LPIPS_native_coeff": 0.170643},
        {"case": "case3", "method": "H50", "PSNR": 34.688713, "SSIM": 0.920412, "LPIPS_1024": 0.074223, "LPIPS_native_coeff": 0.198108},
        {"case": "case3", "method": "H200", "PSNR": 28.865398, "SSIM": 0.812097, "LPIPS_1024": 0.169974, "LPIPS_native_coeff": 0.405851},
        {"case": "case4", "method": "LQ", "PSNR": 18.056199, "SSIM": 0.309323, "LPIPS_1024": 0.654342, "LPIPS_native_coeff": 0.879949},
        {"case": "case4", "method": "H50", "PSNR": 18.084599, "SSIM": 0.310380, "LPIPS_1024": 0.543480, "LPIPS_native_coeff": 0.785311},
        {"case": "case4", "method": "H200", "PSNR": 16.352439, "SSIM": 0.257605, "LPIPS_1024": 0.376833, "LPIPS_native_coeff": 0.626074},
        {"case": "case5", "method": "LQ", "PSNR": 26.275637, "SSIM": 0.864316, "LPIPS_1024": 0.127044, "LPIPS_native_coeff": 0.285579},
        {"case": "case5", "method": "H50", "PSNR": 27.394758, "SSIM": 0.883153, "LPIPS_1024": 0.048371, "LPIPS_native_coeff": 0.244238},
        {"case": "case5", "method": "H200", "PSNR": 23.875816, "SSIM": 0.777249, "LPIPS_1024": 0.071578, "LPIPS_native_coeff": 0.445688},
    ]
    path = out_dir / "phase0_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    notes = out_dir / "phase0_protocol.md"
    notes.write_text(
        "\n".join(
            [
                "# Phase 0 metric freeze",
                "",
                "- PSNR / SSIM: `baseline/experiments/coeff_t_{50,200}/evaluation_metrics.csv` (native resolution, skimage, data_range=255).",
                "- LPIPS_1024: `baseline/experiments/structure_local_restoration_v1/metrics.csv` (Alex, max-side 1024 bilinear). Use this for FR/perception comparisons.",
                "- LPIPS_native_coeff: coeff_t CSVs (Alex, native 4K). Inflated vs 1024 protocol; do not mix.",
                "- No new inference. No HYPIR source changes.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def load_nonoverlap(case: str) -> list[dict[str, str]]:
    rows = []
    with PATHS["patch_metrics"].open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["case"] != case:
                continue
            if int(row["x0"]) % 256 or int(row["y0"]) % 256:
                continue
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no non-overlapping patches for {case}")
    return rows


def stratified_sample(rows: list[dict[str, str]], k_per: int, seed: int) -> list[dict[str, object]]:
    grads = np.asarray([float(row["lq_gt_gradient_difference"]) for row in rows], dtype=np.float64)
    q1, q2 = np.quantile(grads, [1.0 / 3.0, 2.0 / 3.0])
    buckets = {"low": [], "mid": [], "high": []}
    for row, grad in zip(rows, grads):
        if grad <= q1:
            buckets["low"].append(row)
        elif grad <= q2:
            buckets["mid"].append(row)
        else:
            buckets["high"].append(row)
    rng = np.random.default_rng(seed)
    picked: list[dict[str, object]] = []
    for stratum in ("low", "mid", "high"):
        pool = buckets[stratum]
        if len(pool) < k_per:
            raise RuntimeError(f"{stratum}: only {len(pool)} patches, need {k_per}")
        chosen = [pool[i] for i in rng.choice(len(pool), size=k_per, replace=False)]
        chosen.sort(key=lambda row: (int(row["y0"]), int(row["x0"])))
        for index, row in enumerate(chosen, start=1):
            picked.append({"stratum": stratum, "index": index, "q1": float(q1), "q2": float(q2), **row})
    return picked


def crop(array: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    return array[y0:y1, x0:x1].copy()


def make_panel(images: dict[str, np.ndarray], x0: int, y0: int, x1: int, y1: int, title: str) -> np.ndarray:
    native = []
    coarse = []
    for name, label in (("lq", "LQ"), ("h50", "H50"), ("h200", "H200"), ("gt", "GT")):
        patch = crop(images[name], x0, y0, x1, y1)
        native.append(caption(patch, f"{label} 1x"))
        small = downscale(patch, 4)
        up = np.asarray(Image.fromarray(small).resize((patch.shape[1], patch.shape[0]), Image.Resampling.NEAREST), dtype=np.uint8)
        coarse.append(caption(up, f"{label} 1/4 nearest"))
    header = caption(np.full((8, native[0].shape[1] * 4 + 24, 3), 12, dtype=np.uint8), title, bar=24)
    return vstack([header, hstack(native), hstack(coarse)], gap=6)


def run_e1(out_dir: Path, lpips_eval: LpipsAlex) -> list[dict[str, object]]:
    specs = (("case4", 8, SAMPLE_SEED), ("case3", 4, SAMPLE_SEED + 1))
    all_rows: list[dict[str, object]] = []
    for case, k_per, seed in specs:
        images = load_case(case)
        picked = stratified_sample(load_nonoverlap(case), k_per, seed)
        case_dir = out_dir / "e1_panels" / case
        case_dir.mkdir(parents=True, exist_ok=True)
        stratum_rows: dict[str, list[np.ndarray]] = {"low": [], "mid": [], "high": []}
        for item in picked:
            x0, y0, x1, y1 = (int(item[k]) for k in ("x0", "y0", "x1", "y1"))
            pid = f"{item['stratum']}{int(item['index']):02d}"
            panel = make_panel(images, x0, y0, x1, y1, f"{case} {pid}  ({x0},{y0})-({x1},{y1})")
            save_rgb(case_dir / f"{pid}.png", panel)
            thumb = np.asarray(Image.fromarray(panel).resize((panel.shape[1] // 2, panel.shape[0] // 2), Image.Resampling.BILINEAR), dtype=np.uint8)
            stratum_rows[str(item["stratum"])].append(caption(thumb, pid, bar=22))
            lq = crop(images["lq"], x0, y0, x1, y1)
            gt = crop(images["gt"], x0, y0, x1, y1)
            h200 = crop(images["h200"], x0, y0, x1, y1)
            h50 = crop(images["h50"], x0, y0, x1, y1)
            record = {
                "case": case,
                "patch_id": pid,
                "stratum": item["stratum"],
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "lq_gt_gradient_difference": float(item["lq_gt_gradient_difference"]),
                "required_change_l1": float(item["required_change_l1"]),
                "generated_change_200_l1": float(item["generated_change_200_l1"]),
                "improvement_200_l1": float(item["improvement_200_l1"]),
                "change_alignment_200": float(item["change_alignment_200"]),
                "psnr_lq": psnr(gt, lq),
                "psnr_h50": psnr(gt, h50),
                "psnr_h200": psnr(gt, h200),
                "lpips_lq": lpips_eval.score(lq, gt),
                "lpips_h50": lpips_eval.score(h50, gt),
                "lpips_h200": lpips_eval.score(h200, gt),
                "error_type": "",
                "structure_preserved": "",
                "new_object": "",
                "plant_morphology": "",
                "confidence": "",
                "notes": "",
            }
            all_rows.append(record)
        for stratum, thumbs in stratum_rows.items():
            if thumbs:
                save_rgb(case_dir / f"_sheet_{stratum}.png", vstack(thumbs, gap=4))
    fields = list(all_rows[0].keys())
    sample_path = out_dir / "e1_sample.csv"
    with sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    template = out_dir / "e1_labels_template.csv"
    with template.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    (out_dir / "e1_sampling.json").write_text(
        json.dumps({"seed_case4": SAMPLE_SEED, "seed_case3": SAMPLE_SEED + 1, "k_case4": 8, "k_case3": 4, "grid": "nonoverlap 256"}, indent=2),
        encoding="utf-8",
    )
    return all_rows


def run_e2(out_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in CASES:
        images = load_case(case)
        case_dir = out_dir / "e2_multiscale" / case
        case_dir.mkdir(parents=True, exist_ok=True)
        for factor in SCALES:
            tiles = []
            scaled = {name: downscale(images[name], factor) for name in ("lq", "h50", "h200", "gt")}
            gt = scaled["gt"]
            for name, label in (("lq", "LQ"), ("h50", "H50"), ("h200", "H200"), ("gt", "GT")):
                array = scaled[name]
                rows.append(
                    {
                        "case": case,
                        "scale": f"1/{factor}" if factor != 1 else "1x",
                        "factor": factor,
                        "method": label,
                        "PSNR": psnr(gt, array),
                        "SSIM": ssim(gt, array),
                        "width": array.shape[1],
                        "height": array.shape[0],
                    }
                )
                view = array
                if max(view.shape[0], view.shape[1]) > 720:
                    view_im = resize_max_side(Image.fromarray(view), 720)
                    view = np.asarray(view_im, dtype=np.uint8)
                tiles.append(caption(view, f"{label} 1/{factor}" if factor != 1 else f"{label} 1x"))
            save_rgb(case_dir / f"scale_{factor}.png", hstack(tiles, gap=6))
    path = out_dir / "e2_multiscale_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_e3(out_dir: Path, lpips_eval: LpipsAlex) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    preview_dir = out_dir / "e3_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        images = load_case(case)
        lq = images["lq"].astype(np.float32)
        h200 = images["h200"].astype(np.float32)
        gt = images["gt"]
        h50 = images["h50"]
        h50_change = mean_l1(h50, images["lq"])
        for alpha in ALPHAS:
            blend = np.clip(np.rint((1.0 - alpha) * lq + alpha * h200), 0, 255).astype(np.uint8)
            row = {
                "case": case,
                "alpha": alpha,
                "method": f"blend_{alpha:.1f}",
                "change_l1_vs_lq": mean_l1(blend, images["lq"]),
                "PSNR": psnr(gt, blend),
                "SSIM": ssim(gt, blend),
                "LPIPS_1024": lpips_eval.score(blend, gt),
            }
            rows.append(row)
            if case == "case4" and alpha in {0.0, 0.3, 0.5, 1.0}:
                preview = np.asarray(resize_max_side(Image.fromarray(blend), 720), dtype=np.uint8)
                save_rgb(preview_dir / f"case4_alpha_{alpha:.1f}.png", caption(preview, f"case4 blend a={alpha:.1f}"))
        rows.append(
            {
                "case": case,
                "alpha": "",
                "method": "H50",
                "change_l1_vs_lq": h50_change,
                "PSNR": psnr(gt, h50),
                "SSIM": ssim(gt, h50),
                "LPIPS_1024": lpips_eval.score(h50, gt),
            }
        )
    path = out_dir / "e3_blend_curve.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_phase0(out_dir)
    lpips_1024 = LpipsAlex(LPIPS_MAX_SIDE)
    lpips_patch = LpipsAlex(None)
    print("E1 sampling and panels", flush=True)
    run_e1(out_dir, lpips_patch)
    print("E2 multiscale", flush=True)
    run_e2(out_dir)
    print("E3 blend curve", flush=True)
    run_e3(out_dir, lpips_1024)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
