"""Common post-inference benchmark for Identity, HYPIR-50, and DiffIR."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import lpips
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
_CASE_RE = re.compile(r"case(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class BenchmarkCase:
    case: str
    lq: Path
    gt: Path
    outputs: Mapping[str, Path]


def _case_key(path: Path) -> str:
    match = _CASE_RE.search(path.stem)
    return f"case{int(match.group(1))}" if match else path.stem.casefold()


def _files(directory: Path) -> list[Path]:
    return sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS), key=lambda p: (_case_key(p), p.name.casefold()))


def _map_by_case(directory: Path, suffix: str | None = None) -> dict[str, Path]:
    files = _files(directory)
    if suffix is not None:
        files = [path for path in files if path.stem.casefold().endswith(suffix)]
    return {_case_key(path): path for path in files}


def discover_benchmark_cases(lq_dir: Path | str, gt_dir: Path | str, output_dirs: Mapping[str, Path | str]) -> list[BenchmarkCase]:
    """Match images by decoded-file directories and case number, independent of extension."""

    lq_paths = [p for p in _files(Path(lq_dir)) if p.stem.casefold().endswith("_lq")]
    gt_by_case = _map_by_case(Path(gt_dir), "_gt")
    outputs_by_model = {model: _map_by_case(Path(directory), "_lq") for model, directory in output_dirs.items()}
    if not lq_paths:
        raise FileNotFoundError(f"No LQ images found in {lq_dir}")
    cases: list[BenchmarkCase] = []
    for lq in lq_paths:
        key = _case_key(lq)
        if key not in gt_by_case:
            raise FileNotFoundError(f"Missing GT for {key}")
        missing = [model for model, files in outputs_by_model.items() if key not in files]
        if missing:
            raise FileNotFoundError(f"Missing outputs for {key}: {', '.join(missing)}")
        cases.append(BenchmarkCase(key, lq, gt_by_case[key], {model: files[key] for model, files in outputs_by_model.items()}))
    return cases


def load_rgb(path: Path | str) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def resize_for_perceptual(image: Image.Image, max_side: int = 1024) -> Image.Image:
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    scale = min(1.0, max_side / max(image.size))
    if scale == 1.0:
        return image.copy()
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)


def _metric_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device).div(127.5).sub(1.0)


def _lpips(model: lpips.LPIPS, first: Image.Image, second: Image.Image, device: torch.device) -> float:
    with torch.inference_mode():
        return float(model(_metric_tensor(resize_for_perceptual(first), device), _metric_tensor(resize_for_perceptual(second), device)).item())


def _build_dists(device: torch.device) -> Any:
    try:
        from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
        return DeepImageStructureAndTextureSimilarity().to(device).eval()
    except (ImportError, OSError, RuntimeError):
        return None


def _dists(metric: Any, first: Image.Image, second: Image.Image, device: torch.device) -> float | None:
    if metric is None:
        return None
    with torch.inference_mode():
        value = metric(_metric_tensor(resize_for_perceptual(first), device).add(1).div(2), _metric_tensor(resize_for_perceptual(second), device).add(1).div(2))
    return float(value.item())


def evaluate_case(case: BenchmarkCase, model: str, output_path: Path, lpips_model: lpips.LPIPS, dists_model: Any, device: torch.device, runtime: float | None = None, peak_vram: float | None = None) -> dict[str, Any]:
    lq, gt, output = load_rgb(case.lq), load_rgb(case.gt), load_rgb(output_path)
    if len({lq.size, gt.size, output.size}) != 1:
        raise ValueError(f"Dimension mismatch for {case.case}/{model}: LQ={lq.size}, GT={gt.size}, output={output.size}")
    lq_array, gt_array, output_array = map(lambda image: np.asarray(image, dtype=np.uint8), (lq, gt, output))
    dists_value = _dists(dists_model, output, gt, device)
    return {
        "model": model,
        "case": case.case,
        "PSNR": float(peak_signal_noise_ratio(gt_array, output_array, data_range=255)),
        "SSIM": float(structural_similarity(gt_array, output_array, channel_axis=2, data_range=255)),
        "LPIPS": _lpips(lpips_model, output, gt, device),
        "DISTS": dists_value,
        "MUSIQ": None,
        "MANIQA": None,
        "CLIPIQA": None,
        "runtime_sec": runtime,
        "peak_vram_gb": peak_vram,
        "lq_psnr": float(peak_signal_noise_ratio(gt_array, lq_array, data_range=255)),
        "lq_ssim": float(structural_similarity(gt_array, lq_array, channel_axis=2, data_range=255)),
        "output_path": str(output_path),
    }


RESULT_FIELDS = ["model", "case", "PSNR", "SSIM", "LPIPS", "DISTS", "MUSIQ", "MANIQA", "CLIPIQA", "runtime_sec", "peak_vram_gb"]


def write_results(rows: Iterable[dict[str, Any]], per_case_path: Path | str, average_path: Path | str) -> None:
    rows = list(rows)
    for target in (Path(per_case_path), Path(average_path)):
        target.parent.mkdir(parents=True, exist_ok=True)
    with Path(per_case_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in RESULT_FIELDS} for row in rows)
    averages: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in rows}):
        group = [row for row in rows if row["model"] == model]
        average = {"model": model, "case": "average"}
        for field in RESULT_FIELDS[2:]:
            values = [row[field] for row in group if isinstance(row.get(field), (float, int)) and not isinstance(row.get(field), bool)]
            average[field] = float(np.mean(values)) if values else None
        averages.append(average)
    with Path(average_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(averages)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _panel(image: Image.Image, width: int, height: int, label: str) -> Image.Image:
    panel = Image.new("RGB", (width, height + 34), "white")
    image = image.copy()
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel.paste(image, ((width - image.width) // 2, 34 + (height - image.height) // 2))
    ImageDraw.Draw(panel).text((8, 8), label, fill="black", font=_font(20))
    return panel


def create_visuals(cases: list[BenchmarkCase], full_path: Path | str, crops_path: Path | str, *, crop_boxes: Mapping[str, tuple[int, int, int, int]]) -> None:
    labels = [("LQ", "LQ"), ("GT", "GT"), ("Identity", "Identity"), ("HYPIR", "HYPIR"), ("DiffIR", "DiffIR")]
    full_panels: list[list[Image.Image]] = []
    crop_panels: list[list[Image.Image]] = []
    for case in cases:
        images = {
            "LQ": load_rgb(case.lq),
            "GT": load_rgb(case.gt),
            "Identity": load_rgb(case.outputs["Identity"]),
            "HYPIR": load_rgb(case.outputs["HYPIR"]),
            "DiffIR": load_rgb(case.outputs["DiffIR"]),
        }
        full_panels.append([_panel(images[key], 360, 260, f"{case.case} | {label}") for key, label in labels])
        box = crop_boxes[case.case]
        crop_panels.append([_panel(images[key].crop(box), 360, 300, f"{case.case} crop | {label}") for key, label in labels])
    def stack(rows: list[list[Image.Image]], target: Path | str) -> None:
        width, height = sum(panel.width for panel in rows[0]), sum(row[0].height for row in rows)
        canvas = Image.new("RGB", (width, height), "#eeeeee")
        y = 0
        for row in rows:
            x = 0
            for panel in row:
                canvas.paste(panel, (x, y))
                x += panel.width
            y += row[0].height
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(target, format="JPEG", quality=92)
    stack(full_panels, full_path)
    stack(crop_panels, crops_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    root = args.root
    cases = discover_benchmark_cases(root / "baseline" / "input", root / "csig_dataset" / "验证集", {"HYPIR": root / "baseline" / "experiments" / "coeff_t_50" / "output" / "result", "DiffIR": root / "baseline_bakeoff" / "outputs" / "diffir", "Identity": root / "baseline" / "input"})
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    metric_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    dists_model = _build_dists(device)
    manifest_path = root / "baseline_bakeoff" / "logs" / "diffir_inference_manifest.json"
    manifest_cases = {}
    if manifest_path.is_file():
        import json
        manifest_cases = {item["case"]: item for item in json.loads(manifest_path.read_text(encoding="utf-8")).get("cases", [])}
    rows: list[dict[str, Any]] = []
    for case in cases:
        for model, output in [("Identity", case.lq), ("HYPIR", case.outputs["HYPIR"]), ("DiffIR", case.outputs["DiffIR"])]:
            info = manifest_cases.get(case.case, {}) if model == "DiffIR" else {}
            rows.append(evaluate_case(case, model, output, metric_model, dists_model, device, info.get("runtime_sec") or (0.0 if model == "Identity" else None), info.get("peak_vram_gb")))
    write_results(rows, root / "baseline_bakeoff" / "results_per_case.csv", root / "baseline_bakeoff" / "results_average.csv")
    crop_boxes = {"case1": (0, 400, 2048, 1800), "case2": (900, 800, 2400, 2500), "case3": (2100, 1300, 3200, 2500), "case4": (1300, 900, 3000, 2400), "case5": (700, 600, 2700, 2800)}
    create_visuals(cases, root / "baseline_bakeoff" / "visuals" / "full_comparison.jpg", root / "baseline_bakeoff" / "visuals" / "detail_crops.jpg", crop_boxes=crop_boxes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
