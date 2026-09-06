"""Evaluate LQ and HYPIR outputs against ground truth images.

The script is independent from HYPIR inference. It validates image mode and
dimensions before calculating metrics and never silently resizes mismatches.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import lpips
import numpy as np
import torch
from PIL import Image, ImageOps
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms.functional import pil_to_tensor


IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
_CASE_RE = re.compile(r"case(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class EvaluationCase:
    case: str
    lq_path: Path
    gt_path: Path
    output_path: Path


@dataclass(frozen=True)
class MetricRow:
    case: str
    lq_psnr: float
    output_psnr: float
    delta_psnr: float
    psnr_conclusion: str
    lq_ssim: float
    output_ssim: float
    delta_ssim: float
    ssim_conclusion: str
    lq_lpips: float
    output_lpips: float
    delta_lpips: float
    lpips_conclusion: str
    lq_path: Path
    gt_path: Path
    output_path: Path


@dataclass(frozen=True)
class EvaluationSummary:
    rows: list[MetricRow]
    average: MetricRow
    csv_path: Path

    @property
    def case_count(self) -> int:
        return len(self.rows)


def _sort_key(path: Path) -> tuple[int, object, str]:
    match = _CASE_RE.fullmatch(path.stem)
    if match:
        return (0, int(match.group(1)), path.name.casefold())
    return (1, path.stem.casefold(), path.name.casefold())


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS),
        key=_sort_key,
    )


def _case_key(stem: str) -> str:
    normalized = stem.casefold()
    for suffix in ("_lq", "_gt", "_input", "_output"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _by_case_key(paths: Iterable[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in paths:
        result.setdefault(_case_key(path.stem), path)
    return result


def match_evaluation_cases(lq_dir: Path | str, gt_dir: Path | str, output_dir: Path | str) -> list[EvaluationCase]:
    """Match LQ, GT and output by normalized case key, without opening files."""

    lq_paths = _image_files(Path(lq_dir))
    gt_by_case = _by_case_key(_image_files(Path(gt_dir)))
    output_by_case = _by_case_key(_image_files(Path(output_dir)))
    if not lq_paths:
        raise FileNotFoundError(f"No input images found in {Path(lq_dir)}")
    cases: list[EvaluationCase] = []
    missing: list[str] = []
    for lq_path in lq_paths:
        key = _case_key(lq_path.stem)
        gt_path = gt_by_case.get(key)
        output_path = output_by_case.get(key)
        if gt_path is None or output_path is None:
            missing_items = []
            if gt_path is None:
                missing_items.append("GT")
            if output_path is None:
                missing_items.append("Output")
            missing.append(f"{lq_path.name} ({', '.join(missing_items)})")
            continue
        cases.append(EvaluationCase(key, lq_path, gt_path, output_path))
    if missing:
        raise FileNotFoundError("Missing matched files: " + "; ".join(missing))
    return cases


def _load_rgb(path: Path | str) -> Image.Image:
    path = Path(path)
    with Image.open(path) as source:
        if source.mode != "RGB" or len(source.getbands()) != 3:
            raise ValueError(f"{path} 不是 RGB 三通道图像（mode={source.mode}, channels={len(source.getbands())}）")
        return ImageOps.exif_transpose(source).convert("RGB")


def validate_triplet(lq_path: Path | str, gt_path: Path | str, output_path: Path | str) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Load and validate three images; mismatched dimensions raise immediately."""

    lq = _load_rgb(lq_path)
    gt = _load_rgb(gt_path)
    output = _load_rgb(output_path)
    sizes = {"LQ": lq.size, "GT": gt.size, "Output": output.size}
    if len(set(sizes.values())) != 1:
        details = ", ".join(f"{name}={size[0]}x{size[1]}" for name, size in sizes.items())
        raise ValueError(f"尺寸不一致，拒绝静默 resize：{details}")
    return lq, gt, output


def _array(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.uint8)


def _tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    return pil_to_tensor(image).unsqueeze(0).to(device=device, dtype=torch.float32).div(127.5).sub(1.0)


def _conclusion(delta: float, *, lower_is_better: bool = False) -> str:
    if math.isclose(delta, 0.0, abs_tol=1e-12):
        return "Unchanged"
    improved = delta < 0 if lower_is_better else delta > 0
    return "Improved" if improved else "Degraded"


def _lpips_value(model: lpips.LPIPS, first: Image.Image, second: Image.Image, device: torch.device) -> float:
    with torch.inference_mode():
        value = model(_tensor(first, device), _tensor(second, device))
    return float(value.item())


def _metric_row(case: EvaluationCase, lq: Image.Image, gt: Image.Image, output: Image.Image, model: lpips.LPIPS, device: torch.device) -> MetricRow:
    lq_array = _array(lq)
    gt_array = _array(gt)
    output_array = _array(output)
    lq_psnr = float(peak_signal_noise_ratio(gt_array, lq_array, data_range=255))
    output_psnr = float(peak_signal_noise_ratio(gt_array, output_array, data_range=255))
    lq_ssim = float(structural_similarity(gt_array, lq_array, channel_axis=2, data_range=255))
    output_ssim = float(structural_similarity(gt_array, output_array, channel_axis=2, data_range=255))
    lq_lpips = _lpips_value(model, lq, gt, device)
    output_lpips = _lpips_value(model, output, gt, device)
    return MetricRow(
        case=case.case,
        lq_psnr=lq_psnr,
        output_psnr=output_psnr,
        delta_psnr=output_psnr - lq_psnr,
        psnr_conclusion=_conclusion(output_psnr - lq_psnr),
        lq_ssim=lq_ssim,
        output_ssim=output_ssim,
        delta_ssim=output_ssim - lq_ssim,
        ssim_conclusion=_conclusion(output_ssim - lq_ssim),
        lq_lpips=lq_lpips,
        output_lpips=output_lpips,
        delta_lpips=output_lpips - lq_lpips,
        lpips_conclusion=_conclusion(output_lpips - lq_lpips, lower_is_better=True),
        lq_path=case.lq_path,
        gt_path=case.gt_path,
        output_path=case.output_path,
    )


def _average(rows: list[MetricRow], csv_path: Path) -> MetricRow:
    mean = lambda values: float(np.mean(np.asarray(values, dtype=np.float64)))
    return MetricRow(
        case="Average",
        lq_psnr=mean([row.lq_psnr for row in rows]),
        output_psnr=mean([row.output_psnr for row in rows]),
        delta_psnr=mean([row.delta_psnr for row in rows]),
        psnr_conclusion=_conclusion(mean([row.delta_psnr for row in rows])),
        lq_ssim=mean([row.lq_ssim for row in rows]),
        output_ssim=mean([row.output_ssim for row in rows]),
        delta_ssim=mean([row.delta_ssim for row in rows]),
        ssim_conclusion=_conclusion(mean([row.delta_ssim for row in rows])),
        lq_lpips=mean([row.lq_lpips for row in rows]),
        output_lpips=mean([row.output_lpips for row in rows]),
        delta_lpips=mean([row.delta_lpips for row in rows]),
        lpips_conclusion=_conclusion(mean([row.delta_lpips for row in rows]), lower_is_better=True),
        lq_path=Path(""),
        gt_path=Path(""),
        output_path=Path(""),
    )


CSV_FIELDS = [
    "Case", "LQ PSNR (dB)", "Output PSNR (dB)", "Delta PSNR (dB)", "PSNR Conclusion",
    "LQ SSIM", "Output SSIM", "Delta SSIM", "SSIM Conclusion",
    "LQ LPIPS-Alex", "Output LPIPS-Alex", "Delta LPIPS", "LPIPS Conclusion",
    "LQ Path", "GT Path", "Output Path",
]


def _row_dict(row: MetricRow) -> dict[str, object]:
    return {
        "Case": row.case,
        "LQ PSNR (dB)": f"{row.lq_psnr:.6f}",
        "Output PSNR (dB)": f"{row.output_psnr:.6f}",
        "Delta PSNR (dB)": f"{row.delta_psnr:.6f}",
        "PSNR Conclusion": row.psnr_conclusion,
        "LQ SSIM": f"{row.lq_ssim:.6f}",
        "Output SSIM": f"{row.output_ssim:.6f}",
        "Delta SSIM": f"{row.delta_ssim:.6f}",
        "SSIM Conclusion": row.ssim_conclusion,
        "LQ LPIPS-Alex": f"{row.lq_lpips:.6f}",
        "Output LPIPS-Alex": f"{row.output_lpips:.6f}",
        "Delta LPIPS": f"{row.delta_lpips:.6f}",
        "LPIPS Conclusion": row.lpips_conclusion,
        "LQ Path": str(row.lq_path),
        "GT Path": str(row.gt_path),
        "Output Path": str(row.output_path),
    }


def write_csv(rows: list[MetricRow], average: MetricRow, csv_path: Path | str) -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_row_dict(row) for row in [*rows, average])
    return csv_path


def evaluate_dataset(
    lq_dir: Path | str,
    gt_dir: Path | str,
    output_dir: Path | str,
    csv_path: Path | str,
    *,
    device: str = "auto",
) -> EvaluationSummary:
    """Evaluate all matched cases and write a per-case plus average CSV."""

    cases = match_evaluation_cases(lq_dir, gt_dir, output_dir)
    device_name = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    metric_device = torch.device(device_name)
    model = lpips.LPIPS(net="alex", verbose=False).to(metric_device).eval()
    rows: list[MetricRow] = []
    for case in cases:
        lq, gt, output = validate_triplet(case.lq_path, case.gt_path, case.output_path)
        rows.append(_metric_row(case, lq, gt, output, model, metric_device))
    average = _average(rows, Path(csv_path))
    csv_path = write_csv(rows, average, csv_path)
    return EvaluationSummary(rows=rows, average=average, csv_path=csv_path)


def _build_parser() -> argparse.ArgumentParser:
    project_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lq-dir", type=Path, default=project_dir / "baseline" / "input")
    parser.add_argument("--gt-dir", type=Path, default=project_dir / "csig_dataset" / "验证集")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "baseline" / "output" / "result")
    parser.add_argument("--csv", type=Path, default=project_dir / "baseline" / "evaluation_metrics.csv")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser


def _print_summary(summary: EvaluationSummary) -> None:
    print("Case | LQ PSNR | Output PSNR | Delta PSNR | LQ SSIM | Output SSIM | Delta SSIM | LQ LPIPS | Output LPIPS | Delta LPIPS")
    for row in [*summary.rows, summary.average]:
        print(
            f"{row.case} | {row.lq_psnr:.4f} | {row.output_psnr:.4f} | {row.delta_psnr:+.4f} | "
            f"{row.lq_ssim:.5f} | {row.output_ssim:.5f} | {row.delta_ssim:+.5f} | "
            f"{row.lq_lpips:.5f} | {row.output_lpips:.5f} | {row.delta_lpips:+.5f}"
        )
        if row.case != "Average":
            print(f"  Conclusions: PSNR={row.psnr_conclusion}, SSIM={row.ssim_conclusion}, LPIPS={row.lpips_conclusion}")
    print(f"CSV written to {summary.csv_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = evaluate_dataset(args.lq_dir, args.gt_dir, args.output_dir, args.csv, device=args.device)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
