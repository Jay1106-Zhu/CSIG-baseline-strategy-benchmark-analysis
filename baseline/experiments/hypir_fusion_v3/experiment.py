"""Run Laplacian multi-band fusion on existing LQ / H50 / H200 PNGs.

Does not call HYPIR, does not overwrite fusion_v1 / fusion_v2 / texture_selective.
"""
from __future__ import annotations

import argparse
import csv
import math
from datetime import date
from pathlib import Path
from typing import Sequence
import sys

import cv2
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baseline.experiments.hypir_fusion_v1.experiment import (
    load_rgb,
    save_rgb,
    _case_file,
    _case_names,
    _colorize,
    _lpips_scores,
    _metric,
    _panel,
    _resolve_dir,
)
from baseline.experiments.hypir_fusion_v1.fusion import rgb_to_y
from baseline.experiments.hypir_fusion_v3.fusion import (
    BAND_SCALES,
    PYRAMID_LEVELS,
    band_gray,
    blend_rgb,
    decompose_y,
    fuse_multiband,
)


DIAGNOSTIC_CROPS: dict[str, list[tuple[str, int, int, int, int]]] = {
    "case1": [("text", 1024, 512, 1280, 768)],
    "case2": [("spine", 2048, 1024, 2304, 1280)],
    "case3": [
        ("bird", 2048, 1280, 2304, 1536),
        ("water", 1536, 256, 1792, 512),
        ("mud", 512, 512, 768, 768),
    ],
    "case4": [
        ("fish", 2816, 1024, 3072, 1280),
        ("leaf", 3072, 768, 3328, 1024),
        ("yellow_flower", 2560, 1792, 2816, 2048),
    ],
    "case5": [("clock", 1920, 1024, 2176, 1280)],
}

VARIANTS = (
    ("B", "multi_B", 0.0, "LQ", "LQ", "H200"),
    ("C", "multi_C", 0.0, "LQ", "H50", "H200"),
    ("D01", "multi_D01", 0.1, "LQ", "H50+0.1*(H200-H50)", "H200"),
    ("D02", "multi_D02", 0.2, "LQ", "H50+0.2*(H200-H50)", "H200"),
)


def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}f}"


def _write_summary(out: Path, metric_rows: list[dict[str, object]], crop_rows: list[dict[str, object]], lpips_enabled: bool) -> None:
    def avg(method: str, field: str) -> float:
        values = [
            float(row[field])
            for row in metric_rows
            if row["method"] == method and row["case"] != "Average" and row[field] not in ("", None)
        ]
        return float(np.mean(values)) if values else float("nan")

    methods = ["LQ", "HYPIR-50", "HYPIR-200", "texture_selective_h200", "fusion_v1_A", "multi_B", "multi_C", "multi_D01", "multi_D02"]
    lines = [
        "# HYPIR fusion v3 summary",
        "",
        f"- date: {date.today().isoformat()}",
        f"- pyramid levels: {PYRAMID_LEVELS}",
        f"- high: {BAND_SCALES['high']}",
        f"- mid: {BAND_SCALES['mid']}",
        f"- low: {BAND_SCALES['low']}",
        f"- LPIPS: Alex max-side 1024, computed={lpips_enabled}",
        "",
        "## Average metrics",
        "",
        "| method | PSNR | SSIM | LPIPS_1024 | vs 28.48 |",
        "|---|---:|---:|---:|---|",
    ]
    for method in methods:
        if not any(row["method"] == method for row in metric_rows):
            continue
        psnr = avg(method, "PSNR")
        ssim = avg(method, "SSIM")
        lp = avg(method, "LPIPS_Alex")
        lp_s = "" if math.isnan(lp) else f"{lp:.6f}"
        if method == "texture_selective_h200":
            mark = "anchor"
        elif method in ("LQ", "HYPIR-50", "HYPIR-200", "fusion_v1_A"):
            mark = "ref"
        else:
            mark = "YES" if (not math.isnan(psnr) and psnr >= 28.53) else "no"
        lines.append(f"| {method} | {psnr:.6f} | {ssim:.6f} | {lp_s} | {mark} |")
    lines.extend(["", "## Per-case PSNR", ""])
    cases = sorted({row["case"] for row in metric_rows if row["case"] != "Average"})
    header_methods = ["LQ", "HYPIR-200", "multi_B", "multi_C", "multi_D01", "multi_D02", "texture_selective_h200"]
    lines.append("| case | " + " | ".join(header_methods) + " |")
    lines.append("|" + "---|" * (len(header_methods) + 1))

    def psnr(case: str, method: str) -> str:
        row = next((item for item in metric_rows if item["case"] == case and item["method"] == method), None)
        return "" if row is None else f"{float(row['PSNR']):.4f}"

    for case in cases:
        lines.append("| " + " | ".join([case] + [psnr(case, method) for method in header_methods]) + " |")
    lines.extend(["", "## Crops", "", "| case | crop | method | PSNR | SSIM |", "|---|---|---|---:|---:|"])
    for row in crop_rows:
        lines.append(f"| {row['case']} | {row['crop']} | {row['method']} | {float(row['PSNR']):.4f} | {float(row['SSIM']):.4f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    lq_dir: Path | str,
    h50_dir: Path | str,
    h200_dir: Path | str,
    out_dir: Path | str,
    *,
    gt_dir: Path | str | None = None,
    texture_dir: Path | str | None = None,
    fusion_v1_dir: Path | str | None = None,
    cases: Sequence[str] | None = None,
    compute_lpips: bool = False,
    lpips_size: int = 1024,
) -> dict[str, object]:
    lq_dir, h50_dir, h200_dir, out = map(Path, (lq_dir, h50_dir, h200_dir, out_dir))
    gt_path = Path(gt_dir) if gt_dir is not None else None
    texture_path = Path(texture_dir) if texture_dir is not None else None
    fusion_path = Path(fusion_v1_dir) if fusion_v1_dir is not None else None
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("fusion/B", "fusion/C", "fusion/D01", "fusion/D02", "bands", "comparison", "crops"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    available = _case_names(lq_dir)
    if not available:
        raise FileNotFoundError(f"No LQ images found in {lq_dir}")
    if cases is None:
        selected = [name for name in ("case1", "case2", "case3", "case4", "case5") if name in available] or available
    else:
        selected = [name.casefold() for name in cases]
        missing = [name for name in selected if name not in available]
        if missing:
            raise FileNotFoundError(f"Missing LQ cases in {lq_dir}: {missing}")

    lpips_model = lpips_device = None
    if compute_lpips:
        import torch
        import lpips

        lpips_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_model = lpips.LPIPS(net="alex", verbose=False).to(lpips_device).eval()

    metric_rows: list[dict[str, object]] = []
    crop_rows: list[dict[str, object]] = []
    band_rows: list[dict[str, object]] = []
    method_psnr: dict[str, list[float]] = {}
    method_ssim: dict[str, list[float]] = {}
    method_lp: dict[str, list[float]] = {}

    def record(method: str, psnr: float, ssim: float, lp: float) -> None:
        method_psnr.setdefault(method, []).append(psnr)
        method_ssim.setdefault(method, []).append(ssim)
        if not math.isnan(lp):
            method_lp.setdefault(method, []).append(lp)

    for case in selected:
        lq = load_rgb(_case_file(lq_dir, case, ("_lq", "_input")))
        h50 = load_rgb(_case_file(h50_dir, case, ("_lq",)))
        h200 = load_rgb(_case_file(h200_dir, case, ("_lq",)))
        if len({lq.shape, h50.shape, h200.shape}) != 1:
            raise ValueError(f"{case}: LQ/H50/H200 dimensions do not match")
        gt = None
        if gt_path is not None and gt_path.is_dir():
            try:
                gt = load_rgb(_case_file(gt_path, case, ("_gt",)))
            except FileNotFoundError:
                gt = None
            if gt is not None and gt.shape != lq.shape:
                raise ValueError(f"{case}: GT dimensions do not match LQ")

        outputs: dict[str, np.ndarray] = {"LQ": lq, "HYPIR-50": h50, "HYPIR-200": h200}
        if texture_path is not None and texture_path.is_dir():
            try:
                outputs["texture_selective_h200"] = load_rgb(_case_file(texture_path, case, ("_lq",)))
            except FileNotFoundError:
                pass
        if fusion_path is not None and fusion_path.is_dir():
            try:
                outputs["fusion_v1_A"] = load_rgb(_case_file(fusion_path, case, ("_lq",)))
            except FileNotFoundError:
                pass

        mid_d01 = blend_rgb(h50, h200, 0.1)
        mid_d02 = blend_rgb(h50, h200, 0.2)
        fused = {
            "B": fuse_multiband(lq, lq, h200, lq),
            "C": fuse_multiband(lq, h50, h200, lq),
            "D01": fuse_multiband(lq, mid_d01, h200, lq),
            "D02": fuse_multiband(lq, mid_d02, h200, lq),
        }
        for folder, method in (("B", "multi_B"), ("C", "multi_C"), ("D01", "multi_D01"), ("D02", "multi_D02")):
            save_rgb(out / "fusion" / folder / f"{case}.png", fused[folder])
            outputs[method] = fused[folder]

        lq_bands = decompose_y(rgb_to_y(lq))
        h50_bands = decompose_y(rgb_to_y(h50))
        h200_bands = decompose_y(rgb_to_y(h200))
        gt_bands = decompose_y(rgb_to_y(gt)) if gt is not None else None
        for source, bands in (("LQ", lq_bands), ("H50", h50_bands), ("H200", h200_bands)):
            save_rgb(out / "bands" / f"{case}_{source}_low.png", band_gray(bands["low"], residual=False))
            save_rgb(out / "bands" / f"{case}_{source}_mid.png", band_gray(bands["mid"], residual=True))
            save_rgb(out / "bands" / f"{case}_{source}_high.png", band_gray(bands["high"], residual=True))
        save_rgb(
            out / "bands" / f"{case}_H200_minus_LQ_mid.png",
            _colorize(np.abs(h200_bands["mid"] - lq_bands["mid"]), colormap=cv2.COLORMAP_INFERNO),
        )
        save_rgb(
            out / "bands" / f"{case}_H200_minus_LQ_high.png",
            _colorize(np.abs(h200_bands["high"] - lq_bands["high"]), colormap=cv2.COLORMAP_INFERNO),
        )
        band_panel = [
            ("LQ_low", band_gray(lq_bands["low"], residual=False)),
            ("LQ_mid", band_gray(lq_bands["mid"], residual=True)),
            ("LQ_high", band_gray(lq_bands["high"], residual=True)),
            ("H200_low", band_gray(h200_bands["low"], residual=False)),
            ("H200_mid", band_gray(h200_bands["mid"], residual=True)),
            ("H200_high", band_gray(h200_bands["high"], residual=True)),
            ("dMid", _colorize(np.abs(h200_bands["mid"] - lq_bands["mid"]), colormap=cv2.COLORMAP_INFERNO)),
            ("dHigh", _colorize(np.abs(h200_bands["high"] - lq_bands["high"]), colormap=cv2.COLORMAP_INFERNO)),
        ]
        _panel(out / "comparison" / f"{case}_bands.png", band_panel, max_width=512)

        panel = [("LQ", lq), ("H50", h50), ("H200", h200), ("B", fused["B"]), ("C", fused["C"]), ("D01", fused["D01"]), ("D02", fused["D02"])]
        if "texture_selective_h200" in outputs:
            panel.append(("texture", outputs["texture_selective_h200"]))
        if "fusion_v1_A" in outputs:
            panel.append(("fusion_A", outputs["fusion_v1_A"]))
        if gt is not None:
            panel.append(("GT", gt))
        _panel(out / "comparison" / f"{case}.png", panel, max_width=768)

        named = {name: image for name, image in panel}
        for crop_name, x0, y0, x1, y1 in DIAGNOSTIC_CROPS.get(case, ()):
            if y1 > lq.shape[0] or x1 > lq.shape[1]:
                continue
            crop_panel = [(name, image[y0:y1, x0:x1]) for name, image in panel]
            _panel(out / "crops" / f"{case}_{crop_name}.png", crop_panel, max_width=256)
            band_crop = [
                ("LQ_mid", band_gray(lq_bands["mid"][y0:y1, x0:x1], residual=True)),
                ("LQ_high", band_gray(lq_bands["high"][y0:y1, x0:x1], residual=True)),
                ("H200_mid", band_gray(h200_bands["mid"][y0:y1, x0:x1], residual=True)),
                ("H200_high", band_gray(h200_bands["high"][y0:y1, x0:x1], residual=True)),
                ("dMid", _colorize(np.abs(h200_bands["mid"][y0:y1, x0:x1] - lq_bands["mid"][y0:y1, x0:x1]), colormap=cv2.COLORMAP_INFERNO)),
                ("dHigh", _colorize(np.abs(h200_bands["high"][y0:y1, x0:x1] - lq_bands["high"][y0:y1, x0:x1]), colormap=cv2.COLORMAP_INFERNO)),
            ]
            _panel(out / "crops" / f"{case}_{crop_name}_bands.png", band_crop, max_width=256)
            band_rows.append(
                {
                    "case": case,
                    "crop": crop_name,
                    "mean_abs_dLow": f"{float(np.abs(h200_bands['low'][y0:y1, x0:x1] - lq_bands['low'][y0:y1, x0:x1]).mean()):.6f}",
                    "mean_abs_dMid": f"{float(np.abs(h200_bands['mid'][y0:y1, x0:x1] - lq_bands['mid'][y0:y1, x0:x1]).mean()):.6f}",
                    "mean_abs_dHigh": f"{float(np.abs(h200_bands['high'][y0:y1, x0:x1] - lq_bands['high'][y0:y1, x0:x1]).mean()):.6f}",
                }
            )
            if gt is not None:
                gt_crop = gt[y0:y1, x0:x1]
                for name, image in panel:
                    if name == "GT":
                        continue
                    psnr, ssim = _metric(image[y0:y1, x0:x1], gt_crop)
                    crop_rows.append({"case": case, "crop": crop_name, "method": name, "PSNR": f"{psnr:.6f}", "SSIM": f"{ssim:.6f}"})

        if gt is None:
            continue
        lp_values = {} if lpips_model is None else _lpips_scores(outputs, gt, lpips_model, lpips_device, lpips_size)
        lq_psnr, lq_ssim = _metric(lq, gt)
        lq_lp = float("nan") if lpips_model is None else lp_values["LQ"]
        for method, image in outputs.items():
            psnr, ssim = _metric(image, gt)
            lp = float("nan") if lpips_model is None else lp_values[method]
            metric_rows.append(
                {
                    "case": case,
                    "method": method,
                    "PSNR": f"{psnr:.6f}",
                    "SSIM": f"{ssim:.6f}",
                    "LPIPS_Alex": _fmt(lp),
                    "Delta_PSNR_vs_LQ": f"{psnr - lq_psnr:.6f}",
                    "Delta_SSIM_vs_LQ": f"{ssim - lq_ssim:.6f}",
                    "Delta_LPIPS_vs_LQ": "" if math.isnan(lp) else f"{lp - lq_lp:.6f}",
                }
            )
            record(method, psnr, ssim, lp)

    fields = ["case", "method", "PSNR", "SSIM", "LPIPS_Alex", "Delta_PSNR_vs_LQ", "Delta_SSIM_vs_LQ", "Delta_LPIPS_vs_LQ"]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)
        if metric_rows:
            lq_psnr = float(np.mean(method_psnr["LQ"]))
            lq_ssim = float(np.mean(method_ssim["LQ"]))
            lq_lp = float(np.mean(method_lp["LQ"])) if method_lp.get("LQ") else None
            for method in list(method_psnr):
                psnr = float(np.mean(method_psnr[method]))
                ssim = float(np.mean(method_ssim[method]))
                lp = float(np.mean(method_lp[method])) if method_lp.get(method) else None
                writer.writerow(
                    {
                        "case": "Average",
                        "method": method,
                        "PSNR": f"{psnr:.6f}",
                        "SSIM": f"{ssim:.6f}",
                        "LPIPS_Alex": "" if lp is None else f"{lp:.6f}",
                        "Delta_PSNR_vs_LQ": f"{psnr - lq_psnr:.6f}",
                        "Delta_SSIM_vs_LQ": f"{ssim - lq_ssim:.6f}",
                        "Delta_LPIPS_vs_LQ": "" if lp is None or lq_lp is None else f"{lp - lq_lp:.6f}",
                    }
                )
    with (out / "crop_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "crop", "method", "PSNR", "SSIM"])
        writer.writeheader()
        writer.writerows(crop_rows)
    with (out / "band_energy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "crop", "mean_abs_dLow", "mean_abs_dMid", "mean_abs_dHigh"])
        writer.writeheader()
        writer.writerows(band_rows)
    (out / "experiment_metadata.md").write_text(
        "\n".join(
            [
                "# HYPIR fusion v3 metadata",
                "",
                f"- date: {date.today().isoformat()}",
                f"- lq_dir: `{lq_dir}`",
                f"- h50_dir: `{h50_dir}`",
                f"- h200_dir: `{h200_dir}`",
                f"- cases: {', '.join(selected)}",
                f"- pyramid_levels: {PYRAMID_LEVELS}",
                *[f"- {name}: {text}" for name, text in BAND_SCALES.items()],
                "- chroma: LQ YCrCb Cr/Cb",
                "- inference: none",
                "- does not write fusion_v1 / fusion_v2 / texture_selective",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_summary(out, metric_rows, crop_rows, compute_lpips)
    return {"cases": selected, "metrics": metric_rows, "output_dir": out}


def main(argv: list[str] | None = None) -> int:
    experiment_dir = Path(__file__).resolve().parent
    project = experiment_dir.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project)
    parser.add_argument("--lq-dir", type=Path, default=None)
    parser.add_argument("--h50-dir", type=Path, default=None)
    parser.add_argument("--h200-dir", type=Path, default=None)
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument("--texture-dir", type=Path, default=None)
    parser.add_argument("--fusion-v1-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cases", default="case1,case2,case3,case4,case5")
    parser.add_argument("--lpips", dest="compute_lpips", action="store_true", default=True)
    parser.add_argument("--no-lpips", dest="compute_lpips", action="store_false")
    parser.add_argument("--lpips-size", type=int, default=1024)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    lq_dir = _resolve_dir(args.lq_dir, experiment_dir / "data" / "LQ", root / "baseline" / "input")
    h50_dir = _resolve_dir(
        args.h50_dir,
        experiment_dir / "data" / "H50",
        root / "baseline" / "experiments" / "coeff_t_50" / "output" / "result",
    )
    h200_dir = _resolve_dir(
        args.h200_dir,
        experiment_dir / "data" / "H200",
        root / "baseline" / "experiments" / "coeff_t_200" / "output" / "result",
    )
    gt_dir = args.gt_dir or root / "csig_dataset" / "验证集"
    texture_default = root / "baseline" / "experiments" / "structure_local_restoration_v1" / "fusion" / "texture_selective" / "h200"
    texture_dir = args.texture_dir or (texture_default if texture_default.is_dir() else None)
    fusion_default = root / "baseline" / "experiments" / "hypir_fusion_v1" / "results" / "fusion_v1" / "fusion" / "A"
    fusion_v1_dir = args.fusion_v1_dir or (fusion_default if fusion_default.is_dir() else None)
    out_dir = args.output_dir or (experiment_dir / "results" / "fusion_v3")
    cases = [item.strip() for item in args.cases.split(",") if item.strip()]
    result = run_experiment(
        lq_dir,
        h50_dir,
        h200_dir,
        out_dir,
        gt_dir=gt_dir,
        texture_dir=texture_dir,
        fusion_v1_dir=fusion_v1_dir,
        cases=cases,
        compute_lpips=args.compute_lpips,
        lpips_size=args.lpips_size,
    )
    print(f"Wrote {len(result['cases'])} cases to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
