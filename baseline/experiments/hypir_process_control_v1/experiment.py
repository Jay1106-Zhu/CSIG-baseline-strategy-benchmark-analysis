"""Offline process-control evaluation on existing coeff_t PNGs.

Does not call HYPIR, does not train, does not overwrite fusion_v1 / fusion_v2.
Default: validation case1-case5, coeff_t in {50,75,100,150,200}.
"""
from __future__ import annotations

import argparse
import csv
import math
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baseline.experiments.hypir_fusion_v1.experiment import (
    load_rgb,
    save_rgb,
    _case_file,
    _case_names,
    _lpips_scores,
    _metric,
    _panel,
    _resolve_dir,
)
from baseline.experiments.hypir_process_control_v1.process_control import (
    EXISTING_COEFF_T_LEVELS,
    PARAMETER_MAP,
    SELECTED_CONTROLS,
    coeff_t_eps_scale,
    default_coeff_dirs,
)


# Unified 256 crops. case4/case3 boxes match error_decomposition_v1 / fusion_v1.
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
    "case5": [("clock", 1920, 1280, 2176, 1536)],
}


def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}f}"


def _write_summary(
    out: Path,
    metric_rows: list[dict[str, object]],
    crop_rows: list[dict[str, object]],
    lpips_enabled: bool,
) -> None:
    def avg(method: str, field: str) -> float:
        values = [
            float(row[field])
            for row in metric_rows
            if row["method"] == method and row["case"] != "Average" and row[field] not in ("", None)
        ]
        return float(np.mean(values)) if values else float("nan")

    methods = ["LQ"]
    methods.extend(f"coeff_t_{value}" for value in EXISTING_COEFF_T_LEVELS)
    for extra in ("texture_selective_h200", "fusion_v1_A"):
        if any(row["method"] == extra for row in metric_rows):
            methods.append(extra)

    lines = [
        "# HYPIR process control v1 summary",
        "",
        "Offline. Single-step HYPIR. Selected control: `coeff_t` (already inferred).",
        "No HYPIR re-inference, no fusion overwrite, no LoRA/Adapter/ControlNet.",
        "",
        f"- date: {date.today().isoformat()}",
        f"- LPIPS: Alex max-side 1024, computed={lpips_enabled}",
        f"- selected: {', '.join(SELECTED_CONTROLS)}",
        f"- levels: {', '.join(str(v) for v in EXISTING_COEFF_T_LEVELS)}",
        f"- eps scales: " + ", ".join(f"{v}={coeff_t_eps_scale(v):.4f}" for v in EXISTING_COEFF_T_LEVELS),
        "",
        "## Average metrics",
        "",
        "| method | PSNR | SSIM | LPIPS_1024 | vs texture 28.48 |",
        "|---|---:|---:|---:|---|",
    ]
    for method in methods:
        psnr = avg(method, "PSNR")
        ssim = avg(method, "SSIM")
        lp = avg(method, "LPIPS_Alex")
        lp_s = "" if math.isnan(lp) else f"{lp:.6f}"
        vs = "" if math.isnan(psnr) else ("YES" if psnr >= 28.48 - 1e-9 and method != "texture_selective_h200" else "no")
        if method in ("LQ", "texture_selective_h200"):
            vs = "anchor" if method == "texture_selective_h200" else "ref"
        lines.append(f"| {method} | {psnr:.6f} | {ssim:.6f} | {lp_s} | {vs} |")

    lines.extend(["", "## Per-case PSNR", ""])
    cases = sorted({row["case"] for row in metric_rows if row["case"] != "Average"})
    header = "| case | LQ | " + " | ".join(f"t={v}" for v in EXISTING_COEFF_T_LEVELS) + " | texture | fusion_A |"
    lines.append(header)
    lines.append("|" + "---|" * (header.count("|") - 1))

    def psnr(case: str, method: str) -> str:
        row = next((item for item in metric_rows if item["case"] == case and item["method"] == method), None)
        return "" if row is None else f"{float(row['PSNR']):.4f}"

    for case in cases:
        cells = [case, psnr(case, "LQ")]
        cells.extend(psnr(case, f"coeff_t_{value}") for value in EXISTING_COEFF_T_LEVELS)
        cells.append(psnr(case, "texture_selective_h200"))
        cells.append(psnr(case, "fusion_v1_A"))
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## Diagnostic crops (PSNR vs GT)", "", "| case | crop | method | PSNR | SSIM |", "|---|---|---|---:|---:|"])
    for row in crop_rows:
        lines.append(
            f"| {row['case']} | {row['crop']} | {row['method']} | {float(row['PSNR']):.4f} | {float(row['SSIM']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Code facts",
            "",
            "- Inference is one UNet call. Not a diffusion sampler.",
            "- `coeff_t` scales x0 conversion. `model_t` is only the UNet timestep embedding.",
            "- No CFG, no LoRA scale, no noise injection, no start/end schedule.",
            "",
        ]
    )
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    lq_dir: Path | str,
    gt_dir: Path | str,
    coeff_dirs: Mapping[int, Path | str],
    out_dir: Path | str,
    *,
    texture_dir: Path | str | None = None,
    fusion_v1_dir: Path | str | None = None,
    cases: Sequence[str] | None = None,
    compute_lpips: bool = False,
    lpips_size: int = 1024,
    allow_inference: bool = False,
) -> dict[str, object]:
    if allow_inference:
        raise RuntimeError(
            "Process-control v1 refuses HYPIR re-inference. "
            "Reuse existing coeff_t PNGs. Set allow_inference only if you intend to break this rule."
        )
    lq_dir, gt_dir, out = Path(lq_dir), Path(gt_dir), Path(out_dir)
    texture_path = Path(texture_dir) if texture_dir is not None else None
    fusion_path = Path(fusion_v1_dir) if fusion_v1_dir is not None else None
    coeff_paths = {int(value): Path(path) for value, path in coeff_dirs.items()}
    missing_levels = [value for value in EXISTING_COEFF_T_LEVELS if value not in coeff_paths]
    if missing_levels:
        raise FileNotFoundError(f"Missing coeff_t directories for levels {missing_levels}")
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("comparison", "crops"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    available = _case_names(lq_dir)
    if not available:
        raise FileNotFoundError(f"No LQ images found in {lq_dir}")
    if cases is None:
        selected = [name for name in ("case1", "case2", "case3", "case4", "case5") if name in available]
        if not selected:
            selected = available
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
        gt = load_rgb(_case_file(gt_dir, case, ("_gt",)))
        if gt.shape != lq.shape:
            raise ValueError(f"{case}: GT dimensions do not match LQ")
        outputs: dict[str, np.ndarray] = {"LQ": lq}
        for value in EXISTING_COEFF_T_LEVELS:
            outputs[f"coeff_t_{value}"] = load_rgb(_case_file(coeff_paths[value], case, ("_lq",)))
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

        panel = [("LQ", lq)]
        for value in EXISTING_COEFF_T_LEVELS:
            panel.append((f"t={value}", outputs[f"coeff_t_{value}"]))
        if "texture_selective_h200" in outputs:
            panel.append(("texture", outputs["texture_selective_h200"]))
        if "fusion_v1_A" in outputs:
            panel.append(("fusion_A", outputs["fusion_v1_A"]))
        panel.append(("GT", gt))
        _panel(out / "comparison" / f"{case}.png", panel, max_width=768)

        for crop_name, x0, y0, x1, y1 in DIAGNOSTIC_CROPS.get(case, ()):
            if y1 > lq.shape[0] or x1 > lq.shape[1]:
                continue
            crop_panel = [(name, image[y0:y1, x0:x1]) for name, image in panel]
            _panel(out / "crops" / f"{case}_{crop_name}.png", crop_panel, max_width=256)
            gt_crop = gt[y0:y1, x0:x1]
            for name, image in panel:
                if name == "GT":
                    continue
                psnr, ssim = _metric(image[y0:y1, x0:x1], gt_crop)
                crop_rows.append(
                    {
                        "case": case,
                        "crop": crop_name,
                        "method": name,
                        "PSNR": f"{psnr:.6f}",
                        "SSIM": f"{ssim:.6f}",
                    }
                )

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

    fields = [
        "case", "method", "PSNR", "SSIM", "LPIPS_Alex",
        "Delta_PSNR_vs_LQ", "Delta_SSIM_vs_LQ", "Delta_LPIPS_vs_LQ",
    ]
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

    map_lines = ["# Parameter map (from this repo's HYPIR code)", ""]
    for name, info in PARAMETER_MAP.items():
        map_lines.append(f"## {name}")
        map_lines.append("")
        for key, value in info.items():
            map_lines.append(f"- {key}: {value}")
        map_lines.append("")
    (out / "parameter_map.md").write_text("\n".join(map_lines), encoding="utf-8")

    (out / "experiment_metadata.md").write_text(
        "\n".join(
            [
                "# HYPIR process control v1 metadata",
                "",
                f"- date: {date.today().isoformat()}",
                f"- lq_dir: `{lq_dir}`",
                f"- gt_dir: `{gt_dir}`",
                f"- coeff_dirs: { {k: str(v) for k, v in coeff_paths.items()} }",
                f"- cases: {', '.join(selected)}",
                "- inference: none (allow_inference rejected)",
                f"- selected_controls: {SELECTED_CONTROLS}",
                f"- lpips: Alex max-side {lpips_size}, enabled={compute_lpips}",
                "- does not write fusion_v1 or fusion_v2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_summary(out, metric_rows, crop_rows, compute_lpips)
    return {
        "cases": selected,
        "metrics": metric_rows,
        "crop_metrics": crop_rows,
        "output_dir": out,
        "ran_inference": False,
    }


def main(argv: list[str] | None = None) -> int:
    experiment_dir = Path(__file__).resolve().parent
    project = experiment_dir.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project)
    parser.add_argument("--lq-dir", type=Path, default=None)
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
    gt_dir = args.gt_dir or root / "csig_dataset" / "验证集"
    texture_default = (
        root / "baseline" / "experiments" / "structure_local_restoration_v1" / "fusion" / "texture_selective" / "h200"
    )
    texture_dir = args.texture_dir or (texture_default if texture_default.is_dir() else None)
    fusion_default = (
        root / "baseline" / "experiments" / "hypir_fusion_v1" / "results" / "fusion_v1" / "fusion" / "A"
    )
    fusion_v1_dir = args.fusion_v1_dir or (fusion_default if fusion_default.is_dir() else None)
    out_dir = args.output_dir or (experiment_dir / "results" / "process_control_v1")
    cases = [item.strip() for item in args.cases.split(",") if item.strip()]
    result = run_experiment(
        lq_dir,
        gt_dir,
        default_coeff_dirs(root),
        out_dir,
        texture_dir=texture_dir,
        fusion_v1_dir=fusion_v1_dir,
        cases=cases,
        compute_lpips=args.compute_lpips,
        lpips_size=args.lpips_size,
        allow_inference=False,
    )
    print(f"Wrote {len(result['cases'])} cases to {result['output_dir']} (inference={result['ran_inference']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
