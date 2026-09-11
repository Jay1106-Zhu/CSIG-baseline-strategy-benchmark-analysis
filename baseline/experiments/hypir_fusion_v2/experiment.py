"""Run HYPIR residual-confidence fusion (F1) on existing LQ / H50 / H200 images.

No diffusion inference, no HYPIR source edits, no new models.
Does not overwrite hypir_fusion_v1. Default scope is validation case1-case5.
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
    DIAGNOSTIC_CROPS,
    load_rgb,
    save_rgb,
    _case_file,
    _case_names,
    _colorize,
    _lpips_scores,
    _metric,
    _panel,
    _residual_l1,
    _resolve_dir,
)
from baseline.experiments.hypir_fusion_v1.fusion import (
    DEFAULT_BLUR_SIGMA,
    DEFAULT_SCENE_ALPHA,
    SCENE_NOTES,
    compute_structure_mask,
    fuse_scheme_a,
    fuse_scheme_b,
    rgb_to_y,
    scene_alpha_for_case,
)
from baseline.experiments.hypir_fusion_v2.fusion import (
    DEFAULT_HIGH_PERCENTILE,
    DEFAULT_LOW_PERCENTILE,
    DEFAULT_RESIDUAL_BLUR_SIGMA,
    MASK_GROUPS,
    UNIQUE_MASK_VARIANTS,
    combine_masks,
    compute_residual_confidence,
)


BASE_METHODS = ("LQ", "HYPIR-50", "HYPIR-200")
FUSION_METHODS = tuple(
    name
    for variant in UNIQUE_MASK_VARIANTS
    for name in (variant["method_a"], variant["method_b"])
)


def _save_gray(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(np.asarray(values) * 255.0), 0, 255).astype(np.uint8)).save(path, format="PNG")


def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}f}"


def _write_summary(
    out: Path,
    metric_rows: list[dict[str, object]],
    case_info: list[dict[str, object]],
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

    methods = list(BASE_METHODS)
    methods.extend(FUSION_METHODS)
    if any(row["method"] == "texture_selective_h200" for row in metric_rows):
        methods.append("texture_selective_h200")

    header = "| method | group | formula | PSNR | SSIM | LPIPS_1024 |"
    sep = "|---|---|---|---:|---:|---:|"
    avg_lines = [header, sep]
    method_meta = {
        "LQ": ("ref", "unchanged LQ"),
        "HYPIR-50": ("ref", "coeff_t=50"),
        "HYPIR-200": ("ref", "coeff_t=200"),
        "texture_selective_h200": ("anchor", "current HYPIR-family anchor"),
        "fusion_A": ("A/B", "M_struct, base=LQ"),
        "fusion_B": ("A/B", "M_struct, base=H50"),
        "fusion_A_conf05": ("gamma=0.5", "M_struct * conf^0.5, base=LQ"),
        "fusion_B_conf05": ("gamma=0.5", "M_struct * conf^0.5, base=H50"),
        "fusion_A_conf": ("C", "M_struct * conf, base=LQ"),
        "fusion_B_conf": ("C", "M_struct * conf, base=H50"),
        "fusion_A_conf2": ("D", "M_struct * conf^2, base=LQ"),
        "fusion_B_conf2": ("D", "M_struct * conf^2, base=H50"),
    }
    for method in methods:
        group, formula = method_meta.get(method, ("", method))
        lp = avg(method, "LPIPS_Alex")
        lp_s = "" if math.isnan(lp) else f"{lp:.6f}"
        avg_lines.append(
            f"| {method} | {group} | {formula} | {avg(method, 'PSNR'):.6f} | {avg(method, 'SSIM'):.6f} | {lp_s} |"
        )

    case_lines = [
        "| case | scene | alpha | mean_struct | mean_conf | mean_final_conf | mean_final_conf2 | fusion_A | A_conf | A_conf2 | texture |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def psnr(case: str, method: str) -> str:
        row = next((item for item in metric_rows if item["case"] == case and item["method"] == method), None)
        return "" if row is None else f"{float(row['PSNR']):.4f}"

    for info in case_info:
        case = info["case"]
        case_lines.append(
            f"| {case} | {info['scene']} | {float(info['alpha']):.2f} | "
            f"{float(info['mean_struct']):.4f} | {float(info['mean_conf']):.4f} | "
            f"{float(info['mean_final_conf']):.4f} | {float(info['mean_final_conf2']):.4f} | "
            f"{psnr(case, 'fusion_A')} | {psnr(case, 'fusion_A_conf')} | {psnr(case, 'fusion_A_conf2')} | "
            f"{psnr(case, 'texture_selective_h200')} |"
        )

    crop_lines = [
        "| case | crop | method | PSNR | SSIM |",
        "|---|---|---|---:|---:|",
    ]
    for row in crop_rows:
        crop_lines.append(
            f"| {row['case']} | {row['crop']} | {row['method']} | "
            f"{float(row['PSNR']):.4f} | {float(row['SSIM']):.4f} |"
        )

    lines = [
        "# HYPIR fusion v2 summary",
        "",
        "Offline residual-confidence fusion (F1). H200 is a detail candidate;",
        "LQ/H50 are structure anchors; `conf = 1 - normalize(|H200-LQ|)` down-weights",
        "large residuals that the Sobel structure mask still allows.",
        "",
        f"- date: {date.today().isoformat()}",
        f"- LPIPS: Alex max-side 1024, computed={lpips_enabled}",
        "- scheme A: `Y = Y_LQ + α·M_final·(Y_H200 - Y_LQ)`",
        "- scheme B: `Y = Y_H50 + α·M_final·(Y_H200 - Y_H50)`",
        "- chroma: LQ Cb/Cr for both schemes",
        "- A/B mask: `M_final = M_struct` (original fusion_v1; A and B share this formula)",
        "- C mask: `M_final = M_struct * conf`",
        "- D mask: `M_final = M_struct * conf^2`",
        "- extra: `M_final = M_struct * conf^0.5`",
        "",
        "## Average metrics",
        "",
        *avg_lines,
        "",
        "## Per-case (scheme A / LQ base)",
        "",
        *case_lines,
        "",
        "## Diagnostic crops",
        "",
        *(crop_lines if crop_rows else ["(no diagnostic crops in range)"]),
        "",
        "## Visual checklist",
        "",
        "- case4 `crops/case4_mid04_fish.png`: does residual conf further flatten the fish-head vs fusion_v1?",
        "- case4 `crops/case4_high02_leaf.png`: real foliage vs invented serrated edges",
        "- case3 `crops/case3_low02_water.png`: invented water grain vs fusion_v1",
        "- case1/2/5 comparison panels: text, spine, clock hands must not be redrawn",
        "",
        "## How to read the masks",
        "",
        "- `masks/*_struct*`: Sobel agreement. Bright = H200 contour agrees with LQ.",
        "- `masks/*_conf*`: residual confidence. Dark = |H200-LQ| is large.",
        "- `masks/*_final_*`: M_struct * conf^gamma actually applied.",
        "- `heatmaps/*_residual*`: |H200-LQ| before/after percentile norm.",
        "",
        "## Not in this experiment",
        "",
        "HYPIR re-inference, diffusion parameter changes, LoRA, Adapter,",
        "classifier, depth model, training, 100-image test set.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    lq_dir: Path | str,
    h50_dir: Path | str,
    h200_dir: Path | str,
    out_dir: Path | str,
    *,
    gt_dir: Path | str | None = None,
    texture_dir: Path | str | None = None,
    cases: Sequence[str] | None = None,
    schemes: Sequence[str] = ("A", "B"),
    alpha_overrides: dict[str, float] | None = None,
    default_alpha: float = 0.15,
    blur_sigma: float = DEFAULT_BLUR_SIGMA,
    residual_blur_sigma: float = DEFAULT_RESIDUAL_BLUR_SIGMA,
    low_percentile: float = DEFAULT_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_HIGH_PERCENTILE,
    compute_lpips: bool = False,
    lpips_size: int = 1024,
) -> dict[str, object]:
    lq_dir, h50_dir, h200_dir, out = map(Path, (lq_dir, h50_dir, h200_dir, out_dir))
    gt_path = Path(gt_dir) if gt_dir is not None else None
    texture_path = Path(texture_dir) if texture_dir is not None else None
    out.mkdir(parents=True, exist_ok=True)
    fusion_folders = [variant[f"folder_{scheme.lower()}"] for variant in UNIQUE_MASK_VARIANTS for scheme in schemes]
    for sub in (
        *[f"fusion/{folder}" for folder in fusion_folders],
        "masks",
        "heatmaps",
        "comparison",
        "crops",
    ):
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
    schemes = tuple(scheme.upper() for scheme in schemes)
    if any(scheme not in {"A", "B"} for scheme in schemes):
        raise ValueError(f"schemes must be A and/or B, got {schemes}")

    lpips_model = lpips_device = None
    if compute_lpips:
        import torch
        import lpips

        lpips_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_model = lpips.LPIPS(net="alex", verbose=False).to(lpips_device).eval()

    metric_rows: list[dict[str, object]] = []
    case_info: list[dict[str, object]] = []
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
        texture = None
        if texture_path is not None and texture_path.is_dir():
            try:
                texture = load_rgb(_case_file(texture_path, case, ("_lq",)))
            except FileNotFoundError:
                texture = None

        scene, alpha = scene_alpha_for_case(case, overrides=alpha_overrides, default_alpha=default_alpha)
        struct = compute_structure_mask(rgb_to_y(lq), rgb_to_y(h200), blur_sigma=blur_sigma)
        conf, conf_diag = compute_residual_confidence(
            lq,
            h200,
            blur_sigma=residual_blur_sigma,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            return_diagnostics=True,
        )

        outputs: dict[str, np.ndarray] = {"LQ": lq, "HYPIR-50": h50, "HYPIR-200": h200}
        if texture is not None:
            outputs["texture_selective_h200"] = texture

        final_masks: dict[str, np.ndarray] = {}
        for variant in UNIQUE_MASK_VARIANTS:
            final = combine_masks(struct, conf, gamma=variant["gamma"])
            final_masks[variant["key"]] = final
            if "A" in schemes:
                fused_a = fuse_scheme_a(lq, h200, final, alpha)
                save_rgb(out / "fusion" / str(variant["folder_a"]) / f"{case}.png", fused_a)
                outputs[str(variant["method_a"])] = fused_a
            if "B" in schemes:
                fused_b = fuse_scheme_b(lq, h50, h200, final, alpha)
                save_rgb(out / "fusion" / str(variant["folder_b"]) / f"{case}.png", fused_b)
                outputs[str(variant["method_b"])] = fused_b
            save_rgb(out / "masks" / f"{case}_{variant['final_name']}.png", _colorize(final, vmax=1.0))
            _save_gray(out / "masks" / f"{case}_{variant['final_name']}_gray.png", final)
            applied = np.float32(alpha) * final
            save_rgb(
                out / "masks" / f"{case}_{variant['final_name']}_applied.png",
                _colorize(applied, vmax=max(DEFAULT_SCENE_ALPHA.values())),
            )

        residual = _residual_l1(h200, lq)
        save_rgb(out / "masks" / f"{case}_struct.png", _colorize(struct, vmax=1.0))
        _save_gray(out / "masks" / f"{case}_struct_gray.png", struct)
        save_rgb(out / "masks" / f"{case}_conf.png", _colorize(conf, vmax=1.0))
        _save_gray(out / "masks" / f"{case}_conf_gray.png", conf)
        save_rgb(out / "heatmaps" / f"{case}_residual.png", _colorize(residual, colormap=cv2.COLORMAP_INFERNO))
        save_rgb(
            out / "heatmaps" / f"{case}_residual_blur.png",
            _colorize(conf_diag["residual_blur"], colormap=cv2.COLORMAP_INFERNO),
        )
        save_rgb(
            out / "heatmaps" / f"{case}_residual_norm.png",
            _colorize(conf_diag["residual_norm"], vmax=1.0, colormap=cv2.COLORMAP_INFERNO),
        )
        save_rgb(
            out / "heatmaps" / f"{case}_suppressed_struct.png",
            _colorize((1.0 - struct) * residual, colormap=cv2.COLORMAP_INFERNO),
        )
        save_rgb(
            out / "heatmaps" / f"{case}_suppressed_conf.png",
            _colorize((1.0 - final_masks["conf"]) * residual, colormap=cv2.COLORMAP_INFERNO),
        )

        panel = [("LQ", lq), ("H50", h50), ("H200", h200)]
        if "fusion_A" in outputs:
            panel.append(("A_struct", outputs["fusion_A"]))
        if "fusion_A_conf05" in outputs:
            panel.append(("A_conf05", outputs["fusion_A_conf05"]))
        if "fusion_A_conf" in outputs:
            panel.append(("A_conf", outputs["fusion_A_conf"]))
        if "fusion_A_conf2" in outputs:
            panel.append(("A_conf2", outputs["fusion_A_conf2"]))
        if gt is not None:
            panel.append(("GT", gt))
        _panel(out / "comparison" / f"{case}.png", panel, max_width=768)

        panel_b = [("LQ", lq), ("H50", h50), ("H200", h200)]
        if "fusion_B" in outputs:
            panel_b.append(("B_struct", outputs["fusion_B"]))
        if "fusion_B_conf05" in outputs:
            panel_b.append(("B_conf05", outputs["fusion_B_conf05"]))
        if "fusion_B_conf" in outputs:
            panel_b.append(("B_conf", outputs["fusion_B_conf"]))
        if "fusion_B_conf2" in outputs:
            panel_b.append(("B_conf2", outputs["fusion_B_conf2"]))
        if gt is not None:
            panel_b.append(("GT", gt))
        _panel(out / "comparison" / f"{case}_B.png", panel_b, max_width=768)

        mask_panel = [
            ("struct", _colorize(struct, vmax=1.0)),
            ("conf", _colorize(conf, vmax=1.0)),
            ("|H200-LQ|", _colorize(residual, colormap=cv2.COLORMAP_INFERNO)),
            ("final_conf05", _colorize(final_masks["conf05"], vmax=1.0)),
            ("final_conf", _colorize(final_masks["conf"], vmax=1.0)),
            ("final_conf2", _colorize(final_masks["conf2"], vmax=1.0)),
        ]
        _panel(out / "comparison" / f"{case}_masks.png", mask_panel, max_width=768)

        evidence = [
            ("A_struct", outputs["fusion_A"]) if "fusion_A" in outputs else None,
            ("A_conf", outputs["fusion_A_conf"]) if "fusion_A_conf" in outputs else None,
            ("A_conf2", outputs["fusion_A_conf2"]) if "fusion_A_conf2" in outputs else None,
            ("H200", h200),
            ("GT", gt) if gt is not None else None,
        ]
        _panel(out / "comparison" / f"{case}_evidence.png", [item for item in evidence if item is not None], max_width=768)

        crop_methods = [
            ("LQ", lq),
            ("H200", h200),
            ("A_struct", outputs.get("fusion_A")),
            ("A_conf05", outputs.get("fusion_A_conf05")),
            ("A_conf", outputs.get("fusion_A_conf")),
            ("A_conf2", outputs.get("fusion_A_conf2")),
            ("B_struct", outputs.get("fusion_B")),
            ("B_conf", outputs.get("fusion_B_conf")),
            ("GT", gt),
        ]
        crop_methods = [(name, image) for name, image in crop_methods if image is not None]
        for crop_name, x0, y0, x1, y1 in DIAGNOSTIC_CROPS.get(case, ()):
            if y1 > lq.shape[0] or x1 > lq.shape[1]:
                continue
            crop_panel = [(name, image[y0:y1, x0:x1]) for name, image in crop_methods]
            _panel(out / "crops" / f"{case}_{crop_name}.png", crop_panel, max_width=256)
            mask_crop = [
                ("struct", _colorize(struct[y0:y1, x0:x1], vmax=1.0)),
                ("conf", _colorize(conf[y0:y1, x0:x1], vmax=1.0)),
                ("residual", _colorize(residual[y0:y1, x0:x1], colormap=cv2.COLORMAP_INFERNO)),
                ("final_conf", _colorize(final_masks["conf"][y0:y1, x0:x1], vmax=1.0)),
                ("final_conf2", _colorize(final_masks["conf2"][y0:y1, x0:x1], vmax=1.0)),
            ]
            _panel(out / "crops" / f"{case}_{crop_name}_masks.png", mask_crop, max_width=256)
            if gt is not None:
                gt_crop = gt[y0:y1, x0:x1]
                for name, image in crop_methods:
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
                            "mean_struct": f"{float(struct[y0:y1, x0:x1].mean()):.6f}",
                            "mean_conf": f"{float(conf[y0:y1, x0:x1].mean()):.6f}",
                            "mean_final_conf": f"{float(final_masks['conf'][y0:y1, x0:x1].mean()):.6f}",
                            "mean_final_conf2": f"{float(final_masks['conf2'][y0:y1, x0:x1].mean()):.6f}",
                        }
                    )

        mean_struct = float(struct.mean())
        mean_conf = float(conf.mean())
        mean_final_conf = float(final_masks["conf"].mean())
        mean_final_conf2 = float(final_masks["conf2"].mean())
        case_info.append(
            {
                "case": case,
                "scene": scene,
                "alpha": alpha,
                "mean_struct": mean_struct,
                "mean_conf": mean_conf,
                "mean_final_conf": mean_final_conf,
                "mean_final_conf05": float(final_masks["conf05"].mean()),
                "mean_final_conf2": mean_final_conf2,
                "mean_final_struct": mean_struct,
                "suppressed_frac_struct": float((struct < 0.3).mean()),
                "suppressed_frac_conf": float((final_masks["conf"] < 0.3).mean()),
            }
        )

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
                    "scene": scene,
                    "alpha": f"{alpha:.4f}",
                    "method": method,
                    "PSNR": f"{psnr:.6f}",
                    "SSIM": f"{ssim:.6f}",
                    "LPIPS_Alex": _fmt(lp),
                    "Delta_PSNR_vs_LQ": f"{psnr - lq_psnr:.6f}",
                    "Delta_SSIM_vs_LQ": f"{ssim - lq_ssim:.6f}",
                    "Delta_LPIPS_vs_LQ": "" if math.isnan(lp) else f"{lp - lq_lp:.6f}",
                    "mean_struct": f"{mean_struct:.6f}",
                    "mean_conf": f"{mean_conf:.6f}",
                    "mean_final_conf": f"{mean_final_conf:.6f}",
                    "mean_final_conf2": f"{mean_final_conf2:.6f}",
                    "suppressed_frac_struct": f"{float((struct < 0.3).mean()):.6f}",
                    "suppressed_frac_conf": f"{float((final_masks['conf'] < 0.3).mean()):.6f}",
                }
            )
            record(method, psnr, ssim, lp)

    metric_fields = [
        "case", "scene", "alpha", "method", "PSNR", "SSIM", "LPIPS_Alex",
        "Delta_PSNR_vs_LQ", "Delta_SSIM_vs_LQ", "Delta_LPIPS_vs_LQ",
        "mean_struct", "mean_conf", "mean_final_conf", "mean_final_conf2",
        "suppressed_frac_struct", "suppressed_frac_conf",
    ]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
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
                        "scene": "",
                        "alpha": "",
                        "method": method,
                        "PSNR": f"{psnr:.6f}",
                        "SSIM": f"{ssim:.6f}",
                        "LPIPS_Alex": "" if lp is None else f"{lp:.6f}",
                        "Delta_PSNR_vs_LQ": f"{psnr - lq_psnr:.6f}",
                        "Delta_SSIM_vs_LQ": f"{ssim - lq_ssim:.6f}",
                        "Delta_LPIPS_vs_LQ": "" if lp is None or lq_lp is None else f"{lp - lq_lp:.6f}",
                        "mean_struct": "",
                        "mean_conf": "",
                        "mean_final_conf": "",
                        "mean_final_conf2": "",
                        "suppressed_frac_struct": "",
                        "suppressed_frac_conf": "",
                    }
                )

    crop_fields = [
        "case", "crop", "method", "PSNR", "SSIM",
        "mean_struct", "mean_conf", "mean_final_conf", "mean_final_conf2",
    ]
    with (out / "crop_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=crop_fields)
        writer.writeheader()
        writer.writerows(crop_rows)

    (out / "experiment_metadata.md").write_text(
        "\n".join(
            [
                "# HYPIR fusion v2 metadata",
                "",
                f"- date: {date.today().isoformat()}",
                f"- lq_dir: `{lq_dir}`",
                f"- h50_dir: `{h50_dir}`",
                f"- h200_dir: `{h200_dir}`",
                f"- gt_dir: `{gt_path}` (metrics only; never used in the mask)",
                f"- cases: {', '.join(selected)}",
                f"- schemes: {', '.join(schemes)}",
                f"- scene_alpha: {DEFAULT_SCENE_ALPHA}",
                f"- alpha_overrides: {alpha_overrides or {}}",
                f"- struct_blur_sigma: {blur_sigma}",
                f"- residual_blur_sigma: {residual_blur_sigma}",
                f"- residual_percentiles: {low_percentile}-{high_percentile}",
                "- formula: `out_Y = base_Y + alpha * M_final * (H200_Y - base_Y)`",
                "- M_struct: v1 1/4 Sobel cosine * magnitude similarity",
                "- conf: `1 - percentile_norm(GaussianBlur(mean_c |H200-LQ|, σ), 1–99)`",
                "- groups A/B: `M_final = M_struct` (A is original fusion_v1; B is the same formula)",
                "- group C: `M_final = M_struct * conf`",
                "- group D: `M_final = M_struct * conf^2`",
                "- extra gamma=0.5: `M_final = M_struct * conf^0.5`",
                "- chroma: LQ YCrCb Cr/Cb",
                "- inference: none; existing HYPIR PNGs only",
                f"- lpips: Alex max-side {lpips_size}, enabled={compute_lpips}",
                "- does not write hypir_fusion_v1/results/",
                "",
                "## Scene notes",
                "",
                *[f"- {name}: {note}" for name, note in SCENE_NOTES.items()],
                "",
                "## MASK_GROUPS",
                "",
                *[
                    f"- {name}: key={item['key']} gamma={item['gamma']} formula=`{item['formula']}`"
                    for name, item in MASK_GROUPS.items()
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_summary(out, metric_rows, case_info, crop_rows, compute_lpips)
    return {
        "cases": selected,
        "metrics": metric_rows,
        "crop_metrics": crop_rows,
        "output_dir": out,
        "case_info": case_info,
    }


def main(argv: list[str] | None = None) -> int:
    experiment_dir = Path(__file__).resolve().parent
    project = experiment_dir.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project, help="project root")
    parser.add_argument("--lq-dir", type=Path, default=None)
    parser.add_argument("--h50-dir", type=Path, default=None)
    parser.add_argument("--h200-dir", type=Path, default=None)
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument("--texture-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cases", default="case1,case2,case3,case4,case5")
    parser.add_argument("--scheme", choices=["A", "B", "both"], default="both")
    parser.add_argument("--alpha-text", type=float, default=DEFAULT_SCENE_ALPHA["text"])
    parser.add_argument("--alpha-book", type=float, default=DEFAULT_SCENE_ALPHA["book"])
    parser.add_argument("--alpha-bird", type=float, default=DEFAULT_SCENE_ALPHA["bird"])
    parser.add_argument("--alpha-plant", type=float, default=DEFAULT_SCENE_ALPHA["plant"])
    parser.add_argument("--alpha-clock", type=float, default=DEFAULT_SCENE_ALPHA["clock"])
    parser.add_argument("--default-alpha", type=float, default=0.15)
    parser.add_argument("--blur-sigma", type=float, default=DEFAULT_BLUR_SIGMA)
    parser.add_argument("--residual-blur-sigma", type=float, default=DEFAULT_RESIDUAL_BLUR_SIGMA)
    parser.add_argument("--low-percentile", type=float, default=DEFAULT_LOW_PERCENTILE)
    parser.add_argument("--high-percentile", type=float, default=DEFAULT_HIGH_PERCENTILE)
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
    texture_default = (
        root / "baseline" / "experiments" / "structure_local_restoration_v1" / "fusion" / "texture_selective" / "h200"
    )
    texture_dir = args.texture_dir or (texture_default if texture_default.is_dir() else None)
    out_dir = args.output_dir or (experiment_dir / "results" / "fusion_v2")
    cases = [item.strip() for item in args.cases.split(",") if item.strip()]
    schemes = ("A", "B") if args.scheme == "both" else (args.scheme,)
    overrides = {
        "text": args.alpha_text,
        "book": args.alpha_book,
        "bird": args.alpha_bird,
        "plant": args.alpha_plant,
        "clock": args.alpha_clock,
    }
    result = run_experiment(
        lq_dir,
        h50_dir,
        h200_dir,
        out_dir,
        gt_dir=gt_dir,
        texture_dir=texture_dir,
        cases=cases,
        schemes=schemes,
        alpha_overrides=overrides,
        default_alpha=args.default_alpha,
        blur_sigma=args.blur_sigma,
        residual_blur_sigma=args.residual_blur_sigma,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        compute_lpips=args.compute_lpips,
        lpips_size=args.lpips_size,
    )
    print(f"Wrote {len(result['cases'])} cases to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
