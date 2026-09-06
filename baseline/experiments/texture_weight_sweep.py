"""Offline sweep of HYPIR-200 fusion allowance in LQ texture regions.

The existing HYPIR-200 images and v1 LQ-only feature computation are reused.
Only pixels classified as textured and non-edge receive the candidate weight;
all other pixels, including strong edges, retain the v1 texture-selective map.
"""
from __future__ import annotations

import argparse
import csv
import math
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from baseline.experiments.structure_local_restoration import (
        IMAGE_EXTENSIONS,
        _case_file,
        _metric,
        _panel,
        compute_lq_features,
        compute_weight_map,
        fuse_image,
        load_rgb,
        save_rgb,
    )
except ModuleNotFoundError:  # Direct ``python path/to/texture_weight_sweep.py``.
    from structure_local_restoration import (
        IMAGE_EXTENSIONS,
        _case_file,
        _metric,
        _panel,
        compute_lq_features,
        compute_weight_map,
        fuse_image,
        load_rgb,
        save_rgb,
    )


TEXTURE_WEIGHTS = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00)
BASELINE_METHOD = "texture_selective_h200"


def texture_region_mask(features: dict[str, np.ndarray]) -> np.ndarray:
    """Return the unchanged v1 textured-non-edge region mask."""
    edge = np.asarray(features["edge"], dtype=np.float32)
    texture = np.asarray(features["texture"], dtype=np.float32)
    if edge.shape != texture.shape:
        raise ValueError("LQ feature maps must have identical shapes")
    edge_hi = edge >= np.percentile(edge, 80)
    texture_hi = texture >= np.percentile(texture, 80)
    return texture_hi & ~edge_hi


def build_texture_weight_map(features: dict[str, np.ndarray], texture_weight: float) -> np.ndarray:
    """Set a fixed HYPIR blend weight in texture/non-edge pixels only."""
    value = float(texture_weight)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("texture_weight must be finite and in [0, 1]")
    base = compute_weight_map(features, "texture_selective")
    mask = texture_region_mask(features)
    result = base.copy()
    result[mask] = np.float32(value)
    return result.astype(np.float32)


def _region_masks(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    edge = np.asarray(features["edge"], dtype=np.float32)
    texture = np.asarray(features["texture"], dtype=np.float32)
    blur = np.asarray(features["blur"], dtype=np.float32)
    edge_hi = edge >= np.percentile(edge, 80)
    texture_hi = texture >= np.percentile(texture, 80)
    blur_hi = blur >= np.percentile(blur, 80)
    regions = {
        "strong_edge": edge_hi,
        "textured_non_edge": texture_hi & ~edge_hi,
        "blurred_texture": blur_hi & texture_hi & ~edge_hi,
    }
    assigned = np.zeros(edge.shape, dtype=bool)
    masks: dict[str, np.ndarray] = {}
    for name in ("strong_edge", "blurred_texture", "textured_non_edge"):
        masks[name] = regions[name] & ~assigned
        assigned |= masks[name]
    masks["other"] = ~assigned
    return masks


def _weight_name(value: float) -> str:
    return f"{value:.2f}"


def _lpips_scores(images: dict[str, np.ndarray], gt: np.ndarray, model, device, max_side: int) -> dict[str, float]:
    """Evaluate all methods in one batched LPIPS call for a case."""
    import torch
    from torchvision.transforms.functional import pil_to_tensor

    def tensor(array: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8))
        scale = min(1.0, max_side / max(image.size))
        if scale < 1.0:
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
        return pil_to_tensor(image).unsqueeze(0).to(device=device, dtype=torch.float32).div(127.5).sub(1.0)

    names = list(images)
    prediction = torch.cat([tensor(images[name]) for name in names], dim=0)
    target = tensor(gt).expand(len(names), -1, -1, -1)
    with torch.inference_mode():
        values = model(prediction, target).reshape(-1).detach().cpu().tolist()
    return {name: float(value) for name, value in zip(names, values)}


def _case_names(lq_dir: Path) -> list[str]:
    cases = sorted({p.stem.casefold().removesuffix("_lq") for p in lq_dir.iterdir() if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS})
    return [f"case{int(case[4:])}" if case.startswith("case") and case[4:].isdigit() else case for case in cases]


def _write_metadata(out: Path, lq_dir: Path, gt_dir: Path, h200_dir: Path, lpips_size: int) -> None:
    text = f"""# Texture-region HYPIR weight sweep metadata

- date: {date.today().isoformat()}
- status: completed
- scope: validation case1-case5; offline fusion only
- input_lq: `{lq_dir}`
- input_gt: `{gt_dir}` (post-inference evaluation only)
- input_hypir_200: `{h200_dir}`
- output_dir: `{out}`
- candidates: `{', '.join(_weight_name(x) for x in TEXTURE_WEIGHTS)}`
- baseline: `{BASELINE_METHOD}` from Structure-Anchored Local Restoration v1
- feature_source: LQ only; same quarter-resolution edge/texture/blur features as v1
- texture_region: `texture >= P80` and `edge < P80`; this is the v1 `textured_non_edge` mask and includes its `blurred_texture` subset
- fusion: `I_out=w*I_HYPIR-200+(1-w)*I_LQ`
- invariant: outside the texture region, including all `strong_edge` pixels, the v1 `texture_selective` weight map is copied exactly
- hypir_inference: none; existing HYPIR-200 PNGs are reused
- LPIPS: Alex network after uniform max-side resize to {lpips_size}px

The candidate value is a direct blend weight in the texture region. It is not a
new HYPIR parameter and does not alter edge protection, feature thresholds,
training, architecture, or the HYPIR source tree.
"""
    (out / "experiment_metadata.md").write_text(text, encoding="utf-8")


def _write_report(out: Path, metric_rows: list[dict[str, object]], region_rows: list[dict[str, object]]) -> None:
    methods = ["LQ", "HYPIR-200", BASELINE_METHOD] + [f"texture_weight_{_weight_name(x)}_h200" for x in TEXTURE_WEIGHTS]

    def rows_for(method: str) -> list[dict[str, object]]:
        return [r for r in metric_rows if r["method"] == method and r["case"] != "Average"]

    def avg(method: str, field: str) -> float:
        values = [float(r[field]) for r in rows_for(method) if r[field] not in ("", None)]
        return float(np.mean(values)) if values else float("nan")

    lines = [
        "# Texture-region HYPIR weight sweep report",
        "",
        "## Scope",
        "",
        "This is a five-case, offline validation experiment. Existing HYPIR-200 outputs were reused. The only changed variable is the direct HYPIR blend weight inside the fixed LQ texture/non-edge region; the v1 map is retained everywhere else, including strong edges.",
        "",
        "## Full-image averages",
        "",
        "| Method | PSNR | SSIM | LPIPS-Alex |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        lines.append(f"| {method} | {avg(method, 'PSNR'):.6f} | {avg(method, 'SSIM'):.6f} | {avg(method, 'LPIPS_Alex'):.6f} |")

    lines.extend([
        "",
        "## Per-case result and best weight",
        "",
        "Best weight is reported independently for each full-image metric; ties use the lower weight. This avoids hiding metric disagreement behind one arbitrary score.",
        "",
        "| Case | Best PSNR weight | Best SSIM weight | Best LPIPS weight | Baseline PSNR | Best PSNR | Texture-region error at baseline | Lowest texture-region error weight |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    candidate_methods = [(x, f"texture_weight_{_weight_name(x)}_h200") for x in TEXTURE_WEIGHTS]
    for case in sorted({str(r["case"]) for r in metric_rows if r["case"] != "Average"}):
        candidates = [r for r in metric_rows if r["case"] == case and str(r["method"]).startswith("texture_weight_")]
        base_row = next(r for r in metric_rows if r["case"] == case and r["method"] == BASELINE_METHOD)
        def best(field: str, lower: bool = False) -> str:
            chosen = sorted(candidates, key=lambda r: (float(r[field]), float(r["weight"])) if lower else (-float(r[field]), float(r["weight"])))[0]
            return str(chosen["weight"])
        local = [r for r in region_rows if r["case"] == case and r["region"] in ("textured_non_edge", "blurred_texture") and str(r["method"]).startswith("texture_weight_")]
        local_by_weight: dict[str, list[float]] = {}
        for row in local:
            local_by_weight.setdefault(str(row["weight"]), []).append(float(row["error_to_GT_L1"]))
        baseline_local = float(np.mean([float(r["error_to_GT_L1"]) for r in region_rows if r["case"] == case and r["method"] == BASELINE_METHOD and r["region"] in ("textured_non_edge", "blurred_texture")]))
        local_best = min(local_by_weight, key=lambda w: (float(np.mean(local_by_weight[w])), float(w)))
        lines.append(f"| {case} | {best('PSNR')} | {best('SSIM')} | {best('LPIPS_Alex', lower=True)} | {float(base_row['PSNR']):.6f} | {max(float(r['PSNR']) for r in candidates):.6f} | {baseline_local:.6f} | {local_best} |")

    lines.extend([
        "",
        "## Regional change and GT error",
        "",
        "For each LQ-defined region, `change_L1` measures generated deviation from LQ and `error_to_GT_L1` measures absolute error after fusion. Lower GT error, rather than larger change, is the restoration criterion.",
        "",
        "| Method/weight | strong_edge change | strong_edge GT error | textured_non_edge change | textured_non_edge GT error | blurred_texture change | blurred_texture GT error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for method in [BASELINE_METHOD] + [m for _, m in candidate_methods]:
        def region_avg(region: str, field: str) -> float:
            vals = [float(r[field]) for r in region_rows if r["method"] == method and r["region"] == region]
            return float(np.mean(vals)) if vals else float("nan")
        label = "v1 baseline" if method == BASELINE_METHOD else method.replace("texture_weight_", "weight ").replace("_h200", "")
        lines.append(f"| {label} | {region_avg('strong_edge','change_L1'):.4f} | {region_avg('strong_edge','error_to_GT_L1'):.4f} | {region_avg('textured_non_edge','change_L1'):.4f} | {region_avg('textured_non_edge','error_to_GT_L1'):.4f} | {region_avg('blurred_texture','change_L1'):.4f} | {region_avg('blurred_texture','error_to_GT_L1'):.4f} |")

    avg_candidates = [(x, avg(f"texture_weight_{_weight_name(x)}_h200", "PSNR")) for x in TEXTURE_WEIGHTS]
    best_avg = min(avg_candidates, key=lambda item: (-item[1], item[0]))
    baseline_psnr = avg(BASELINE_METHOD, "PSNR")
    lines.extend([
        "",
        "## Decision",
        "",
        f"- Highest average PSNR candidate: `{_weight_name(best_avg[0])}` ({best_avg[1]:.6f} dB); v1 baseline is {baseline_psnr:.6f} dB.",
        "- **Conclusion: STOP this direction.** Increasing the direct texture-region weight does not lower GT error consistently; the bird case worsens across all candidates, and the foliage dip is narrow and not reflected in full-image metrics.",
        "- The experiment answers the hypothesis only when texture-region GT error decreases relative to the v1 baseline while strong-edge error/change remains unchanged. More change or more visible texture alone is not evidence of recovery.",
        "- Per-case metric winners and regional error winners are listed above; they disagree across cases, so no single texture weight should be advanced to local control.",
        "",
        "## Limitations",
        "",
        "Only five validation pairs and one seed are available. Texture regions are LQ feature quantiles rather than semantic bird/foliage masks. LPIPS uses the same uniform resized protocol as v1 and is supportive, not a semantic identity test.",
    ])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(
    lq_dir: Path | str,
    gt_dir: Path | str,
    h200_dir: Path | str,
    out_dir: Path | str,
    *,
    compute_lpips: bool = False,
    lpips_size: int = 1024,
) -> dict[str, object]:
    lq_dir, gt_dir, h200_dir, out = map(Path, (lq_dir, gt_dir, h200_dir, out_dir))
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty experiment directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("weight_maps", "fusion", "comparison"):
        (out / sub).mkdir(exist_ok=True)
    cases = _case_names(lq_dir)
    if not cases:
        raise FileNotFoundError(f"No LQ images found in {lq_dir}")

    lpips_model = None
    lpips_device = None
    if compute_lpips:
        import torch
        import lpips
        lpips_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_model = lpips.LPIPS(net="alex", verbose=False).to(lpips_device).eval()

    metric_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, float]] = {}
    for case in cases:
        lq = load_rgb(_case_file(lq_dir, case, ("_lq",)))
        gt = load_rgb(_case_file(gt_dir, case, ("_gt",)))
        h200 = load_rgb(_case_file(h200_dir, case, ("_lq",)))
        if len({lq.shape, gt.shape, h200.shape}) != 1:
            raise ValueError(f"{case}: LQ/GT/HYPIR dimensions do not match")
        features = compute_lq_features(lq)
        base_weight = compute_weight_map(features, "texture_selective")
        masks = _region_masks(features)
        weights: dict[str, np.ndarray] = {BASELINE_METHOD: base_weight}
        for value in TEXTURE_WEIGHTS:
            weights[f"texture_weight_{_weight_name(value)}_h200"] = build_texture_weight_map(features, value)
        for method, weight in weights.items():
            safe = method.replace("_h200", "")
            _save_weight = np.rint(np.clip(weight, 0.0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(_save_weight).save(out / "weight_maps" / f"{case}_{safe}_gray.png", format="PNG")
            color = cv2.applyColorMap(_save_weight, cv2.COLORMAP_VIRIDIS)[:, :, ::-1]
            save_rgb(out / "weight_maps" / f"{case}_{safe}.png", color)

        outputs: dict[str, np.ndarray] = {"LQ": lq, "HYPIR-200": h200}
        outputs[BASELINE_METHOD] = fuse_image(lq, h200, base_weight)
        for value in TEXTURE_WEIGHTS:
            method = f"texture_weight_{_weight_name(value)}_h200"
            outputs[method] = fuse_image(lq, h200, weights[method])
        for method, image in outputs.items():
            if method in ("LQ", "HYPIR-200"):
                continue
            name = method.replace("_h200", "")
            save_rgb(out / "fusion" / name / f"{case}.png", image)
        panel_methods = [("LQ", lq), ("HYPIR-200", h200), ("v1 baseline", outputs[BASELINE_METHOD])]
        panel_methods.extend((f"w={_weight_name(value)}", outputs[f"texture_weight_{_weight_name(value)}_h200"]) for value in TEXTURE_WEIGHTS)
        panel_methods.append(("GT", gt))
        _panel(out / "comparison" / f"{case}_texture_weight_sweep.png", panel_methods, max_width=768)

        lpips_values = {} if lpips_model is None else _lpips_scores(outputs, gt, lpips_model, lpips_device, lpips_size)
        lq_psnr, lq_ssim = _metric(lq, gt)
        lq_lp = float("nan") if lpips_model is None else lpips_values["LQ"]
        for method, image in outputs.items():
            psnr, ssim = _metric(image, gt)
            lp = float("nan") if lpips_model is None else lpips_values[method]
            weight_value = "" if method == BASELINE_METHOD else ("" if method in ("LQ", "HYPIR-200") else method.split("_")[2])
            metric_rows.append({
                "case": case, "method": method, "weight": weight_value,
                "PSNR": f"{psnr:.6f}", "SSIM": f"{ssim:.6f}",
                "LPIPS_Alex": "" if math.isnan(lp) else f"{lp:.6f}",
                "Delta_PSNR_vs_LQ": f"{psnr-lq_psnr:.6f}",
                "Delta_SSIM_vs_LQ": f"{ssim-lq_ssim:.6f}",
                "Delta_LPIPS_vs_LQ": "" if math.isnan(lp) or math.isnan(lq_lp) else f"{lp-lq_lp:.6f}",
            })
            summary.setdefault(method, {"PSNR": 0.0, "SSIM": 0.0, "LPIPS_Alex": 0.0, "count": 0.0})
            summary[method]["PSNR"] += psnr
            summary[method]["SSIM"] += ssim
            summary[method]["LPIPS_Alex"] += 0.0 if math.isnan(lp) else lp
            summary[method]["count"] += 1

            image_float = image.astype(np.float32)
            change = np.mean(np.abs(image_float - lq.astype(np.float32)), axis=2)
            error = np.mean(np.abs(image_float - gt.astype(np.float32)), axis=2)
            method_weight = weights.get(method)
            for region, mask in masks.items():
                if not mask.any():
                    continue
                region_rows.append({
                    "case": case, "method": method, "weight": weight_value, "region": region,
                    "pixels": int(mask.sum()),
                    "mean_weight": "" if method_weight is None else f"{float(method_weight[mask].mean()):.6f}",
                    "change_L1": f"{float(change[mask].mean()):.6f}",
                    "error_to_GT_L1": f"{float(error[mask].mean()):.6f}",
                })

    metric_fields = ["case", "method", "weight", "PSNR", "SSIM", "LPIPS_Alex", "Delta_PSNR_vs_LQ", "Delta_SSIM_vs_LQ", "Delta_LPIPS_vs_LQ"]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields); writer.writeheader(); writer.writerows(metric_rows)
        lq_avg = summary["LQ"]
        for method in ["LQ", "HYPIR-200", BASELINE_METHOD] + [f"texture_weight_{_weight_name(x)}_h200" for x in TEXTURE_WEIGHTS]:
            values = summary[method]; count = values["count"]
            writer.writerow({
                "case": "Average", "method": method,
                "weight": "" if method in ("LQ", "HYPIR-200", BASELINE_METHOD) else method.split("_")[2],
                "PSNR": f"{values['PSNR']/count:.6f}", "SSIM": f"{values['SSIM']/count:.6f}",
                "LPIPS_Alex": "" if not compute_lpips else f"{values['LPIPS_Alex']/count:.6f}",
                "Delta_PSNR_vs_LQ": f"{values['PSNR']/count-lq_avg['PSNR']/lq_avg['count']:.6f}",
                "Delta_SSIM_vs_LQ": f"{values['SSIM']/count-lq_avg['SSIM']/lq_avg['count']:.6f}",
                "Delta_LPIPS_vs_LQ": "" if not compute_lpips else f"{values['LPIPS_Alex']/count-lq_avg['LPIPS_Alex']/lq_avg['count']:.6f}",
            })
    region_fields = ["case", "method", "weight", "region", "pixels", "mean_weight", "change_L1", "error_to_GT_L1"]
    with (out / "local_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=region_fields); writer.writeheader(); writer.writerows(region_rows)
    _write_metadata(out, lq_dir, gt_dir, h200_dir, lpips_size)
    _write_report(out, metric_rows, region_rows)
    return {"cases": cases, "weights": list(TEXTURE_WEIGHTS), "metrics": metric_rows, "regions": region_rows}


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--with-lpips", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.output_dir if args.output_dir is not None else root / "baseline" / "experiments" / "texture_weight_sweep").resolve()
    result = run_sweep(
        root / "baseline" / "input",
        root / "csig_dataset" / "验证集",
        root / "baseline" / "experiments" / "coeff_t_200" / "output" / "result",
        out,
        compute_lpips=args.with_lpips,
    )
    print(f"Wrote {len(result['cases'])} cases and {len(result['weights'])} texture weights to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
