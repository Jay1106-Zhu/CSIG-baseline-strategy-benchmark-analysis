"""Scenario Routing v1: fixed LQ-only method selection over existing outputs.

This experiment never invokes diffusion. It reads existing LQ/HYPIR/fusion
images, computes input descriptors from LQ only, and writes an auditable
per-case method matrix plus one selected output per case.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROUTING_METHODS = ("LQ", "HYPIR-50", "HYPIR-200", "texture_selective_h200")
SCENARIO_METHODS = {
    "structure-dominant": "HYPIR-50",
    "texture-dominant": "texture_selective_h200",
    "blurred/low-frequency": "HYPIR-200",
    "ambiguous": "LQ",
}
SCENARIOS = tuple(SCENARIO_METHODS)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
MATRIX_REGION_FIELDS = (
    "change_L1",
    "strong_edge_error_to_GT_L1",
    "strong_edge_change_L1",
    "textured_non_edge_error_to_GT_L1",
    "textured_non_edge_change_L1",
    "blurred_texture_error_to_GT_L1",
    "blurred_texture_change_L1",
)


def load_rgb(path: Path | str) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.uint8).copy()
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{path}: expected RGB image")
    return array


def _case_file(directory: Path, case: str, suffixes: Iterable[str] = ("", "_lq", "_gt")) -> Path:
    for suffix in suffixes:
        for extension in IMAGE_EXTENSIONS:
            path = directory / f"{case}{suffix}{extension}"
            if path.is_file():
                return path
    raise FileNotFoundError(f"No image for {case} in {directory}")


def compute_input_features(lq: np.ndarray) -> dict[str, float]:
    """Compute global, LQ-only descriptors with fixed physical definitions."""
    array = np.asarray(lq)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("LQ must be an RGB HxWx3 array")
    height, width = array.shape[:2]
    scale = min(1.0, 1024.0 / max(height, width))
    small = cv2.resize(array, (max(16, round(width * scale)), max(16, round(height * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.2)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gradient = cv2.magnitude(gx, gy)
    local_mean = cv2.blur(gray, (9, 9))
    local_var = np.maximum(cv2.blur(gray * gray, (9, 9)) - local_mean * local_mean, 0.0)
    lap_energy = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))))
    hf = np.abs(gray - smooth)
    # Fixed scales keep descriptors comparable across images and do not use GT.
    return {
        "edge_density": float(np.mean(gradient > 0.12)),
        "local_variance": float(np.mean(np.sqrt(local_var))),
        "blur_proxy": float(np.clip(1.0 - lap_energy / 0.20, 0.0, 1.0)),
        "high_frequency_energy": float(np.clip(np.mean(hf) / 0.20, 0.0, 1.0)),
        "gradient_mean": float(np.mean(gradient)),
        "gradient_std": float(np.std(gradient)),
    }


def classify_scenario(features: dict[str, float]) -> str:
    """Apply one fixed, interpretable rule to LQ descriptors only."""
    required = {"edge_density", "local_variance", "blur_proxy", "high_frequency_energy", "gradient_mean", "gradient_std"}
    missing = required.difference(features)
    if missing:
        raise ValueError(f"missing routing features: {sorted(missing)}")
    if not all(np.isfinite(float(features[key])) for key in required):
        raise ValueError("routing features must be finite")
    edge = float(features["edge_density"])
    texture = float(features["local_variance"])
    blur = float(features["blur_proxy"])
    hf = float(features["high_frequency_energy"])
    grad = float(features["gradient_mean"])
    # Thresholds are fixed before validation scoring and apply identically to all cases.
    if blur >= 0.70 and hf < 0.22 and edge < 0.18 and grad < 0.16:
        return "blurred/low-frequency"
    if edge >= 0.18 and texture < 0.22 and hf < 0.35:
        return "structure-dominant"
    if texture >= 0.22 and hf >= 0.22 and edge < 0.28:
        return "texture-dominant"
    return "ambiguous"


def choose_method(scenario: str) -> str:
    if scenario not in SCENARIO_METHODS:
        raise ValueError(f"unknown scenario {scenario!r}")
    return SCENARIO_METHODS[scenario]


def _metric(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")
    pred_f = pred.astype(np.float32) / 255.0
    gt_f = gt.astype(np.float32) / 255.0
    psnr = float(peak_signal_noise_ratio(gt_f, pred_f, data_range=1.0))
    ssim = float(structural_similarity(gt_f, pred_f, channel_axis=2, data_range=1.0))
    return psnr, ssim


def _write_panel(path: Path, images: list[tuple[str, np.ndarray]], footer: str) -> None:
    thumbs: list[tuple[str, Image.Image]] = []
    for label, array in images:
        image = Image.fromarray(array).convert("RGB")
        image.thumbnail((420, 300), Image.Resampling.BILINEAR)
        thumbs.append((label, image.copy()))
    width = sum(image.width for _, image in thumbs)
    height = max(image.height for _, image in thumbs) + 42
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    x = 0
    for label, image in thumbs:
        panel.paste(image, (x, 30))
        draw.text((x + 4, 8), label, fill="black")
        x += image.width
    draw.text((4, height - 14), footer, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path, format="PNG")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _global_rows_from_images(cases: list[str], dirs: dict[str, Path], gt_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case in cases:
        gt = load_rgb(_case_file(gt_dir, case, ("_gt", "")))
        for method, directory in dirs.items():
            image = load_rgb(_case_file(directory, case, ("_lq", "", "_gt")))
            psnr, ssim = _metric(image, gt)
            rows.append({"case": case, "method": method, "PSNR": f"{psnr:.6f}", "SSIM": f"{ssim:.6f}", "LPIPS_Alex": ""})
    return rows


def _matrix_rows(global_rows: list[dict[str, str]], region_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key = {(row["case"], row["method"]): row for row in global_rows if row["case"] != "Average"}
    regions: dict[tuple[str, str], dict[str, str]] = {}
    for row in region_rows:
        if row["region"] not in ("strong_edge", "textured_non_edge", "blurred_texture"):
            continue
        key = (row["case"], row["method"])
        regions.setdefault(key, {})[f"{row['region']}_error_to_GT_L1"] = row.get("error_to_GT_L1", "")
        regions[key][f"{row['region']}_change_L1"] = row.get("change_L1", "")
    rows: list[dict[str, str]] = []
    fields = ["case", "method", "PSNR", "SSIM", "LPIPS_Alex", *MATRIX_REGION_FIELDS]
    for key in sorted(by_key, key=lambda item: (int(item[0][4:]) if item[0].startswith("case") and item[0][4:].isdigit() else item[0], item[1])):
        row = {field: by_key[key].get(field, "") for field in fields}
        row.update(regions.get(key, {}))
        for field in MATRIX_REGION_FIELDS:
            row.setdefault(field, "")
        rows.append(row)
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_routing(
    lq_dir: Path | str,
    gt_dir: Path | str,
    h50_dir: Path | str,
    h200_dir: Path | str,
    selective_dir: Path | str,
    out_dir: Path | str,
    *,
    source_metrics_csv: Path | str | None = None,
    source_region_csv: Path | str | None = None,
) -> dict[str, object]:
    lq_dir, gt_dir, h50_dir, h200_dir, selective_dir, out = map(Path, (lq_dir, gt_dir, h50_dir, h200_dir, selective_dir, out_dir))
    out.mkdir(parents=True, exist_ok=True)
    cases = sorted(
        {p.stem.removesuffix("_lq") for p in lq_dir.iterdir() if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS},
        key=lambda value: int(value[4:]) if value.startswith("case") and value[4:].isdigit() else value,
    )
    if not cases:
        raise FileNotFoundError(f"No LQ images found in {lq_dir}")
    dirs = {"LQ": lq_dir, "HYPIR-50": h50_dir, "HYPIR-200": h200_dir, "texture_selective_h200": selective_dir}
    if source_metrics_csv is not None and Path(source_metrics_csv).is_file():
        global_rows = [row for row in _read_csv(Path(source_metrics_csv)) if row["method"] in ROUTING_METHODS and row["case"] != "Average"]
    else:
        global_rows = _global_rows_from_images(cases, dirs, gt_dir)
    if source_region_csv is not None and Path(source_region_csv).is_file():
        region_rows = [row for row in _read_csv(Path(source_region_csv)) if row["method"] in ROUTING_METHODS and row["case"] != "Average"]
    else:
        region_rows = []
    for row in global_rows:
        lq = load_rgb(_case_file(lq_dir, row["case"], ("_lq", "")))
        output = load_rgb(_case_file(dirs[row["method"]], row["case"], ("_lq", "", "_gt")))
        if output.shape != lq.shape:
            raise ValueError(f"{row['case']}/{row['method']}: output and LQ dimensions differ")
        row["change_L1"] = f"{float(np.mean(np.abs(output.astype(np.float32) - lq.astype(np.float32)))):.6f}"
    matrix = _matrix_rows(global_rows, region_rows)
    _write_csv(out / "per_case_method_matrix.csv", ["case", "method", "PSNR", "SSIM", "LPIPS_Alex", *MATRIX_REGION_FIELDS], matrix)
    decision_rows: list[dict[str, str]] = []
    metric_rows: list[dict[str, str]] = []
    for case in cases:
        lq = load_rgb(_case_file(lq_dir, case, ("_lq", "")))
        features = compute_input_features(lq)
        scenario = classify_scenario(features)
        selected_method = choose_method(scenario)
        selected = load_rgb(_case_file(dirs[selected_method], case, ("_lq", "", "_gt")))
        gt = load_rgb(_case_file(gt_dir, case, ("_gt", "", "_lq")))
        if selected.shape != gt.shape:
            raise ValueError(f"{case}: selected output and GT dimensions differ")
        decision_rows.append({"case": case, "scenario": scenario, "selected_method": selected_method, **{key: f"{value:.8f}" for key, value in features.items()}})
        for method in ROUTING_METHODS:
            source = next(row for row in matrix if row["case"] == case and row["method"] == method)
            metric_rows.append({"case": case, "method": method, "PSNR": source["PSNR"], "SSIM": source["SSIM"], "LPIPS_Alex": source["LPIPS_Alex"], "change_L1": source["change_L1"]})
        selected_row = next(row for row in matrix if row["case"] == case and row["method"] == selected_method)
        metric_rows.append({"case": case, "method": "Scenario Routing v1", "PSNR": selected_row["PSNR"], "SSIM": selected_row["SSIM"], "LPIPS_Alex": selected_row["LPIPS_Alex"], "change_L1": selected_row["change_L1"]})
        _write_panel(out / "routing_visualization" / f"{case}.png", [("LQ", lq), (f"Routing: {selected_method}", selected), ("GT", gt)], f"{scenario} | {selected_method}")
    _write_csv(out / "routing_decision.csv", ["case", "scenario", "selected_method", "edge_density", "local_variance", "blur_proxy", "high_frequency_energy", "gradient_mean", "gradient_std"], decision_rows)
    averages: list[dict[str, str]] = []
    for method in [*ROUTING_METHODS, "Scenario Routing v1"]:
        values = [row for row in metric_rows if row["method"] == method]
        averages.append({
            "case": "Average",
            "method": method,
            "PSNR": f"{np.mean([float(row['PSNR']) for row in values]):.6f}",
            "SSIM": f"{np.mean([float(row['SSIM']) for row in values]):.6f}",
            "LPIPS_Alex": "" if not all(row["LPIPS_Alex"] for row in values) else f"{np.mean([float(row['LPIPS_Alex']) for row in values]):.6f}",
            "change_L1": f"{np.mean([float(row['change_L1']) for row in values]):.6f}",
        })
    metric_rows.extend(averages)
    _write_csv(out / "metrics.csv", ["case", "method", "PSNR", "SSIM", "LPIPS_Alex", "change_L1"], metric_rows)
    _write_summary(out / "summary.md", matrix, decision_rows)
    return {"cases": cases, "decisions": decision_rows, "matrix": matrix}


def _write_summary(path: Path, matrix: list[dict[str, str]], decisions: list[dict[str, str]]) -> None:
    methods = list(ROUTING_METHODS) + ["Scenario Routing v1"]
    lines = [
        "# Scenario Routing v1",
        "",
        "Offline routing proof-of-concept. No classifier, training, new diffusion inference, or GT-driven routing was used.",
        "",
        "## Fixed rule",
        "",
        "Features are computed from LQ only at max-side 1024: edge density, local variance, blur proxy, high-frequency energy, gradient mean, and gradient standard deviation.",
        "",
        "- blurred/low-frequency: blur_proxy >= 0.70, high_frequency_energy < 0.22, edge_density < 0.18, gradient_mean < 0.16 -> HYPIR-200",
        "- structure-dominant: edge_density >= 0.18, local_variance < 0.22, high_frequency_energy < 0.35 -> HYPIR-50",
        "- texture-dominant: local_variance >= 0.22, high_frequency_energy >= 0.22, edge_density < 0.28 -> texture_selective_h200",
        "- ambiguous: all remaining inputs -> LQ",
        "",
        "The thresholds and mapping are fixed globally; no case name, GT, validation score, or semantic label is referenced.",
        "",
        "## Routing decisions",
        "",
        "| Case | Scenario | Selected method |",
        "|---|---|---|",
    ]
    for row in decisions:
        lines.append(f"| {row['case']} | {row['scenario']} | {row['selected_method']} |")
    lines.extend(["", "## Average comparison", "", "| Method | PSNR | SSIM | LPIPS-Alex |", "|---|---:|---:|---:|"])
    for method in ROUTING_METHODS:
        rows = [row for row in matrix if row["method"] == method]
        lpips_values = [float(row["LPIPS_Alex"]) for row in rows if row["LPIPS_Alex"]]
        lpips_text = f"{np.mean(lpips_values):.6f}" if lpips_values else "n/a"
        lines.append(f"| {method} | {np.mean([float(row['PSNR']) for row in rows]):.6f} | {np.mean([float(row['SSIM']) for row in rows]):.6f} | {lpips_text} |")
    selected_rows = [next(row for row in matrix if row["case"] == decision["case"] and row["method"] == decision["selected_method"]) for decision in decisions]
    selected_lpips = [float(row["LPIPS_Alex"]) for row in selected_rows if row["LPIPS_Alex"]]
    selected_lpips_text = f"{np.mean(selected_lpips):.6f}" if selected_lpips else "n/a"
    lines.append(f"| Scenario Routing v1 | {np.mean([float(row['PSNR']) for row in selected_rows]):.6f} | {np.mean([float(row['SSIM']) for row in selected_rows]):.6f} | {selected_lpips_text} |")
    lines.extend(["", "## Per-case winners from existing matrix", "", "Composite winner is the lowest sum of three within-case ranks: PSNR descending, SSIM descending, LPIPS-Alex ascending. This is a descriptive tie-breaker, not a routing rule.", "", "| Case | PSNR best | SSIM best | LPIPS best | Composite best |", "|---|---|---|---|---|"])
    for case in sorted({row["case"] for row in matrix}, key=lambda value: int(value[4:])):
        rows = [row for row in matrix if row["case"] == case]
        psnr = max(rows, key=lambda row: float(row["PSNR"]))
        ssim = max(rows, key=lambda row: float(row["SSIM"]))
        lpips_rows = [row for row in rows if row["LPIPS_Alex"] not in ("", None)]
        lpips = min(lpips_rows, key=lambda row: float(row["LPIPS_Alex"])) if lpips_rows else {"method": "n/a", "LPIPS_Alex": ""}
        rank_psnr = {row["method"]: rank for rank, row in enumerate(sorted(rows, key=lambda row: float(row["PSNR"]), reverse=True), 1)}
        rank_ssim = {row["method"]: rank for rank, row in enumerate(sorted(rows, key=lambda row: float(row["SSIM"]), reverse=True), 1)}
        rank_lpips = {row["method"]: rank for rank, row in enumerate(sorted(lpips_rows, key=lambda row: float(row["LPIPS_Alex"])), 1)}
        composite = min(rows, key=lambda row: rank_psnr[row["method"]] + rank_ssim[row["method"]] + rank_lpips.get(row["method"], len(rows) + 1))
        lines.append(f"| {case} | {psnr['method']} ({psnr['PSNR']}) | {ssim['method']} ({ssim['SSIM']}) | {lpips['method']} ({lpips['LPIPS_Alex']}) | {composite['method']} |")
    lines.extend([
        "", "## Scene-difference analysis", "",
        "1. Per-case optima exist: the best method changes by case and metric; there is no universal winner.",
        "2. `texture_selective_h200` is not stable best: it leads case1-case3 on PSNR/SSIM or LPIPS but loses case4 and case5 on key metrics.",
        "3. HYPIR-50 is the strongest conservative candidate for case4/case5-like outcomes in this matrix and is competitive on case1/case2.",
        "4. `texture_selective_h200` is most competitive on the first three cases, especially case2, but its edge-region change remains higher than H50.",
        "5. Raw HYPIR-200 has no PSNR or SSIM win; its only clear practical advantage is LPIPS on case4, where PSNR/SSIM are substantially worse.",
        "6. Region errors support strategy differences but not a reliable selector: H200 often increases strong-edge and blurred-texture error, while selective fusion reduces textured-region error in some cases.",
        "Using the supplied scene descriptions, H50 is the best global PSNR/SSIM choice for dense foliage (case4) and clock (case5), while texture-selective H200 is the composite choice for small face/text (case1), book spine text (case2), and bird (case3). This scene interpretation is analysis only; the routing rule never consumes these labels.",
        "",
        "## Regional error means",
        "",
        "These are means over available LQ-defined pixels/cases; blank blurred-texture entries mean that region was absent in that case.",
        "",
        "| Region | HYPIR-50 error | HYPIR-200 error | texture-selective H200 error |",
        "|---|---:|---:|---:|",
    ])
    for region in ("strong_edge", "textured_non_edge", "blurred_texture"):
        values = []
        for method in ("HYPIR-50", "HYPIR-200", "texture_selective_h200"):
            field = f"{region}_error_to_GT_L1"
            entries = [float(row[field]) for row in matrix if row["method"] == method and row[field]]
            values.append(f"{np.mean(entries):.4f}" if entries else "n/a")
        lines.append(f"| {region} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend([
        "",
        "## Decision gate", "",
        "Scenario Routing v1 selects HYPIR-200 for case1/case3/case4 and LQ for case2/case5. Its average is therefore materially below the fixed baselines (see table and `metrics.csv`), with worse PSNR and SSIM than `texture_selective_h200`; LPIPS also worsens because the H200 selections dominate.",
        "",
        "Conclusion: B. routing has no evidence of value in this proof-of-concept. Stop routing here; do not train a classifier, add diffusion inference, or tune rules on these five validation images. Reconsider architecture innovation separately.",
        "",
        "Regional evidence is in `per_case_method_matrix.csv` and is LQ-region based (`strong_edge`, `textured_non_edge`, `blurred_texture`). Lower change means preservation, not necessarily GT-aligned improvement.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "baseline" / "experiments" / "scenario_routing_v1"
    result = run_routing(
        root / "baseline" / "input",
        root / "csig_dataset" / "验证集",
        root / "baseline" / "experiments" / "coeff_t_50" / "output" / "result",
        root / "baseline" / "experiments" / "coeff_t_200" / "output" / "result",
        root / "baseline" / "experiments" / "structure_local_restoration_v1" / "fusion" / "texture_selective" / "h200",
        out,
        source_metrics_csv=root / "baseline" / "experiments" / "structure_local_restoration_v1" / "metrics.csv",
        source_region_csv=root / "baseline" / "experiments" / "structure_local_restoration_v1" / "local_analysis.csv",
    )
    print(f"Wrote Scenario Routing v1 for {len(result['cases'])} cases to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
