"""Run HYPIR output-space fusion on existing LQ / H50 / H200 images.

No diffusion inference, no HYPIR source edits, no new models.
Default scope is validation case1-case5.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
METHODS = ("LQ", "HYPIR-50", "HYPIR-200", "fusion_A", "fusion_B")
_CASE_RE = re.compile(r"^(case\d+)", re.IGNORECASE)

# Known hallucination / false-texture patches from error_decomposition_v1.
DIAGNOSTIC_CROPS: dict[str, list[tuple[str, int, int, int, int]]] = {
    "case3": [
        ("low02_water", 1536, 256, 1792, 512),
        ("mid02_mud", 512, 512, 768, 768),
    ],
    "case4": [
        ("mid04_fish", 2816, 1024, 3072, 1280),
        ("high02_leaf", 3072, 768, 3328, 1024),
    ],
}


def load_rgb(path: Path | str) -> np.ndarray:
    path = Path(path)
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"{path}: expected RGB, got {image.mode}")
        array = np.asarray(image, dtype=np.uint8).copy()
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{path}: expected HxWx3 image, got {array.shape}")
    return array


def save_rgb(path: Path, image: np.ndarray) -> None:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{path}: expected HxWx3 output, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: non-finite output")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8)).save(path, format="PNG")


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    )


def _case_key(stem: str) -> str:
    match = _CASE_RE.match(stem)
    return match.group(1).casefold() if match else stem.casefold()


def _case_names(directory: Path) -> list[str]:
    names = sorted({_case_key(path.stem) for path in _image_files(directory)})
    return names


def _case_file(directory: Path, case: str, suffixes: Iterable[str] = ()) -> Path:
    expected = {case.casefold(), *(f"{case}{suffix}".casefold() for suffix in suffixes)}
    candidates = [path for path in _image_files(directory) if path.stem.casefold() in expected]
    if not candidates:
        raise FileNotFoundError(f"{case}: no image in {directory}")
    return sorted(candidates, key=lambda path: path.name.casefold())[0]


def _colorize(values: np.ndarray, vmax: float | None = None, colormap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    peak = float(vmax) if vmax is not None else float(array.max() if array.size else 1.0)
    if peak <= 1e-8:
        scaled = np.zeros_like(array, dtype=np.uint8)
    else:
        scaled = np.clip(np.rint(array / peak * 255.0), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, colormap)[:, :, ::-1]


def _panel(path: Path, arrays: list[tuple[str, np.ndarray]], max_width: int = 1024) -> None:
    scale = min(1.0, max_width / max(image.shape[1] for _, image in arrays))
    thumbs: list[tuple[str, Image.Image]] = []
    for name, image in arrays:
        height, width = image.shape[:2]
        resized = Image.fromarray(np.clip(np.rint(image), 0, 255).astype(np.uint8)).resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
        thumbs.append((name, resized))
    gap, header = 8, 38
    canvas = Image.new(
        "RGB",
        (sum(image.width for _, image in thumbs) + gap * (len(thumbs) - 1), header + max(image.height for _, image in thumbs)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    x = 0
    for name, image in thumbs:
        draw.text((x + 3, 8), name, fill=(0, 0, 0))
        canvas.paste(image, (x, header))
        x += image.width + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")


def _metric(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred_u8 = np.clip(np.rint(pred), 0, 255).astype(np.uint8)
    gt_u8 = np.clip(np.rint(gt), 0, 255).astype(np.uint8)
    if np.array_equal(pred_u8, gt_u8):
        return float("inf"), 1.0
    return (
        float(peak_signal_noise_ratio(gt_u8, pred_u8, data_range=255)),
        float(structural_similarity(gt_u8, pred_u8, channel_axis=2, data_range=255)),
    )


def _lpips_scores(images: dict[str, np.ndarray], gt: np.ndarray, model, device, max_side: int) -> dict[str, float]:
    import torch
    from torchvision.transforms.functional import pil_to_tensor

    def tensor(array: np.ndarray):
        image = Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8))
        scale = min(1.0, max_side / max(image.size))
        if scale < 1.0:
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.BILINEAR,
            )
        return pil_to_tensor(image).unsqueeze(0).to(device=device, dtype=torch.float32).div(127.5).sub(1.0)

    names = list(images)
    with torch.inference_mode():
        pred = torch.cat([tensor(images[name]) for name in names], dim=0)
        target = tensor(gt).expand(len(names), -1, -1, -1)
        values = model(pred, target).reshape(-1).detach().cpu().tolist()
    return {name: float(value) for name, value in zip(names, values)}


def _residual_l1(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32)), axis=2)


def _write_summary(
    out: Path,
    metric_rows: list[dict[str, object]],
    case_info: list[dict[str, object]],
    lpips_enabled: bool,
) -> None:
    def avg(method: str, field: str) -> float:
        values = [
            float(row[field])
            for row in metric_rows
            if row["method"] == method and row["case"] != "Average" and row[field] not in ("", None)
        ]
        return float(np.mean(values)) if values else float("nan")

    methods = ["LQ", "HYPIR-50", "HYPIR-200", "fusion_A", "fusion_B"]
    if any(row["method"] == "texture_selective_h200" for row in metric_rows):
        methods.append("texture_selective_h200")

    header = "| method | PSNR | SSIM | LPIPS_1024 |"
    sep = "|---|---:|---:|---:|"
    avg_lines = [header, sep]
    for method in methods:
        lp = avg(method, "LPIPS_Alex")
        lp_s = "" if math.isnan(lp) else f"{lp:.6f}"
        avg_lines.append(f"| {method} | {avg(method, 'PSNR'):.6f} | {avg(method, 'SSIM'):.6f} | {lp_s} |")

    case_lines = [
        "| case | scene | alpha | mean_mask | fusion_A PSNR | fusion_B PSNR | H50 PSNR | H200 PSNR | LQ PSNR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cases = [row["case"] for row in case_info]
    for case in cases:
        info = next(item for item in case_info if item["case"] == case)

        def psnr(method: str) -> str:
            row = next((item for item in metric_rows if item["case"] == case and item["method"] == method), None)
            return "" if row is None else f"{float(row['PSNR']):.4f}"

        case_lines.append(
            f"| {case} | {info['scene']} | {float(info['alpha']):.2f} | {float(info['mean_mask']):.4f} | "
            f"{psnr('fusion_A')} | {psnr('fusion_B')} | {psnr('HYPIR-50')} | {psnr('HYPIR-200')} | {psnr('LQ')} |"
        )

    lines = [
        "# HYPIR fusion v1 summary",
        "",
        "Offline output-space fusion. H200 is a detail candidate; LQ/H50 are structure anchors;",
        "the Sobel structure mask suppresses contours that H200 invents.",
        "",
        f"- date: {date.today().isoformat()}",
        f"- LPIPS: Alex max-side 1024, computed={lpips_enabled}",
        "- scheme A: `Y = Y_LQ + α·mask·(Y_H200 - Y_LQ)`",
        "- scheme B: `Y = Y_H50 + α·mask·(Y_H200 - Y_H50)`",
        "- chroma: LQ Cb/Cr for both schemes",
        "",
        "## Average metrics",
        "",
        *avg_lines,
        "",
        "## Per-case",
        "",
        *case_lines,
        "",
        "## Visual checklist",
        "",
        "- case4 `crops/case4_mid04_fish.png`: Fusion attenuates H200 (not detect-and-delete); fish-head should collapse toward a pink blur",
        "- case4 `crops/case4_high02_leaf.png`: new serrated edges should be mask-suppressed; GT compound leaves are not recovered",
        "- case3 `crops/case3_low02_water.png`: invented water grain should fade toward LQ because bird alpha is small",
        "- case1/2/5 comparison panels: text, spine, clock hands must not be redrawn",
        "",
        "## How to read the mask",
        "",
        "Yellow/green in `masks/` = H200 structure agrees with LQ (detail allowed).",
        "Dark in `heatmaps/*_suppressed.png` = H200 residual that the mask blocked.",
        "",
        "## Not in this experiment",
        "",
        "Degradation Encoder, Adapter, LoRA, diffusion finetune, ControlNet, Global Attention.",
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
    compute_lpips: bool = False,
    lpips_size: int = 1024,
) -> dict[str, object]:
    lq_dir, h50_dir, h200_dir, out = map(Path, (lq_dir, h50_dir, h200_dir, out_dir))
    gt_path = Path(gt_dir) if gt_dir is not None else None
    texture_path = Path(texture_dir) if texture_dir is not None else None
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("fusion/A", "fusion/B", "masks", "heatmaps", "comparison", "crops"):
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
        mask = compute_structure_mask(rgb_to_y(lq), rgb_to_y(h200), blur_sigma=blur_sigma)
        outputs: dict[str, np.ndarray] = {"LQ": lq, "HYPIR-50": h50, "HYPIR-200": h200}
        if "A" in schemes:
            fused_a = fuse_scheme_a(lq, h200, mask, alpha)
            save_rgb(out / "fusion" / "A" / f"{case}.png", fused_a)
            outputs["fusion_A"] = fused_a
        if "B" in schemes:
            fused_b = fuse_scheme_b(lq, h50, h200, mask, alpha)
            save_rgb(out / "fusion" / "B" / f"{case}.png", fused_b)
            outputs["fusion_B"] = fused_b
        if texture is not None:
            outputs["texture_selective_h200"] = texture

        residual = _residual_l1(h200, lq)
        suppressed = (1.0 - mask) * residual
        save_rgb(out / "masks" / f"{case}.png", _colorize(mask, vmax=1.0, colormap=cv2.COLORMAP_VIRIDIS))
        Image.fromarray(np.clip(np.rint(mask * 255.0), 0, 255).astype(np.uint8)).save(
            out / "masks" / f"{case}_gray.png", format="PNG"
        )
        applied = np.float32(alpha) * mask
        save_rgb(out / "masks" / f"{case}_applied.png", _colorize(applied, vmax=max(DEFAULT_SCENE_ALPHA.values()), colormap=cv2.COLORMAP_VIRIDIS))
        save_rgb(out / "heatmaps" / f"{case}_residual.png", _colorize(residual, colormap=cv2.COLORMAP_INFERNO))
        save_rgb(out / "heatmaps" / f"{case}_suppressed.png", _colorize(suppressed, colormap=cv2.COLORMAP_INFERNO))

        panel = [("LQ", lq), ("H50", h50), ("H200", h200)]
        if "fusion_A" in outputs:
            panel.append(("Fusion-A", outputs["fusion_A"]))
        if "fusion_B" in outputs:
            panel.append(("Fusion-B", outputs["fusion_B"]))
        if gt is not None:
            panel.append(("GT", gt))
        _panel(out / "comparison" / f"{case}.png", panel, max_width=768)
        evidence = [
            ("mask", _colorize(mask, vmax=1.0)),
            ("|H200-LQ|", _colorize(residual)),
            ("suppressed", _colorize(suppressed)),
        ]
        if "fusion_A" in outputs:
            evidence.append(("Fusion-A", outputs["fusion_A"]))
        if "fusion_B" in outputs:
            evidence.append(("Fusion-B", outputs["fusion_B"]))
        _panel(out / "comparison" / f"{case}_evidence.png", evidence, max_width=768)

        for crop_name, x0, y0, x1, y1 in DIAGNOSTIC_CROPS.get(case, ()):
            if y1 > lq.shape[0] or x1 > lq.shape[1]:
                continue
            crop_panel = [(name, image[y0:y1, x0:x1]) for name, image in panel]
            _panel(out / "crops" / f"{case}_{crop_name}.png", crop_panel, max_width=256)

        mean_mask = float(mask.mean())
        suppressed_frac = float((mask < 0.3).mean())
        case_info.append(
            {
                "case": case,
                "scene": scene,
                "alpha": alpha,
                "mean_mask": mean_mask,
                "suppressed_frac": suppressed_frac,
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
                    "LPIPS_Alex": "" if math.isnan(lp) else f"{lp:.6f}",
                    "Delta_PSNR_vs_LQ": f"{psnr - lq_psnr:.6f}",
                    "Delta_SSIM_vs_LQ": f"{ssim - lq_ssim:.6f}",
                    "Delta_LPIPS_vs_LQ": "" if math.isnan(lp) else f"{lp - lq_lp:.6f}",
                    "mean_mask": f"{mean_mask:.6f}",
                    "suppressed_frac": f"{suppressed_frac:.6f}",
                }
            )
            record(method, psnr, ssim, lp)

    metric_fields = [
        "case", "scene", "alpha", "method", "PSNR", "SSIM", "LPIPS_Alex",
        "Delta_PSNR_vs_LQ", "Delta_SSIM_vs_LQ", "Delta_LPIPS_vs_LQ",
        "mean_mask", "suppressed_frac",
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
                        "mean_mask": "",
                        "suppressed_frac": "",
                    }
                )

    (out / "experiment_metadata.md").write_text(
        "\n".join(
            [
                "# HYPIR fusion v1 metadata",
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
                f"- blur_sigma: {blur_sigma}",
                "- formula: `out_Y = base_Y + alpha * mask * (H200_Y - base_Y)`",
                "- mask: 1/4 Sobel cosine * magnitude similarity, Gaussian blur, upsample",
                "- chroma: LQ YCrCb Cr/Cb",
                "- inference: none; existing HYPIR PNGs only",
                f"- lpips: Alex max-side {lpips_size}, enabled={compute_lpips}",
                "",
                "## Scene notes",
                "",
                *[f"- {name}: {note}" for name, note in SCENE_NOTES.items()],
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_summary(out, metric_rows, case_info, compute_lpips)
    return {"cases": selected, "metrics": metric_rows, "output_dir": out, "case_info": case_info}


def _resolve_dir(explicit: Path | None, local: Path, fallback: Path) -> Path:
    if explicit is not None:
        return explicit
    if local.is_dir() and _image_files(local):
        return local
    return fallback


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
    out_dir = args.output_dir or (experiment_dir / "results" / "fusion_v1")
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
        compute_lpips=args.compute_lpips,
        lpips_size=args.lpips_size,
    )
    print(f"Wrote {len(result['cases'])} cases to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
