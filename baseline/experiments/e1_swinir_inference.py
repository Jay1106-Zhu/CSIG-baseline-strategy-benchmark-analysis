"""E1 SwinIR color JPEG artifact-reduction fidelity baseline.

The wrapper follows the official SwinIR color_jpeg_car model definition and
official tiled averaging behavior, while using the project's E0 metrics.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

# Keep direct ``python baseline/experiments/e1_swinir_inference.py`` execution
# equivalent to module execution by making the project root importable.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from basicsr.archs.swinir_arch import SwinIR
from control_conditioned_v1.control_conditioned_hypir_v1 import _lpips_scores, _metric


MODEL_CONFIG = {
    "upscale": 1,
    "in_chans": 3,
    "img_size": 126,
    "window_size": 7,
    "img_range": 255.0,
    "depths": [6, 6, 6, 6, 6, 6],
    "embed_dim": 180,
    "num_heads": [6, 6, 6, 6, 6, 6],
    "mlp_ratio": 2,
    "upsampler": "",
    "resi_connection": "1conv",
}
MODEL_NAME = "SwinIR-M color JPEG CAR JPEG40"
DEFAULT_TILE = 504
DEFAULT_OVERLAP = 32
WINDOW_SIZE = 7


def validate_tiling(tile: int, overlap: int, window_size: int = WINDOW_SIZE) -> tuple[int, int, int]:
    if tile <= 0:
        raise ValueError("tile must be positive")
    if window_size <= 0 or tile % window_size != 0:
        raise ValueError(f"tile must be a multiple of window_size ({window_size})")
    if overlap < 0 or overlap >= tile:
        raise ValueError("overlap must satisfy 0 <= overlap < tile")
    return tile, overlap, tile - overlap


def pad_to_window(image: torch.Tensor, window_size: int = WINDOW_SIZE) -> tuple[torch.Tensor, tuple[int, int]]:
    if image.ndim != 4:
        raise ValueError("image must be NCHW")
    height, width = image.shape[-2:]
    h_pad = (window_size - height % window_size) % window_size
    w_pad = (window_size - width % window_size) % window_size
    padded = image
    if h_pad:
        padded = torch.cat([padded, torch.flip(padded, [2])], dim=2)[..., :height + h_pad, :]
    if w_pad:
        padded = torch.cat([padded, torch.flip(padded, [3])], dim=3)[..., :, :width + w_pad]
    return padded, (height, width)


def _tile_indices(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    values = list(range(0, length - tile, stride))
    final = length - tile
    if not values or values[-1] != final:
        values.append(final)
    return values


def tiled_forward(
    image: torch.Tensor,
    model: torch.nn.Module,
    *,
    tile: int = DEFAULT_TILE,
    overlap: int = DEFAULT_OVERLAP,
    window_size: int = WINDOW_SIZE,
) -> torch.Tensor:
    """Run same-resolution tiled inference with overlap averaging."""

    tile, overlap, stride = validate_tiling(tile, overlap, window_size)
    padded, original = pad_to_window(image, window_size)
    batch, channels, height, width = padded.shape
    effective_tile = min(tile, height, width)
    if effective_tile != tile:
        if effective_tile % window_size != 0:
            raise ValueError("padded image is smaller than tile and not window aligned")
        tile = effective_tile
        stride = tile - min(overlap, tile - 1)
    h_indices = _tile_indices(height, tile, stride)
    w_indices = _tile_indices(width, tile, stride)
    estimate = torch.zeros((batch, channels, height, width), dtype=padded.dtype, device=padded.device)
    weights = torch.zeros_like(estimate)
    with torch.inference_mode():
        for h0 in h_indices:
            for w0 in w_indices:
                patch = padded[..., h0:h0 + tile, w0:w0 + tile]
                prediction = model(patch)
                if prediction.shape != patch.shape:
                    raise ValueError(f"model changed tile shape {tuple(patch.shape)} -> {tuple(prediction.shape)}")
                estimate[..., h0:h0 + tile, w0:w0 + tile].add_(prediction)
                weights[..., h0:h0 + tile, w0:w0 + tile].add_(1)
    output = estimate / weights.clamp_min(1)
    return output[..., :original[0], :original[1]]


def save_rgb_png(path: Path | str, array: np.ndarray) -> None:
    value = np.asarray(array)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"expected HxWx3 output, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("output contains non-finite values")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(value), 0, 255).astype(np.uint8)).save(path, format="PNG")


def image_to_model_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert an RGB uint8 image to the official SwinIR [0, 1] tensor input."""
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 HxWx3 image, got {value.shape} {value.dtype}")
    return torch.from_numpy(value.transpose(2, 0, 1)).unsqueeze(0).to(device, dtype=torch.float32).div(255.0)


def model_tensor_to_image(output_tensor: torch.Tensor) -> np.ndarray:
    """Convert SwinIR's same-resolution [0, 1] output to an HxWx3 float image."""
    if output_tensor.ndim != 4 or output_tensor.shape[0] != 1 or output_tensor.shape[1] != 3:
        raise ValueError(f"expected 1x3xHxW output, got {tuple(output_tensor.shape)}")
    return output_tensor.squeeze(0).permute(1, 2, 0).detach().float().cpu().numpy() * 255.0


def load_rgb(path: Path | str) -> tuple[np.ndarray, str]:
    path = Path(path)
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"{path}: expected RGB, got {image.mode}")
        return np.asarray(image, dtype=np.uint8).copy(), str(image.format)


def _case_file(directory: Path, case: str, suffixes: Iterable[str] = ()) -> Path:
    candidates = []
    for path in directory.iterdir() if directory.is_dir() else ():
        if not path.is_file() or path.suffix.casefold() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            continue
        stem = path.stem.casefold()
        if stem == case.casefold() or any(stem == f"{case}{suffix}".casefold() for suffix in suffixes):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"{case}: no image in {directory}")
    return sorted(candidates, key=lambda item: item.name.casefold())[0]


def _case_names(directory: Path) -> list[str]:
    cases = []
    for path in directory.iterdir() if directory.is_dir() else ():
        if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            stem = path.stem.casefold()
            cases.append(stem[:-3] if stem.endswith("_lq") else stem)
    return sorted(set(cases), key=lambda value: (0, int(value[4:])) if value.startswith("case") and value[4:].isdigit() else (1, value))


def load_model(checkpoint: Path | str, device: torch.device) -> SwinIR:
    model = SwinIR(**MODEL_CONFIG)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload["params"] if isinstance(payload, dict) and "params" in payload else payload
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(device)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_e1(
    *,
    checkpoint: Path,
    input_dir: Path,
    gt_dir: Path,
    h50_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    tile: int = DEFAULT_TILE,
    overlap: int = DEFAULT_OVERLAP,
    lpips_size: int = 1024,
) -> dict[str, object]:
    validate_tiling(tile, overlap, WINDOW_SIZE)
    cases = _case_names(input_dir)
    expected = [f"case{i}" for i in range(1, 6)]
    if cases != expected:
        raise ValueError(f"E1 requires exactly {expected}, found {cases}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [path for path in output_dir.iterdir() if path.is_file()]
    if output_paths:
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {output_dir}")
    selected_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    _sync(selected_device)
    load_start = time.perf_counter()
    model = load_model(checkpoint, selected_device)
    _sync(selected_device)
    model_load_seconds = time.perf_counter() - load_start

    import lpips

    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(selected_device).eval()
    rows: list[dict[str, object]] = []
    peak_allocated = 0
    peak_reserved = 0
    for case in cases:
        lq, input_format = load_rgb(_case_file(input_dir, case, ("_lq",)))
        gt, _ = load_rgb(_case_file(gt_dir, case, ("_gt",)))
        h50, _ = load_rgb(_case_file(h50_dir, case, ("_lq",)))
        if len({lq.shape, gt.shape, h50.shape}) != 1:
            raise ValueError(f"{case}: input/GT/HYPIR-50 dimensions do not match")
        tensor = image_to_model_tensor(lq, selected_device)
        if selected_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(selected_device)
        _sync(selected_device)
        start = time.perf_counter()
        output_tensor = tiled_forward(tensor, model, tile=tile, overlap=overlap, window_size=WINDOW_SIZE)
        _sync(selected_device)
        elapsed = time.perf_counter() - start
        output = model_tensor_to_image(output_tensor)
        save_rgb_png(output_dir / f"{case}.png", output)
        fidelity_u8 = np.clip(np.rint(output), 0, 255).astype(np.uint8)
        h_psnr, h_ssim = _metric(h50, gt)
        f_psnr, f_ssim = _metric(fidelity_u8, gt)
        lpips_values = _lpips_scores({"HYPIR-50": h50, "Fidelity": fidelity_u8}, gt, lpips_model, selected_device, lpips_size)
        h_lpips, f_lpips = lpips_values["HYPIR-50"], lpips_values["Fidelity"]
        if selected_device.type == "cuda":
            allocated = torch.cuda.max_memory_allocated(selected_device)
            reserved = torch.cuda.max_memory_reserved(selected_device)
            peak_allocated = max(peak_allocated, allocated)
            peak_reserved = max(peak_reserved, reserved)
        else:
            allocated = reserved = 0
        rows.append({
            "case": case,
            "HYPIR-50_PSNR": f"{h_psnr:.6f}", "Fidelity_PSNR": f"{f_psnr:.6f}", "Delta_PSNR": f"{f_psnr - h_psnr:.6f}",
            "HYPIR-50_SSIM": f"{h_ssim:.6f}", "Fidelity_SSIM": f"{f_ssim:.6f}", "Delta_SSIM": f"{f_ssim - h_ssim:.6f}",
            "HYPIR-50_LPIPS": f"{h_lpips:.6f}", "Fidelity_LPIPS": f"{f_lpips:.6f}", "Delta_LPIPS": f"{f_lpips - h_lpips:.6f}",
            "input_format": input_format, "inference_time_s": f"{elapsed:.6f}",
            "peak_memory_allocated_gib": f"{allocated / 1024**3:.6f}", "peak_memory_reserved_gib": f"{reserved / 1024**3:.6f}",
        })

    metric_fields = [
        "case", "HYPIR-50_PSNR", "Fidelity_PSNR", "Delta_PSNR", "HYPIR-50_SSIM", "Fidelity_SSIM", "Delta_SSIM",
        "HYPIR-50_LPIPS", "Fidelity_LPIPS", "Delta_LPIPS", "input_format", "inference_time_s",
        "peak_memory_allocated_gib", "peak_memory_reserved_gib",
    ]
    average: dict[str, object] = {"case": "Average", "input_format": "mixed"}
    for field in metric_fields[1:]:
        if field in {"input_format"}:
            continue
        values = [float(row[field]) for row in rows]
        average[field] = f"{float(np.mean(values)):.6f}"
    average["peak_memory_allocated_gib"] = f"{peak_allocated / 1024**3:.6f}"
    average["peak_memory_reserved_gib"] = f"{peak_reserved / 1024**3:.6f}"
    with (output_dir.parent / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows([*rows, average])

    metadata = output_dir.parent / "experiment_metadata.md"
    metadata.write_text(
        "# E1 SwinIR fidelity baseline metadata\n\n"
        f"- model: {MODEL_NAME}\n- checkpoint: `{checkpoint}`\n- device: `{selected_device}`\n"
        f"- model_load_seconds: {model_load_seconds:.6f}\n- tile: {tile}\n- tile_overlap: {overlap}\n- window_size: {WINDOW_SIZE}\n"
        f"- padding: reflected flip to window multiple; output crop to original size\n- cases: {', '.join(cases)}\n"
        f"- input_dir: `{input_dir}`\n- gt_dir: `{gt_dir}`\n- h50_dir: `{h50_dir}`\n- output_dir: `{output_dir}`\n"
        "- inference: no TTA, ensemble, finetuning, LoRA, SAM, or OCR\n"
        "- metrics: E0-compatible PSNR/SSIM and LPIPS-Alex (max-side 1024)\n",
        encoding="utf-8",
    )
    return {"cases": cases, "rows": rows, "average": average, "model_load_seconds": model_load_seconds, "peak_allocated": peak_allocated, "peak_reserved": peak_reserved}


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument("--h50-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE)
    parser.add_argument("--tile-overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--lpips-size", type=int, default=1024)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    base = root / "baseline" / "experiments" / "E1_fidelity" / "swinir_car_jpeg40"
    result = run_e1(
        checkpoint=(args.checkpoint or base / "checkpoint" / "006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth").resolve(),
        input_dir=(args.input_dir or root / "baseline" / "input").resolve(),
        gt_dir=(args.gt_dir or root / "csig_dataset" / "验证集").resolve(),
        h50_dir=(args.h50_dir or root / "baseline" / "experiments" / "coeff_t_50" / "output" / "result").resolve(),
        output_dir=(args.output_dir or base / "output").resolve(),
        device=args.device,
        tile=args.tile,
        overlap=args.tile_overlap,
        lpips_size=args.lpips_size,
    )
    print(f"Completed {len(result['cases'])} cases; model_load={result['model_load_seconds']:.3f}s")
    print(f"Peak allocated={result['peak_allocated'] / 1024**3:.3f} GiB; reserved={result['peak_reserved'] / 1024**3:.3f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
