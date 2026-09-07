"""Run the official DiffIR Motion Deblurring S2 checkpoint using LQ images only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


def _case_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem.casefold().removesuffix("_lq")
    if stem.startswith("case") and stem[4:].isdigit():
        return (int(stem[4:]), path.name.casefold())
    return (sys.maxsize, path.name.casefold())


def discover_inputs(input_dir: Path | str) -> list[Path]:
    """Return readable `*_lq` images in case order, based on decoded content."""

    directory = Path(input_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")
    inputs: list[Path] = []
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        if not path.stem.casefold().endswith("_lq"):
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, Image.UnidentifiedImageError) as exc:
            raise ValueError(f"Unreadable LQ image: {path}") from exc
        inputs.append(path)
    if not inputs:
        raise FileNotFoundError(f"No readable *_lq images found in {directory}")
    return sorted(inputs, key=_case_sort_key)


def load_lq_tensor(path: Path | str) -> torch.Tensor:
    """Decode an LQ image as RGB without relying on its filename extension."""

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div(255.0)


def infer_lq(model: Callable[[torch.Tensor], torch.Tensor], lq: torch.Tensor) -> torch.Tensor:
    """Run a model with LQ alone; GT is deliberately absent from this boundary."""

    if lq.ndim == 3:
        lq = lq.unsqueeze(0)
    if lq.ndim != 4 or lq.shape[1] != 3:
        raise ValueError(f"Expected an RGB BCHW tensor, received shape={tuple(lq.shape)}")
    with torch.inference_mode():
        output = model(lq)
    if not isinstance(output, torch.Tensor):
        raise TypeError("DiffIR inference must return a tensor in evaluation mode")
    return output


def tensor_to_rgb_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a one-image RGB tensor in the [0, 1] range to a Pillow RGB image."""

    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"Expected a single output image, received batch={tensor.shape[0]}")
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected an RGB CHW tensor, received shape={tuple(tensor.shape)}")
    array = tensor.detach().float().cpu().clamp_(0.0, 1.0).mul_(255.0).round().byte()
    return Image.fromarray(array.permute(1, 2, 0).numpy())


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_run_manifest(
    *,
    checkpoint: Path | str,
    checkpoint_sha256: str,
    source_commit: str,
    input_dir: Path | str,
    output_dir: Path | str,
    config: dict[str, Any],
    environment: dict[str, str],
    seed: int = 0,
) -> dict[str, Any]:
    """Describe the LQ-only inference contract without recording any GT source."""

    return {
        "model": "DiffIR Motion Deblurring",
        "stage": "DiffIRS2",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "source_commit": source_commit,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "config": config,
        "environment": environment,
        "seed": seed,
        "gt_used_during_inference": False,
    }


def _build_model(checkpoint: Path, diffir_root: Path, device: torch.device) -> torch.nn.Module:
    source_root = diffir_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"DiffIR source directory does not exist: {source_root}")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from DiffIR.archs.S2_arch import DiffIRS2

    model = DiffIRS2(
        n_encoder_res=5,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[3, 5, 6, 6],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type="WithBias",
        n_denoise_res=1,
        linear_start=0.1,
        linear_end=0.99,
        timesteps=4,
    )
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_data["params_ema"], strict=True)
    return model.to(device).eval()


def _pad_to_multiple(lq: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = lq.shape[-2:]
    pad_height = (-height) % multiple
    pad_width = (-width) % multiple
    if not pad_height and not pad_width:
        return lq, (height, width)
    padded = torch.nn.functional.pad(lq, (0, pad_width, 0, pad_height), mode="reflect")
    return padded, (height, width)


def _infer_tiled(model: torch.nn.Module, lq: torch.Tensor, tile: int, overlap: int) -> torch.Tensor:
    if tile <= overlap or overlap < 0:
        raise ValueError("tile must be greater than a non-negative overlap")
    _, channels, height, width = lq.shape
    stride = tile - overlap
    output = torch.zeros((1, channels, height, width), device=lq.device, dtype=lq.dtype)
    weights = torch.zeros((1, 1, height, width), device=lq.device, dtype=lq.dtype)
    window_1d = torch.hann_window(tile, periodic=False, device=lq.device, dtype=lq.dtype).clamp_min_(1e-3)
    window = window_1d.outer(window_1d).view(1, 1, tile, tile)
    def starts(length: int) -> list[int]:
        final = length - tile
        result = list(range(0, final + 1, stride))
        if result[-1] != final:
            result.append(final)
        return result

    for top in starts(height):
        for left in starts(width):
            patch = lq[:, :, top : top + tile, left : left + tile]
            restored = infer_lq(model, patch)
            output[:, :, top : top + tile, left : left + tile] += restored * window
            weights[:, :, top : top + tile, left : left + tile] += window
    return output / weights


def run_image(model: torch.nn.Module, lq: torch.Tensor, *, tile: int | None = None, overlap: int = 128) -> torch.Tensor:
    """Run one image, preserving its dimensions after optional OOM fallback tiling."""

    if lq.ndim == 3:
        lq = lq.unsqueeze(0)
    if lq.ndim != 4 or lq.shape[1] != 3:
        raise ValueError(f"Expected an RGB BCHW tensor, received shape={tuple(lq.shape)}")
    original_height, original_width = lq.shape[-2:]
    padded, _ = _pad_to_multiple(lq)
    if tile is None:
        restored = infer_lq(model, padded)
    else:
        tile = min(tile, padded.shape[-2], padded.shape[-1])
        if tile <= overlap:
            raise ValueError("Requested tile is too small for the configured overlap")
        restored = _infer_tiled(model, padded, tile, overlap)
    return restored[:, :, :original_height, :original_width]


def run_with_oom_retry(
    model: torch.nn.Module,
    lq: torch.Tensor,
    *,
    overlap: int = 128,
    tile_candidates: tuple[int, ...] = (512, 384, 256),
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Try full-frame inference, then deterministic weighted tiles on OOM."""

    try:
        output = run_image(model, lq, tile=None, overlap=overlap)
    except RuntimeError as exc:
        full_frame_error = str(exc)
    else:
        return output, {
            "oom_retry": False,
            "tile": None,
            "overlap": None,
        }
    if "out of memory" not in full_frame_error.casefold():
        raise RuntimeError(full_frame_error)
    # Exit the exception handler before clearing: retaining the traceback can retain 4K activations.
    gc.collect()
    if lq.device.type == "cuda":
        torch.cuda.empty_cache()
    last_oom_error = full_frame_error
    for tile in tile_candidates:
        if min(lq.shape[-2:]) <= overlap:
            break
        try:
            output = run_image(model, lq, tile=tile, overlap=overlap)
        except RuntimeError as exc:
            tile_error = str(exc)
        else:
            return output, {
                "oom_retry": True,
                "tile": tile,
                "overlap": overlap,
            }
        if "out of memory" not in tile_error.casefold():
            raise RuntimeError(tile_error)
        last_oom_error = tile_error
        gc.collect()
        if lq.device.type == "cuda":
            torch.cuda.empty_cache()
    raise RuntimeError(last_oom_error)


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--diffir-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-commit", default="293f86cdf313914ea0ffb2457ed032e4f1bd9dd2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0, help="Fixed seed for DiffIR's sampled IPR initialization.")
    parser.add_argument("--tile", type=int, default=0, help="Optional tile size; only use after an OOM retry.")
    parser.add_argument("--overlap", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device = _device_from_arg(args.device)
    tile = args.tile or None
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"DiffIR checkpoint does not exist: {args.checkpoint}")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = _build_model(args.checkpoint, args.diffir_root, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for path in discover_inputs(args.input_dir):
        lq_cpu = load_lq_tensor(path)
        lq = lq_cpu.to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        output, retry = run_with_oom_retry(model, lq, overlap=args.overlap, tile_candidates=((tile,) if tile else (512, 384, 256)))
        elapsed = time.perf_counter() - started
        image = tensor_to_rgb_image(output)
        if image.size != (lq_cpu.shape[2], lq_cpu.shape[1]):
            raise RuntimeError(f"DiffIR changed dimensions for {path.name}: {image.size}")
        output_path = args.output_dir / f"{path.stem}.png"
        image.save(output_path, format="PNG")
        cases.append(
            {
                "case": path.stem.removesuffix("_lq"),
                "input": str(path),
                "output": str(output_path),
                "runtime_sec": elapsed,
                "peak_vram_gb": (torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else None,
                "tile": retry["tile"],
                "overlap": retry["overlap"],
                "oom_retry": retry["oom_retry"],
            }
        )
    manifest = make_run_manifest(
        checkpoint=args.checkpoint,
        checkpoint_sha256=sha256_file(args.checkpoint),
        source_commit=args.source_commit,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config={"timesteps": 4, "tile": tile, "overlap": args.overlap},
        environment={"torch": torch.__version__, "device": str(device)},
        seed=args.seed,
    )
    manifest["cases"] = cases
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
