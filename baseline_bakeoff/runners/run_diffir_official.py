"""Run DiffIRS2 from the official test yaml, with Hann tiling after full-frame OOM."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from baseline_bakeoff.runners.run_diffir import (
    discover_inputs,
    load_lq_tensor,
    make_run_manifest,
    run_with_oom_retry,
    sha256_file,
    tensor_to_rgb_image,
)

DEFAULT_SOURCE_COMMIT = "293f86cdf313914ea0ffb2457ed032e4f1bd9dd2"
CASE_IDS = ("case1", "case2", "case3", "case4", "case5")


def load_yaml(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def stage_csig_val_pairs(src_dir: Path | str, dest_root: Path | str) -> dict[str, Path]:
    """Copy caseN_lq/gt to ASCII folders named caseN.jpg so DeblurPairedDataset can pair them."""

    source = Path(src_dir)
    dest = Path(dest_root)
    lq_dir = dest / "lq"
    gt_dir = dest / "gt"
    lq_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    for case in CASE_IDS:
        lq_src = source / f"{case}_lq.jpg"
        gt_src = source / f"{case}_gt.jpg"
        if not lq_src.is_file():
            raise FileNotFoundError(f"Missing LQ: {lq_src}")
        if not gt_src.is_file():
            raise FileNotFoundError(f"Missing GT: {gt_src}")
        lq_dest = lq_dir / f"{case}.jpg"
        gt_dest = gt_dir / f"{case}.jpg"
        shutil.copy2(lq_src, lq_dest)
        shutil.copy2(gt_src, gt_dest)
        staged[case] = lq_dest
    return staged


def network_kwargs(opt: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(opt["network_g"])
    kwargs.pop("type", None)
    return kwargs


def build_model_from_yaml(opt: dict[str, Any], diffir_root: Path, device: torch.device) -> torch.nn.Module:
    source_root = diffir_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"DiffIR source directory does not exist: {source_root}")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from DiffIR.archs.S2_arch import DiffIRS2

    checkpoint = Path(opt["path"]["pretrain_network_g"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DiffIR checkpoint does not exist: {checkpoint}")
    model = DiffIRS2(**network_kwargs(opt))
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    param_key = opt["path"].get("param_key_g", "params_ema")
    strict = bool(opt["path"].get("strict_load_g", True))
    model.load_state_dict(checkpoint_data[param_key], strict=strict)
    return model.to(device).eval()


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--diffir-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, default=None)
    parser.add_argument("--source-commit", default=DEFAULT_SOURCE_COMMIT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overlap", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    opt = load_yaml(args.opt)
    seed = int(opt.get("manual_seed", 0))
    device = _device_from_arg(args.device)
    if args.stage_dir is not None:
        stage_csig_val_pairs(args.stage_dir, Path(opt["datasets"]["val"]["dataroot_lq"]).parent)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_model_from_yaml(opt, args.diffir_root, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    overlap = args.overlap
    for path in discover_inputs(args.input_dir):
        lq_cpu = load_lq_tensor(path)
        lq = lq_cpu.to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        output, retry = run_with_oom_retry(model, lq, overlap=overlap)
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
    checkpoint = Path(opt["path"]["pretrain_network_g"])
    manifest = make_run_manifest(
        checkpoint=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        source_commit=args.source_commit,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config={
            "opt": str(args.opt),
            "timesteps": opt["network_g"]["timesteps"],
            "tile": None,
            "overlap": overlap,
            "param_key": opt["path"].get("param_key_g", "params_ema"),
            "official_test_py": "DiffIR/test.py full-frame first; Hann tile fallback after OOM",
        },
        environment={"torch": torch.__version__, "device": str(device)},
        seed=seed,
    )
    manifest["model"] = "DiffIR Motion Deblurring (official yaml)"
    manifest["cases"] = cases
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
