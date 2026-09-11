"""Frozen texture_selective_h200 test inference wrapper.

Does not change the fusion formula. Does not import fusion_v1 scene tables.
Does not run 100 images unless --confirm-full is passed.
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baseline.experiments.structure_local_restoration import (
    compute_lq_features,
    compute_weight_map,
    fuse_image,
    load_rgb,
)


FROZEN = {
    "method": "texture_selective_h200",
    "model_t": 200,
    "coeff_t": 200,
    "upscale": 1,
    "seed": 231,
    "captioner": "empty",
    "lora_rank": 256,
    "lora_modules": "to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj",
    "patch_size": 512,
    "stride": 256,
    "scale_by": "factor",
    "base_model_type": "sd2",
    "max_weight": 0.60,
}

_CASE_RE = re.compile(r"^case(\d+)$", re.IGNORECASE)
EXPECTED_CASES = [f"case{i}" for i in range(1, 101)]


def assert_formula_frozen() -> None:
    src = inspect.getsource(compute_weight_map)
    if "0.05 + 0.46 * texture * (0.35 + 0.65 * protect)" not in src:
        raise RuntimeError("texture_selective formula changed; refusing to run")
    if "np.clip(weight, 0.0, 0.60)" not in src:
        raise RuntimeError("texture_selective clip range changed; refusing to run")
    fuse_src = inspect.getsource(fuse_image)
    if "weights[..., None] * hypir_array" not in fuse_src or "(1.0 - weights[..., None]) * lq_array" not in fuse_src:
        raise RuntimeError("RGB fusion formula changed; refusing to run")


def inventory_test_dir(test_dir: Path) -> dict[str, object]:
    test_dir = Path(test_dir)
    if not test_dir.is_dir():
        raise SystemExit(f"test dir missing: {test_dir}")
    files = sorted(
        path
        for path in test_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png"}
    )
    stems = [path.stem for path in files]
    counts: dict[str, int] = {}
    for stem in stems:
        counts[stem] = counts.get(stem, 0) + 1
    duplicate = [stem for stem, n in counts.items() if n > 1]
    missing = [name for name in EXPECTED_CASES if name not in counts]
    extra = [stem for stem in stems if stem not in EXPECTED_CASES]
    non_jpg = [path.name for path in files if path.suffix.casefold() not in {".jpg", ".jpeg"}]
    bad_mode = []
    sizes: dict[str, list[int]] = {}
    for path in files:
        with Image.open(path) as image:
            if image.mode != "RGB":
                bad_mode.append(f"{path.name}:{image.mode}")
            key = f"{image.size[0]}x{image.size[1]}"
            sizes.setdefault(key, []).append(1)
    size_summary = {key: len(value) for key, value in sizes.items()}
    report = {
        "count": len(files),
        "missing": missing,
        "duplicate": duplicate,
        "extra": extra,
        "non_jpg": non_jpg,
        "bad_mode": bad_mode,
        "sizes": size_summary,
    }
    if len(files) != 100 or missing or duplicate or extra or non_jpg or bad_mode:
        raise SystemExit(f"test inventory failed: {report}")
    return report


def build_hypir_command(python: Path, lq_dir: Path, output_dir: Path, hypir_root: Path) -> list[str]:
    if FROZEN["upscale"] != 1:
        raise RuntimeError("frozen upscale must be 1")
    return [
        str(python),
        str(hypir_root / "test.py"),
        "--base_model_type", FROZEN["base_model_type"],
        "--base_model_path", str(hypir_root / "models" / "stable-diffusion-2-1-base"),
        "--model_t", str(FROZEN["model_t"]),
        "--coeff_t", str(FROZEN["coeff_t"]),
        "--lora_rank", str(FROZEN["lora_rank"]),
        "--lora_modules", FROZEN["lora_modules"],
        "--weight_path", str(hypir_root / "weights" / "HYPIR_sd2.pth"),
        "--patch_size", str(FROZEN["patch_size"]),
        "--stride", str(FROZEN["stride"]),
        "--lq_dir", str(lq_dir),
        "--scale_by", FROZEN["scale_by"],
        "--upscale", str(FROZEN["upscale"]),
        "--captioner", FROZEN["captioner"],
        "--output_dir", str(output_dir),
        "--seed", str(FROZEN["seed"]),
        "--device", "cuda",
    ]


def save_jpg(path: Path, image: np.ndarray) -> None:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{path}: expected HxWx3")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: non-finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8)).save(
        path, format="JPEG", quality=95, subsampling=0
    )


def texture_selective_fuse(lq_path: Path, h200_path: Path, out_jpg: Path) -> dict[str, float]:
    assert_formula_frozen()
    lq = load_rgb(lq_path)
    h200 = load_rgb(h200_path)
    if lq.shape != h200.shape:
        raise ValueError(f"size mismatch LQ {lq.shape} vs H200 {h200.shape}")
    weight = compute_weight_map(compute_lq_features(lq), "texture_selective")
    if float(weight.max()) > FROZEN["max_weight"] + 1e-6:
        raise ValueError(f"weight max {weight.max()} exceeds {FROZEN['max_weight']}")
    fused = fuse_image(lq, h200, weight)
    save_jpg(out_jpg, fused)
    with Image.open(out_jpg) as written:
        if written.mode != "RGB" or written.size != (lq.shape[1], lq.shape[0]):
            raise ValueError(f"written JPG invalid: {written.mode} {written.size}")
    return {
        "min_weight": float(weight.min()),
        "max_weight": float(weight.max()),
        "mean_weight": float(weight.mean()),
    }


def _audit_image(path: Path, expected_size: tuple[int, int] | None = None) -> dict[str, object]:
    with Image.open(path) as image:
        array = np.asarray(image)
        info = {
            "path": str(path),
            "mode": image.mode,
            "size": image.size,
            "bytes": path.stat().st_size,
            "finite": bool(np.isfinite(array).all()) if array.size else False,
            "min": int(array.min()) if array.size else None,
            "max": int(array.max()) if array.size else None,
        }
    if expected_size is not None and info["size"] != expected_size:
        raise ValueError(f"{path}: size {info['size']} != {expected_size}")
    if info["mode"] != "RGB":
        raise ValueError(f"{path}: mode {info['mode']}")
    if info["bytes"] <= 0:
        raise ValueError(f"{path}: empty file")
    if not info["finite"]:
        raise ValueError(f"{path}: non-finite pixels")
    return info


def run_hypir(python: Path, lq_dir: Path, h200_dir: Path, hypir_root: Path) -> None:
    cmd = build_hypir_command(python, lq_dir, h200_dir, hypir_root)
    if cmd[cmd.index("--upscale") + 1] != "1":
        raise RuntimeError("refusing to launch HYPIR without upscale=1")
    h200_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(cmd, cwd=str(hypir_root), check=False)
    if completed.returncode != 0:
        raise SystemExit(f"HYPIR test.py failed with code {completed.returncode}")


def prepare_subset(test_dir: Path, dest: Path, cases: list[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for case in cases:
        src = test_dir / f"{case}.jpg"
        if not src.is_file():
            raise FileNotFoundError(src)
        Image.open(src).convert("RGB").save(dest / f"{case}.jpg", format="JPEG", quality=95, subsampling=0)


def dry_run(project: Path) -> dict[str, object]:
    assert_formula_frozen()
    # Test-set case1 is not the validation image of the same filename.
    test_dir = project / "csig_dataset" / "测试集"
    inventory = inventory_test_dir(test_dir)
    work = project / "baseline" / "experiments" / "final_test_inference" / "dry_run"
    lq_dir = work / "lq"
    h200_dir = work / "h200"
    out_dir = work / "output"
    if out_dir.exists():
        for path in out_dir.glob("*"):
            if path.is_file():
                path.unlink()
    prepare_subset(test_dir, lq_dir, ["case1"])
    python = project / ".conda" / "python.exe"
    hypir_root = project / "HYPIR"
    run_hypir(python, lq_dir, h200_dir, hypir_root)
    h200_png = h200_dir / "result" / "case1.png"
    lq_jpg = lq_dir / "case1.jpg"
    out_jpg = out_dir / "case1.jpg"
    with Image.open(lq_jpg) as lq_im:
        lq_size = lq_im.size
        lq_mode = lq_im.mode
    lq_info = _audit_image(lq_jpg)
    h200_info = _audit_image(h200_png, expected_size=lq_size)
    prompt_path = h200_dir / "prompt" / "case1.txt"
    prompt_text = prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else None
    if prompt_text not in ("", None):
        raise RuntimeError(f"captioner was not empty: {prompt_text!r}")
    if h200_info["size"][0] == 4 * lq_size[0] or h200_info["size"][1] == 4 * lq_size[1]:
        raise RuntimeError("H200 is 4x input; upscale=4 was used")
    fuse_info = texture_selective_fuse(lq_jpg, h200_png, out_jpg)
    out_info = _audit_image(out_jpg, expected_size=lq_size)
    audit = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "input_inventory": inventory,
        "dry_run_case": "case1",
        "note": "test case1 is not the validation image; no scene table is consulted",
        "lq": lq_info,
        "h200": h200_info,
        "output": out_info,
        "lq_mode": lq_mode,
        "prompt": prompt_text,
        "fusion": fuse_info,
        "config": FROZEN,
        "status": "DRY_RUN_OK",
    }
    (work / "DRY_RUN_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def run_full(project: Path) -> dict[str, object]:
    """100-image run. Do not call unless the operator passed --confirm-full."""
    assert_formula_frozen()
    test_dir = project / "csig_dataset" / "测试集"
    inventory = inventory_test_dir(test_dir)
    work = project / "baseline" / "experiments" / "final_test_inference" / "final_submission"
    h200_dir = work / "h200"
    out_dir = work / "output_dir"
    python = project / ".conda" / "python.exe"
    hypir_root = project / "HYPIR"
    run_hypir(python, test_dir, h200_dir, hypir_root)
    missing = []
    invalid = []
    wrong_size = []
    wrong_mode = []
    corrupted = []
    for i in range(1, 101):
        case = f"case{i}"
        lq_path = test_dir / f"{case}.jpg"
        h200_path = h200_dir / "result" / f"{case}.png"
        out_path = out_dir / f"{case}.jpg"
        if not lq_path.is_file() or not h200_path.is_file():
            missing.append(case)
            continue
        try:
            with Image.open(lq_path) as lq_im:
                expected = lq_im.size
            texture_selective_fuse(lq_path, h200_path, out_path)
            info = _audit_image(out_path, expected_size=expected)
            if info["mode"] != "RGB":
                wrong_mode.append(case)
        except Exception as exc:  # noqa: BLE001 — record and continue the audit
            corrupted.append(f"{case}:{exc}")
    extras = [path.name for path in out_dir.glob("*") if path.name not in {f"case{i}.jpg" for i in range(1, 101)}]
    status = "READY" if not (missing or invalid or wrong_size or wrong_mode or corrupted or extras) else "NOT READY"
    audit = {
        "Input": inventory,
        "Inference": {k: FROZEN[k] for k in ("model_t", "coeff_t", "upscale", "seed", "captioner")},
        "Fusion": {
            "method": FROZEN["method"],
            "formula": "0.05 + 0.46*texture*(0.35+0.65*(1-edge)); clip 0..0.60; RGB w*H200+(1-w)*LQ",
            "max_weight": FROZEN["max_weight"],
        },
        "Output": {
            "count": len(list(out_dir.glob("case*.jpg"))) if out_dir.is_dir() else 0,
            "missing": missing,
            "invalid": invalid,
            "wrong_size": wrong_size,
            "wrong_mode": wrong_mode,
            "corrupted": corrupted,
            "extra": extras,
        },
        "Status": status,
    }
    (work / "FINAL_SUBMISSION_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_ROOT)
    parser.add_argument("--mode", choices=["inventory", "dry-run", "full"], required=True)
    parser.add_argument("--confirm-full", action="store_true")
    args = parser.parse_args(argv)
    project = args.root.resolve()
    if args.mode == "inventory":
        report = inventory_test_dir(project / "csig_dataset" / "测试集")
        print(json.dumps(report, indent=2))
        return 0
    if args.mode == "full":
        if not args.confirm_full:
            raise SystemExit("Refusing 100-image run. Pass --mode full --confirm-full after dry-run sign-off.")
        audit = run_full(project)
        print(json.dumps(audit, indent=2))
        return 0 if audit["Status"] == "READY" else 1
    audit = dry_run(project)
    print(json.dumps(audit, indent=2))
    print("DRY-RUN OK. Not starting 100-image run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
