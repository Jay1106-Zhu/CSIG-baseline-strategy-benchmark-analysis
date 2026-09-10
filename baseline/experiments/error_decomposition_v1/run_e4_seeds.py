"""Run HYPIR-200 on case4 only, three new seeds. Does not modify HYPIR source."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HYPIR = ROOT / "HYPIR"
PYTHON = ROOT / ".conda" / "python.exe"
SEEDS = (17, 89, 401)


def main() -> int:
    for seed in SEEDS:
        out = ROOT / "baseline" / "experiments" / "error_decomposition_v1" / "e4_seeds" / f"seed_{seed}" / "output"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(PYTHON),
            "test.py",
            "--base_model_type", "sd2",
            "--base_model_path", "models/stable-diffusion-2-1-base",
            "--model_t", "200",
            "--coeff_t", "200",
            "--lora_rank", "256",
            "--lora_modules", "to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj",
            "--weight_path", "weights/HYPIR_sd2.pth",
            "--patch_size", "512",
            "--stride", "256",
            "--lq_dir", str(ROOT / "baseline" / "experiments" / "error_decomposition_v1" / "e4_input"),
            "--scale_by", "factor",
            "--upscale", "1",
            "--captioner", "empty",
            "--output_dir", str(out),
            "--seed", str(seed),
            "--device", "cuda",
        ]
        print(f"=== seed {seed} ===", flush=True)
        proc = subprocess.run(cmd, cwd=str(HYPIR))
        if proc.returncode != 0:
            return proc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
