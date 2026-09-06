# Confidence Fusion v1 Experiment Metadata

- date: 2026-09-01
- status: completed
- scope: validation set case1-case5 only; offline fusion of existing HYPIR outputs
- git_commit_HYPIR: b61d107c6cef38f01a93c7833558869731cfa8c1
- base_model_type: sd2
- base_model_repo: sd-research/stable-diffusion-2-1-base
- base_model_local_path: HYPIR/models/stable-diffusion-2-1-base
- base_model_metadata_commit: 0708cecd370b4d1c3a6ff3f7332f5e9aea78896f
- LoRA_path: HYPIR/weights/HYPIR_sd2.pth
- LoRA_SHA256: D538A2CB925451FAB1F75ADFE715AC2B1C8BB12FA32A0851BA01BE2347B354D6
- seed: 231 (inherited from source HYPIR outputs)
- model_t: 200 (HYPIR-200 and HYPIR-50 source outputs)
- coeff_t_values: 200, 50 (source outputs; no new diffusion inference)
- lora_rank: 256
- lora_modules: to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj
- patch_size: 512
- stride: 256
- scale_by: factor
- upscale: 1
- prompt: empty
- device: cuda for source inference; CPU/GPU-independent NumPy/OpenCV fusion
- Python: 3.11.16
- PyTorch: 2.11.0+cu128
- CUDA runtime: 12.8
- GPU: NVIDIA GeForce RTX 5080 Laptop GPU (compute capability 12.0)

## Reused source outputs

The five `coeff_t=200` images are read from `baseline/experiments/coeff_t_200/output/result/`; the five `coeff_t=50` images are read from `baseline/experiments/coeff_t_50/output/result/`. Their metadata was checked before this run. The original baseline directories and CSV files were not written.

## Confidence heuristic

This is an **image-derived structural reliability heuristic**, not semantic confidence, ground-truth confidence, or learned confidence. For each LQ image:

1. Convert RGB to grayscale and downsample to one quarter resolution.
2. Compute Sobel gradient magnitude, absolute Laplacian, 9x9 local standard deviation, and a high-frequency residual (`abs(image - GaussianBlur(image, sigma=1.2))`).
3. Robustly normalize each response using its 2nd and 98th percentiles. Combine `0.45*gradient + 0.30*laplacian + 0.25*local_contrast - 0.18*residual`.
4. Clip to [0,1], Gaussian smooth at downsampled scale (sigma 3), resize to the original size, Gaussian smooth (sigma 9), clip, and map to `[0.08, 0.92]` as a blend weight.

The residual term is a soft noise penalty so isolated high-frequency speckle is not treated as reliable detail. A per-case median threshold creates exactly two masks: `high_confidence = confidence >= median` and `low_confidence = complement`. The threshold is deliberately simple and is not tuned against GT.

## Fusion equations

- `fixed_50_50 = 0.5*HYPIR200 + 0.5*HYPIR50`
- `confidence_200_50 = confidence*HYPIR200 + (1-confidence)*HYPIR50`
- `confidence_200_LQ = confidence*HYPIR200 + (1-confidence)*LQ`

All arithmetic is float32, clipped/rounded to uint8, and saved as RGB PNG. No spatial seam is introduced by a hard switch.

## Commands

```powershell
& .\\.conda\\python.exe baseline\\experiments\\run_confidence_fusion.py
& .\\.conda\\python.exe baseline\\experiments\\evaluate_confidence_fusion.py
& .\\.conda\\python.exe baseline\\experiments\\region_change_analysis.py
```

## Sanity checks

- All 15 fusion outputs are RGB PNG, finite, uint8-valued, and exactly match their LQ dimensions.
- All confidence maps match LQ dimensions; masks are grayscale and have equal high/low pixel counts within one pixel.
- `git -C HYPIR status --short --untracked-files=all` showed only the pre-existing untracked model/weight files; no tracked HYPIR source changes.
