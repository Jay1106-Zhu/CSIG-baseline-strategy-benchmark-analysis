# Phase 0 metric freeze

- PSNR / SSIM: `baseline/experiments/coeff_t_{50,200}/evaluation_metrics.csv` (native resolution, skimage, data_range=255).
- LPIPS_1024: `baseline/experiments/structure_local_restoration_v1/metrics.csv` (Alex, max-side 1024 bilinear). Use this for FR/perception comparisons.
- LPIPS_native_coeff: coeff_t CSVs (Alex, native 4K). Inflated vs 1024 protocol; do not mix.
- Patch LPIPS: `patch_metrics_full.csv` (Alex on native 256 tiles). Do not mix with LPIPS_1024 or native 4K.
- No new inference. No HYPIR source changes.
