# Texture-region HYPIR weight sweep

This is the completed five-case offline sweep requested for the Structure-Anchored Local Restoration v1 `texture_selective_h200` baseline.

- Candidates: `0.25`, `0.40`, `0.55`, `0.70`, `0.85`, `1.00`.
- Existing HYPIR-200 PNGs are reused; no diffusion inference, training, model, or HYPIR source changes.
- Only the fixed LQ texture/non-edge mask (`texture >= P80` and `edge < P80`) is assigned the candidate weight. Strong-edge weights are unchanged.
- `metrics.csv`: full-image PSNR/SSIM/LPIPS-Alex per case and average.
- `local_analysis.csv`: change L1 and GT error for `strong_edge`, `textured_non_edge`, and `blurred_texture`.
- `comparison/`: LQ, HYPIR-200, v1 baseline, six candidates, and GT panels.
- `report.md`: per-case winners and the final STOP decision.

Reproduce from the project root:

```powershell
& .\.conda\python.exe baseline\experiments\texture_weight_sweep.py --with-lpips --output-dir baseline\experiments\texture_weight_sweep_v2
```

The output directory must be empty because the script refuses to overwrite an existing experiment.
