# Structure-Anchored Local Restoration v1

This directory contains the offline LQ-only spatial fusion experiment for validation `case1`-`case5`. Existing HYPIR-50 and HYPIR-200 images were reused; no new diffusion inference was run.

Run from the project root:

```powershell
& .\.conda\python.exe baseline\experiments\structure_local_restoration.py --with-lpips
```

The command writes three fixed strategies (`structure_guard`, `texture_selective`, `blurred_texture`) with both HYPIR sources. `weight_maps/` contains color and grayscale maps, `fusion/` contains the 30 RGB fusion images, and `comparison/` contains per-case panels. `metrics.csv` includes LQ, HYPIR-50, HYPIR-200, the existing `confidence_200_LQ`, and all six fusion variants. `local_analysis.csv` records LQ-quantile regions, change from LQ, and post-inference error to GT. Formulae, paths, fixed constants, and limitations are in `experiment_metadata.md`; conclusions are in `report.md`.

The LPIPS values use the same Alex network and a uniform 1024-pixel maximum-side resize for every method, including the baselines.
