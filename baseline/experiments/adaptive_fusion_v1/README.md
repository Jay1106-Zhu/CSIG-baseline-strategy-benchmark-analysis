# Adaptive H50/H200 Fusion v1

Offline fusion of existing LQ, HYPIR-50, HYPIR-200, and texture-selective outputs. No diffusion inference or training is performed.

Run from the project root:

```powershell
& .\.conda\python.exe baseline\experiments\adaptive_fusion_v1\experiment.py --root . --with-lpips
```

Artifacts: `metrics.csv`, `region_metrics.csv`, `alpha_maps/`, `fusion/`, `comparison/`, and `summary.md`.
