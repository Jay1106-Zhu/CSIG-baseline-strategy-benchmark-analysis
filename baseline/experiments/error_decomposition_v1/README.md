# Error decomposition v1

Offline (E0–E3)、gated multi-seed (E4) 以及对非重叠 256 块的全网格 PSNR/SSIM/LPIPS。不修改 HYPIR 源码。

现行计划：[`CURRENT_PLAN.md`](../../../CURRENT_PLAN.md)。结论：[`report.md`](report.md)。

```powershell
& .\.conda\python.exe baseline\experiments\error_decomposition_v1\run_error_decomposition.py
& .\.conda\python.exe baseline\experiments\error_decomposition_v1\run_e4_seeds.py
& .\.conda\python.exe baseline\experiments\error_decomposition_v1\analyze_e4.py
& .\.conda\python.exe baseline\experiments\error_decomposition_v1\compute_patch_metrics.py
```

分块产物：`patch_metrics_full.csv`（960 块）、`patch_metrics_summary.csv`、`e1_patch_metrics.csv`（已标注 36 块）。块 LPIPS 是 256 原生 Alex，不可与全图 1024 协议混用。
