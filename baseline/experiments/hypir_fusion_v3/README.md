# NO-GO multi-band

# HYPIR fusion v3 — Laplacian multi-band

独立实验。不修改 HYPIR，不覆盖 fusion_v1 / fusion_v2 / texture_selective。

HYPIR 自带 wavelet 只有 2 带（全部高频 vs 低频），不能拆中频。本实验用 **Laplacian pyramid** 三带。

```powershell
& .\.conda\python.exe -m unittest tests.test_hypir_fusion_v3
& .\.conda\python.exe baseline\experiments\hypir_fusion_v3\experiment.py --root .
```
