# Adaptive H50/H200 Fusion v1 Summary

## Decision

Adaptive average metrics are PSNR 28.243879, SSIM 0.776406, LPIPS-Alex 0.157813 (LPIPS computed=True).
The current best `texture_selective_h200` reference is PSNR 28.480280, SSIM 0.781421, LPIPS-Alex 0.164546.
**1. 是否超过当前最佳？ 否。** 未同时明确超过参考，按决策标准停止 adaptive fusion 方向。

## Per-case PSNR

| Case | texture_selective_h200 | adaptive | Delta | 结果 |
|---|---:|---:|---:|---|
| case1 | 32.154456 | 32.143319 | -0.011137 | 受损 |
| case2 | 29.241406 | 28.978656 | -0.262750 | 受损 |
| case3 | 35.630109 | 34.631473 | -0.998636 | 受损 |
| case4 | 18.019865 | 18.070760 | +0.050895 | 获益 |
| case5 | 27.355563 | 27.395184 | +0.039621 | 获益 |

**2. 哪个 case 获益/受损？** 以上表格按 PSNR 列出；正值为获益，负值为受损。

## Alpha interpretation

Alpha uses only LQ Sobel gradient magnitude, local variance texture strength, inverse-Laplacian blur proxy, and an LQ strong-edge mask (top 20% gradient). The fixed formula is `clip(0.035 + 0.11*texture + 0.045*edge*(1-strong_edge) + 0.035*blur*texture*(1-strong_edge) - 0.18*strong_edge, 0, 0.30)`. Fusion is `(1-alpha)*H50 + alpha*H200`.
**3. α(x) 是否比之前 texture mask 更有效？** 只能以全图和区域 GT error 判断；alpha 的连续性与边缘抑制是设计性质，不等于恢复正确。若本实验未超过参考，则没有证据表明它比 texture mask 更有效。

## Regional evidence

`region_metrics.csv` records `change_L1` (deviation from LQ) and `error_to_GT_L1` for LQ-defined strong_edge, textured_non_edge, and blurred_texture regions. Lower GT error is the restoration criterion; lower change only indicates preservation.

| Region | H50 change | H50 error | texture-selective change | texture-selective error | adaptive change | adaptive error |
|---|---:|---:|---:|---:|---:|---:|
| strong_edge | 7.1654 | 15.1776 | 4.5613 | 15.2183 | 7.1654 | 15.1776 |
| textured_non_edge | 2.2565 | 8.4473 | 1.3165 | 8.1903 | 2.5214 | 8.4063 |
| blurred_texture | 2.9815 | 29.4938 | 5.7963 | 29.3457 | 5.5179 | 28.9512 |

## Scope and stop

No GT value entered alpha construction or threshold selection. No diffusion inference, training, LoRA, or parameter sweep was run. H50/H200/LQ/texture-selective files were reused read-only.
**4. 失败，明确停止 adaptive fusion 方向。** Adaptive v1 没有明确超过当前最佳，按要求不再 sweep 或扩展该方向。
