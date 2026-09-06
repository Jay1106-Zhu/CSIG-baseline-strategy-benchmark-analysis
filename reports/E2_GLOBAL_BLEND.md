# E2 Global Alpha Blend

## 1. Objective

Evaluate whether a fixed global blend of the existing HYPIR-50 output and the
existing E1 SwinIR fidelity output improves validation metrics on the five
cases. This is an offline pixel-space experiment only; HYPIR and SwinIR were
not rerun.

## 2. Inputs

- HYPIR-50: `baseline/experiments/coeff_t_50/output/result/case*_lq.png`.
- Fidelity: `baseline/experiments/E1_fidelity/swinir_car_jpeg40/output/case*.png`.
- GT: `csig_dataset/验证集/case*_gt.jpg`.
- All five HYPIR/Fidelity/GT triplets were checked before blending: RGB,
  uint8, value range 0--255, and exactly matching resolution. No resize was
  performed.
- Outputs: `baseline/experiments/E2_global_blend/outputs/alpha_{0.0...1.0}/case*.png`.
- Output health check passed for all 30 PNGs. Endpoint sanity check passed:
  `alpha=0.0` equals Fidelity and `alpha=1.0` equals HYPIR-50 pixel-for-pixel
  for every case.

## 3. Formula

For HYPIR-50 image `H`, Fidelity image `F`, and global scalar alpha:

```text
I_alpha = alpha * H + (1 - alpha) * F
I_alpha = clip(I_alpha, 0, 255)
I_alpha = round(I_alpha).astype(uint8)
```

Each result is saved as an RGB PNG. No spatial map, texture map, Sobel,
Gaussian smoothing, training, LoRA, SAM, OCR, ensemble, or model inference was
used.

## 4. Alpha Values

`{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`

- `0.0`: Fidelity only.
- `1.0`: HYPIR-50 only.

## 5. Average Results

| Alpha | PSNR | SSIM | LPIPS-Alex |
|---:|---:|---:|---:|
| 0.0 | 28.028629 | 0.780163 | 0.216824 |
| 0.2 | 28.292694 | 0.783450 | 0.205195 |
| 0.4 | 28.464477 | **0.784890** | 0.191612 |
| 0.6 | **28.507181** | 0.783521 | 0.177557 |
| 0.8 | 28.431208 | 0.780532 | 0.167934 |
| 1.0 | 28.257533 | 0.776935 | **0.162039** |

LPIPS is lower-is-better.

## 6. Per-case Results

| Case | Alpha | PSNR | SSIM | LPIPS-Alex |
|---|---:|---:|---:|---:|
| case1 | 0.0 | 32.105050 | 0.951272 | 0.084784 |
| case1 | 0.2 | 32.215433 | 0.952450 | 0.081447 |
| case1 | 0.4 | 32.280139 | 0.952345 | 0.076160 |
| case1 | 0.6 | 32.268785 | 0.949603 | 0.069179 |
| case1 | 0.8 | 32.216515 | 0.946926 | 0.066373 |
| case1 | 1.0 | 32.143319 | 0.945300 | 0.064668 |
| case2 | 0.0 | 28.111951 | 0.826285 | 0.109086 |
| case2 | 0.2 | 28.555478 | 0.831007 | 0.098565 |
| case2 | 0.4 | 28.890274 | 0.833225 | 0.090336 |
| case2 | 0.6 | 29.077392 | 0.832408 | 0.084177 |
| case2 | 0.8 | 29.103657 | 0.829287 | 0.080584 |
| case2 | 1.0 | 28.976278 | 0.825432 | 0.079454 |
| case3 | 0.0 | 35.404210 | 0.932529 | 0.102273 |
| case3 | 0.2 | 35.557228 | 0.933241 | 0.095609 |
| case3 | 0.4 | 35.566114 | 0.932511 | 0.087163 |
| case3 | 0.6 | 35.382589 | 0.929093 | 0.074909 |
| case3 | 0.8 | 35.068389 | 0.924563 | 0.071398 |
| case3 | 1.0 | 34.688713 | 0.920412 | 0.074222 |
| case4 | 0.0 | 18.074936 | 0.312053 | 0.663786 |
| case4 | 0.2 | 18.089977 | 0.312850 | 0.645243 |
| case4 | 0.4 | 18.098758 | 0.313203 | 0.619810 |
| case4 | 0.6 | 18.100669 | 0.312884 | 0.592573 |
| case4 | 0.8 | 18.095744 | 0.311790 | 0.566467 |
| case4 | 1.0 | 18.084599 | 0.310380 | 0.543474 |
| case5 | 0.0 | 26.446996 | 0.878676 | 0.124192 |
| case5 | 0.2 | 27.045354 | 0.887704 | 0.105111 |
| case5 | 0.4 | 27.487101 | 0.893164 | 0.084589 |
| case5 | 0.6 | 27.706468 | 0.893615 | 0.066949 |
| case5 | 0.8 | 27.671733 | 0.890096 | 0.054848 |
| case5 | 1.0 | 27.394758 | 0.883153 | 0.048376 |

The complete per-case result and delta table is
`baseline/experiments/E2_global_blend/metrics.csv`.

## 7. Delta vs HYPIR-50

Delta is `Blend(alpha) - HYPIR-50`; positive PSNR/SSIM is better, while
negative LPIPS is better.

| Alpha | Delta PSNR | Delta SSIM | Delta LPIPS |
|---:|---:|---:|---:|
| 0.0 | -0.228905 | +0.003227 | +0.054786 |
| 0.2 | +0.035161 | +0.006515 | +0.043157 |
| 0.4 | +0.206944 | +0.007954 | +0.029573 |
| 0.6 | +0.249647 | +0.006586 | +0.015519 |
| 0.8 | +0.173674 | +0.003597 | +0.005896 |
| 1.0 | +0.000000 | +0.000000 | +0.000000 |

No alpha below 1.0 improves average LPIPS relative to HYPIR-50.

## 8. Delta vs SwinIR

Delta is `Blend(alpha) - Fidelity`; positive PSNR/SSIM is better, while
negative LPIPS is better.

| Alpha | Delta PSNR | Delta SSIM | Delta LPIPS |
|---:|---:|---:|---:|
| 0.0 | +0.000000 | +0.000000 | +0.000000 |
| 0.2 | +0.264065 | +0.003287 | -0.011629 |
| 0.4 | +0.435849 | +0.004727 | -0.025213 |
| 0.6 | +0.478552 | +0.003358 | -0.039267 |
| 0.8 | +0.402579 | +0.000369 | -0.048890 |
| 1.0 | +0.228905 | -0.003227 | -0.054786 |

On the five-case average, alphas 0.2 through 0.8 improve all three metrics
relative to Fidelity.

## 9. Best Alpha

Average best alpha:

| Metric | Best alpha | Value |
|---|---:|---:|
| PSNR | 0.6 | 28.507181 |
| SSIM | 0.4 | 0.784890 |
| LPIPS-Alex | 1.0 | 0.162039 |

Per-case best alpha:

| Case | Best PSNR alpha | Best SSIM alpha | Best LPIPS alpha |
|---|---:|---:|---:|
| case1 | 0.4 | 0.2 | 1.0 |
| case2 | 0.8 | 0.4 | 1.0 |
| case3 | 0.4 | 0.2 | 0.8 |
| case4 | 0.6 | 0.4 | 1.0 |
| case5 | 0.6 | 0.6 | 1.0 |

## 10. Conclusion

1. Global Blend improves average PSNR and SSIM over HYPIR-50 at alpha 0.4 or
   0.6, but worsens average LPIPS at every alpha below 1.0. No evaluated alpha
   is better than HYPIR-50 on all three average metrics.
2. Global Blend at alpha 0.2 through 0.8 is better than SwinIR Fidelity on all
   three average metrics.
3. There is no single three-metric alpha sweet spot: average PSNR is best at
   0.6, average SSIM at 0.4, and average LPIPS at the HYPIR-only endpoint 1.0.
4. Best alpha differs by case and metric, as listed above.

No E3 experiment was started. The next experiment decision is left to ChatGPT.
