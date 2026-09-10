# HYPIR fusion v1 summary

Offline output-space fusion. H200 is a detail candidate; LQ/H50 are structure anchors;
the Sobel structure mask suppresses contours that H200 invents.

- date: 2026-09-10
- LPIPS: Alex max-side 1024, computed=True
- scheme A: `Y = Y_LQ + α·mask·(Y_H200 - Y_LQ)`
- scheme B: `Y = Y_H50 + α·mask·(Y_H200 - Y_H50)`
- chroma: LQ Cb/Cr for both schemes

## Average metrics

| method | PSNR | SSIM | LPIPS_1024 |
|---|---:|---:|---:|
| LQ | 28.034420 | 0.777699 | 0.204318 |
| HYPIR-50 | 28.257533 | 0.776935 | 0.162039 |
| HYPIR-200 | 24.528587 | 0.684926 | 0.159664 |
| fusion_A | 28.460407 | 0.781670 | 0.188585 |
| fusion_B | 28.256170 | 0.778593 | 0.156365 |
| texture_selective_h200 | 28.480280 | 0.781421 | 0.164549 |

## Per-case

| case | scene | alpha | mean_mask | fusion_A PSNR | fusion_B PSNR | H50 PSNR | H200 PSNR | LQ PSNR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| case1 | text | 0.25 | 0.2765 | 32.1744 | 32.1258 | 32.1433 | 29.3353 | 32.0288 |
| case2 | book | 0.30 | 0.5920 | 29.2045 | 29.0343 | 28.9763 | 24.2140 | 28.1894 |
| case3 | bird | 0.12 | 0.6311 | 35.7429 | 34.7155 | 34.6887 | 28.8654 | 35.6221 |
| case4 | plant | 0.08 | 0.4804 | 18.0636 | 18.0696 | 18.0846 | 16.3524 | 18.0562 |
| case5 | clock | 0.30 | 0.5561 | 27.1167 | 27.3357 | 27.3948 | 23.8758 | 26.2756 |

## Visual checklist

- case4 `crops/case4_mid04_fish.png`: pink pod / fish-head hallucination should be weaker than raw H200
- case4 `crops/case4_high02_leaf.png`: serrated wrong leaf vs compound GT
- case3 `crops/case3_low02_water.png`: invented water grain should be closer to LQ/H50
- case1/2/5 comparison panels: text, spine, clock hands must not be redrawn

## How to read the mask

Yellow/green in `masks/` = H200 structure agrees with LQ (detail allowed).
Dark in `heatmaps/*_suppressed.png` = H200 residual that the mask blocked.

## Not in this experiment

Degradation Encoder, Adapter, LoRA, diffusion finetune, ControlNet, Global Attention.
