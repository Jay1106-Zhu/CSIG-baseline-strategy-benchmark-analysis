# Scenario Routing v1

Offline routing proof-of-concept. No classifier, training, new diffusion inference, or GT-driven routing was used.

## Fixed rule

Features are computed from LQ only at max-side 1024: edge density, local variance, blur proxy, high-frequency energy, gradient mean, and gradient standard deviation.

- blurred/low-frequency: blur_proxy >= 0.70, high_frequency_energy < 0.22, edge_density < 0.18, gradient_mean < 0.16 -> HYPIR-200
- structure-dominant: edge_density >= 0.18, local_variance < 0.22, high_frequency_energy < 0.35 -> HYPIR-50
- texture-dominant: local_variance >= 0.22, high_frequency_energy >= 0.22, edge_density < 0.28 -> texture_selective_h200
- ambiguous: all remaining inputs -> LQ

The thresholds and mapping are fixed globally; no case name, GT, validation score, or semantic label is referenced.

## Routing decisions

| Case | Scenario | Selected method |
|---|---|---|
| case1 | blurred/low-frequency | HYPIR-200 |
| case2 | ambiguous | LQ |
| case3 | blurred/low-frequency | HYPIR-200 |
| case4 | blurred/low-frequency | HYPIR-200 |
| case5 | ambiguous | LQ |

## Average comparison

| Method | PSNR | SSIM | LPIPS-Alex |
|---|---:|---:|---:|
| LQ | 28.034420 | 0.777699 | 0.204315 |
| HYPIR-50 | 28.257533 | 0.776935 | 0.162039 |
| HYPIR-200 | 24.528587 | 0.684926 | 0.159666 |
| texture_selective_h200 | 28.480280 | 0.781421 | 0.164546 |
| Scenario Routing v1 | 25.803636 | 0.722419 | 0.169598 |

## Per-case winners from existing matrix

Composite winner is the lowest sum of three within-case ranks: PSNR descending, SSIM descending, LPIPS-Alex ascending. This is a descriptive tie-breaker, not a routing rule.

| Case | PSNR best | SSIM best | LPIPS best | Composite best |
|---|---|---|---|---|
| case1 | texture_selective_h200 (32.154456) | texture_selective_h200 (0.949652) | HYPIR-50 (0.064659) | texture_selective_h200 |
| case2 | texture_selective_h200 (29.241406) | texture_selective_h200 (0.838088) | texture_selective_h200 (0.079302) | texture_selective_h200 |
| case3 | texture_selective_h200 (35.630109) | LQ (0.934958) | texture_selective_h200 (0.056104) | texture_selective_h200 |
| case4 | HYPIR-50 (18.084599) | HYPIR-50 (0.310380) | HYPIR-200 (0.376833) | HYPIR-50 |
| case5 | HYPIR-50 (27.394758) | HYPIR-50 (0.883153) | HYPIR-50 (0.048371) | HYPIR-50 |

## Scene-difference analysis

1. Per-case optima exist: the best method changes by case and metric; there is no universal winner.
2. `texture_selective_h200` is not stable best: it leads case1-case3 on PSNR/SSIM or LPIPS but loses case4 and case5 on key metrics.
3. HYPIR-50 is the strongest conservative candidate for case4/case5-like outcomes in this matrix and is competitive on case1/case2.
4. `texture_selective_h200` is most competitive on the first three cases, especially case2, but its edge-region change remains higher than H50.
5. Raw HYPIR-200 has no PSNR or SSIM win; its only clear practical advantage is LPIPS on case4, where PSNR/SSIM are substantially worse.
6. Region errors support strategy differences but not a reliable selector: H200 often increases strong-edge and blurred-texture error, while selective fusion reduces textured-region error in some cases.
Using the supplied scene descriptions, H50 is the best global PSNR/SSIM choice for dense foliage (case4) and clock (case5), while texture-selective H200 is the composite choice for small face/text (case1), book spine text (case2), and bird (case3). This scene interpretation is analysis only; the routing rule never consumes these labels.

## Regional error means

These are means over available LQ-defined pixels/cases; blank blurred-texture entries mean that region was absent in that case.

| Region | HYPIR-50 error | HYPIR-200 error | texture-selective H200 error |
|---|---:|---:|---:|
| strong_edge | 15.1776 | 20.9474 | 15.2215 |
| textured_non_edge | 8.4473 | 9.3239 | 8.1972 |
| blurred_texture | 29.4938 | 33.9444 | 29.3456 |

## Decision gate

Scenario Routing v1 selects HYPIR-200 for case1/case3/case4 and LQ for case2/case5. Its average is therefore materially below the fixed baselines (see table and `metrics.csv`), with worse PSNR and SSIM than `texture_selective_h200`; LPIPS also worsens because the H200 selections dominate.

Conclusion: B. routing has no evidence of value in this proof-of-concept. Stop routing here; do not train a classifier, add diffusion inference, or tune rules on these five validation images. Reconsider architecture innovation separately.

Regional evidence is in `per_case_method_matrix.csv` and is LQ-region based (`strong_edge`, `textured_non_edge`, `blurred_texture`). Lower change means preservation, not necessarily GT-aligned improvement.
