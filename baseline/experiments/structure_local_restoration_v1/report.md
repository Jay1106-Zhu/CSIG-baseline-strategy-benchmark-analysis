# Structure-Anchored Local Restoration v1 Report

## Scope and conclusion

This is an offline, validation-only fusion study. Existing HYPIR-50 and HYPIR-200 images were reused; no diffusion inference, HYPIR source edit, or GT-driven weight was used.

The three maps are deliberately simple LQ-only heuristics. The decision below is based on all five cases and the local change/error CSV, not on a single-case optimum.

## Average metrics

| Method | PSNR | SSIM | LPIPS-Alex |
|---|---:|---:|---:|
| LQ | 28.034420 | 0.777699 | 0.204315 |
| HYPIR-50 | 28.257533 | 0.776935 | 0.162039 |
| HYPIR-200 | 24.528587 | 0.684926 | 0.159666 |
| confidence_200_LQ | 28.048711 | 0.776696 | 0.148463 |
| structure_guard_h50 | 28.257735 | 0.781777 | 0.195297 |
| structure_guard_h200 | 28.335641 | 0.774326 | 0.165913 |
| texture_selective_h50 | 28.327109 | 0.782096 | 0.193797 |
| texture_selective_h200 | 28.480280 | 0.781421 | 0.164546 |
| blurred_texture_h50 | 28.186536 | 0.780413 | 0.198931 |
| blurred_texture_h200 | 28.343105 | 0.780888 | 0.179947 |

## Strategy interpretation

- `structure_guard` assigns the lowest weights to LQ strong-edge regions; it is the primary text/book-spine/clock protection control.
- `texture_selective` permits more HYPIR only where local variance is high and edge strength is lower; it tests texture allowance without treating every high frequency as correct.
- `blurred_texture` gives its largest allowance to blurred, textured, non-edge areas; it tests whether uncertainty/blur should be a reason for modest generation.

## Local change/error evidence

`local_analysis.csv` reports LQ-only regions (`strong_edge`, `blurred_texture`, `textured_non_edge`, `other`) and measures both change from LQ and error to GT after inference. A lower change in a region is preservation; it is not by itself restoration.

| LQ region | H200 change | Guard-H200 change | Selective-H200 change | H200 error | Guard-H200 error | Selective-H200 error |
|---|---:|---:|---:|---:|---:|---:|
| strong_edge | 19.9579 | 3.3265 | 4.5728 | 20.9474 | 15.4210 | 15.2215 |
| textured_non_edge | 5.7186 | 1.4835 | 1.3618 | 9.3239 | 8.2152 | 8.1972 |
| blurred_texture | 24.5926 | 5.6132 | 5.8288 | 33.9444 | 29.3445 | 29.3456 |

- `confidence_200_LQ` average PSNR is 28.0487 dB; delta vs HYPIR-50 is -0.2088 dB and vs HYPIR-200 is +3.5201 dB.
- `structure_guard_h50` average PSNR is 28.2577 dB; delta vs HYPIR-50 is +0.0002 dB and vs HYPIR-200 is +3.7291 dB.
- `structure_guard_h200` average PSNR is 28.3356 dB; delta vs HYPIR-50 is +0.0781 dB and vs HYPIR-200 is +3.8071 dB.
- `texture_selective_h50` average PSNR is 28.3271 dB; delta vs HYPIR-50 is +0.0696 dB and vs HYPIR-200 is +3.7985 dB.
- `texture_selective_h200` average PSNR is 28.4803 dB; delta vs HYPIR-50 is +0.2227 dB and vs HYPIR-200 is +3.9517 dB.
- `blurred_texture_h50` average PSNR is 28.1865 dB; delta vs HYPIR-50 is -0.0710 dB and vs HYPIR-200 is +3.6579 dB.
- `blurred_texture_h200` average PSNR is 28.3431 dB; delta vs HYPIR-50 is +0.0856 dB and vs HYPIR-200 is +3.8145 dB.

## Answers to the requested questions

A. LQ structure features can determine where to reduce diffusion change in an image-space sense: the strong-edge region receives lower weights by construction and its measured change is reduced. The five-case data do not prove that these features predict GT-aligned improvement; prior diagnosis found only weak improvement correlations.

B. Text/book-spine/clock-like structure is plausibly protected because strong LQ edges receive less HYPIR. The evidence supports reduced rewriting, not exact character, numeral, or pointer identity; no OCR claim is made.

C. Texture regions are worth testing with a higher weight than protected edges, but only selectively. The texture and blurred-texture maps are controls for this hypothesis; higher frequency or higher weight is not treated as automatically better, and GT-aligned local error remains the gate.

D. This is sufficient to justify a small next local-control study, not a broad sweep or LoRA/module change. Advance only if a held-out case or seed reproduces lower edge-region rewriting without sacrificing global metrics; otherwise stop this direction.

## Limitations

Only five validation pairs and one seed are available. Regions are LQ feature quantiles, not semantic text/clock/texture masks. LPIPS is evaluated at a uniform max-side resize when enabled. Metrics cannot establish OCR or object identity.
