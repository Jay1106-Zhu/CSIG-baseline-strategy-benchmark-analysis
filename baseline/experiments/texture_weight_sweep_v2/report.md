# Texture-region HYPIR weight sweep report

## Scope

This is a five-case, offline validation experiment. Existing HYPIR-200 outputs were reused. The only changed variable is the direct HYPIR blend weight inside the fixed LQ texture/non-edge region; the v1 map is retained everywhere else, including strong edges.

## Full-image averages

| Method | PSNR | SSIM | LPIPS-Alex |
|---|---:|---:|---:|
| LQ | 28.034420 | 0.777699 | 0.204318 |
| HYPIR-200 | 24.528587 | 0.684926 | 0.159664 |
| texture_selective_h200 | 28.480280 | 0.781421 | 0.164549 |
| texture_weight_0.25_h200 | 28.479684 | 0.781403 | 0.164545 |
| texture_weight_0.40_h200 | 28.476294 | 0.781214 | 0.164317 |
| texture_weight_0.55_h200 | 28.468252 | 0.780793 | 0.164004 |
| texture_weight_0.70_h200 | 28.456087 | 0.780213 | 0.163652 |
| texture_weight_0.85_h200 | 28.439112 | 0.779462 | 0.163265 |
| texture_weight_1.00_h200 | 28.418456 | 0.778650 | 0.162872 |

## Per-case result and best weight

Best weight is reported independently for each full-image metric; ties use the lower weight. This avoids hiding metric disagreement behind one arbitrary score.

| Case | Best PSNR weight | Best SSIM weight | Best LPIPS weight | Baseline PSNR | Best PSNR | Texture-region error at baseline | Lowest texture-region error weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| case1 | 0.25 | 0.25 | 1.00 | 32.154456 | 32.154426 | 1.881103 | 0.25 |
| case2 | 0.25 | 0.25 | 0.70 | 29.241406 | 29.240913 | 5.648774 | 0.40 |
| case3 | 0.25 | 0.25 | 0.25 | 35.630109 | 35.627483 | 2.821497 | 0.25 |
| case4 | 0.25 | 0.25 | 1.00 | 18.019865 | 18.019823 | 26.646569 | 0.25 |
| case5 | 0.55 | 0.40 | 1.00 | 27.355563 | 27.359083 | 6.687310 | 0.55 |

## Regional change and GT error

For each LQ-defined region, `change_L1` measures generated deviation from LQ and `error_to_GT_L1` measures absolute error after fusion. Lower GT error, rather than larger change, is the restoration criterion.

| Method/weight | strong_edge change | strong_edge GT error | textured_non_edge change | textured_non_edge GT error | blurred_texture change | blurred_texture GT error |
|---|---:|---:|---:|---:|---:|---:|
| v1 baseline | 4.5728 | 15.2215 | 1.3618 | 8.1972 | 5.8288 | 29.3456 |
| weight 0.25 | 4.5728 | 15.2215 | 1.4297 | 8.2142 | 6.1481 | 29.2994 |
| weight 0.40 | 4.5728 | 15.2215 | 2.2874 | 8.2505 | 9.8370 | 29.2481 |
| weight 0.55 | 4.5728 | 15.2215 | 3.1452 | 8.3922 | 13.5259 | 29.7000 |
| weight 0.70 | 4.5728 | 15.2215 | 4.0030 | 8.6282 | 17.2148 | 30.6161 |
| weight 0.85 | 4.5728 | 15.2215 | 4.8608 | 8.9448 | 20.9037 | 32.0222 |
| weight 1.00 | 4.5728 | 15.2215 | 5.7186 | 9.3239 | 24.5926 | 33.9444 |

## Decision

- Highest average PSNR candidate: `0.25` (28.479684 dB); v1 baseline is 28.480280 dB.
- **Conclusion: STOP this direction.** Increasing the direct texture-region weight does not lower GT error consistently; the bird case worsens across all candidates, and the foliage dip is narrow and not reflected in full-image metrics.
- The experiment answers the hypothesis only when texture-region GT error decreases relative to the v1 baseline while strong-edge error/change remains unchanged. More change or more visible texture alone is not evidence of recovery.
- Per-case metric winners and regional error winners are listed above; they disagree across cases, so no single texture weight should be advanced to local control.

## Limitations

Only five validation pairs and one seed are available. Texture regions are LQ feature quantiles rather than semantic bird/foliage masks. LPIPS uses the same uniform resized protocol as v1 and is supportive, not a semantic identity test.
