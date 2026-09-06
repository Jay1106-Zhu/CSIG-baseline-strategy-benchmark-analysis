# E3-A Texture-only Adaptive Alpha

## Scope

This is an offline blend of existing HYPIR-50 and SwinIR outputs. No HYPIR or SwinIR inference, model change, training, detector, semantic label, output-derived feature, GT-derived alpha, parameter sweep, or additional experiment was performed.

## Fixed Method

The LQ-only grayscale texture score is the equal-weight mean of robustly normalized Sobel magnitude, 9x9 local variance, and absolute Gaussian blur residual (blur sigma=2.0). Each feature uses fixed 1st/99th percentile normalization to [0,1]. The combined score is spatially smoothed once with fixed Gaussian sigma=8 pixels.

- `global_alpha_0.6`: `I=0.6*HYPIR + 0.4*SwinIR`.
- `adaptive_texture`: `alpha=0.30+0.40*T`; `I=alpha*HYPIR+(1-alpha)*SwinIR`.
- `adaptive_reverse`: `alpha=0.70-0.40*T`; otherwise identical.

Every blend is clipped to [0,255], rounded, converted to RGB uint8, and saved at its original resolution. GT is used only for metrics.

## Metrics

PSNR uses `skimage` with `data_range=255`; SSIM uses `skimage` RGB channel-axis handling; LPIPS-Alex uses the existing E0/E1/E2 preprocessing with a longest-side <=1024 bilinear resize. Averages are arithmetic means of the five per-case results.

The `global_alpha_0.6` output is pixel-identical to E2's saved alpha 0.6 output; its six-decimal baseline values are retained here for exact comparison.

| Method | PSNR | SSIM | LPIPS-Alex | Mean alpha | Mean alpha std |
| --- | ---: | ---: | ---: | ---: | ---: |
| global_alpha_0.6 | 28.507181 | 0.783521 | 0.177557 | 0.600000 | 0.000000 |
| adaptive_texture | 28.475310 | 0.785022 | 0.191381 | 0.340079 | 0.044270 |
| adaptive_reverse | 28.499589 | 0.783138 | 0.177883 | 0.659921 | 0.044270 |

## Per-case Results And Map Statistics

`Texture_Alpha_Pearson` is undefined (`N/A`) for the constant global alpha map; it is otherwise the per-pixel Pearson correlation of the same LQ texture score and alpha map.

| Case | Method | PSNR | SSIM | LPIPS-Alex | Mean alpha | Std | Min | Max | Mean T | Texture-alpha r |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| case1 | global_alpha_0.6 | 32.268785 | 0.949603 | 0.069179 | 0.600000 | 0.000000 | 0.600000 | 0.600000 | 0.060866 | N/A |
| case1 | adaptive_texture | 32.274974 | 0.952446 | 0.075713 | 0.324347 | 0.049108 | 0.300000 | 0.562090 | 0.060866 | 1.000000 |
| case1 | adaptive_reverse | 32.279137 | 0.949481 | 0.069626 | 0.675653 | 0.049108 | 0.437910 | 0.700000 | 0.060866 | -1.000000 |
| case2 | global_alpha_0.6 | 29.077392 | 0.832408 | 0.084177 | 0.600000 | 0.000000 | 0.600000 | 0.600000 | 0.085447 | N/A |
| case2 | adaptive_texture | 28.910265 | 0.833095 | 0.091641 | 0.334179 | 0.043575 | 0.300263 | 0.578018 | 0.085447 | 1.000000 |
| case2 | adaptive_reverse | 29.058603 | 0.832239 | 0.083095 | 0.665821 | 0.043575 | 0.421982 | 0.699737 | 0.085447 | -1.000000 |
| case3 | global_alpha_0.6 | 35.382589 | 0.929093 | 0.074909 | 0.600000 | 0.000000 | 0.600000 | 0.600000 | 0.103119 | N/A |
| case3 | adaptive_texture | 35.570823 | 0.932778 | 0.087910 | 0.341247 | 0.041296 | 0.300964 | 0.651080 | 0.103119 | 1.000000 |
| case3 | adaptive_reverse | 35.372916 | 0.928572 | 0.075049 | 0.658753 | 0.041296 | 0.348920 | 0.699036 | 0.103119 | -1.000000 |
| case4 | global_alpha_0.6 | 18.100669 | 0.312884 | 0.592573 | 0.600000 | 0.000000 | 0.600000 | 0.600000 | 0.164916 | N/A |
| case4 | adaptive_texture | 18.097815 | 0.313373 | 0.618088 | 0.365966 | 0.044959 | 0.303710 | 0.624285 | 0.164916 | 1.000000 |
| case4 | adaptive_reverse | 18.101726 | 0.312606 | 0.593826 | 0.634034 | 0.044959 | 0.375715 | 0.696290 | 0.164916 | -1.000000 |
| case5 | global_alpha_0.6 | 27.706468 | 0.893615 | 0.066949 | 0.600000 | 0.000000 | 0.600000 | 0.600000 | 0.086639 | N/A |
| case5 | adaptive_texture | 27.522673 | 0.893416 | 0.083554 | 0.334656 | 0.042410 | 0.300000 | 0.557020 | 0.086639 | 1.000000 |
| case5 | adaptive_reverse | 27.685561 | 0.892792 | 0.067821 | 0.665344 | 0.042410 | 0.442980 | 0.700000 | 0.086639 | -1.000000 |

## Per-case Deltas

| Case | Method | dPSNR vs global | dSSIM vs global | dLPIPS vs global | dPSNR vs HYPIR-50 | dSSIM vs HYPIR-50 | dLPIPS vs HYPIR-50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| case1 | global_alpha_0.6 | +0.000000 | +0.000000 | +0.000000 | +0.125465 | +0.004304 | +0.004510 |
| case1 | adaptive_texture | +0.006189 | +0.002843 | +0.006534 | +0.131654 | +0.007147 | +0.011044 |
| case1 | adaptive_reverse | +0.010352 | -0.000123 | +0.000447 | +0.135817 | +0.004181 | +0.004958 |
| case2 | global_alpha_0.6 | +0.000000 | +0.000000 | +0.000000 | +0.101114 | +0.006976 | +0.004724 |
| case2 | adaptive_texture | -0.167127 | +0.000687 | +0.007464 | -0.066012 | +0.007664 | +0.012188 |
| case2 | adaptive_reverse | -0.018788 | -0.000169 | -0.001082 | +0.082326 | +0.006807 | +0.003641 |
| case3 | global_alpha_0.6 | +0.000000 | +0.000000 | +0.000000 | +0.693876 | +0.008681 | +0.000688 |
| case3 | adaptive_texture | +0.188234 | +0.003685 | +0.013001 | +0.882110 | +0.012366 | +0.013688 |
| case3 | adaptive_reverse | -0.009673 | -0.000521 | +0.000140 | +0.684204 | +0.008160 | +0.000827 |
| case4 | global_alpha_0.6 | +0.000000 | +0.000000 | +0.000000 | +0.016070 | +0.002505 | +0.049099 |
| case4 | adaptive_texture | -0.002854 | +0.000489 | +0.025515 | +0.013216 | +0.002994 | +0.074614 |
| case4 | adaptive_reverse | +0.001056 | -0.000279 | +0.001254 | +0.017126 | +0.002226 | +0.050353 |
| case5 | global_alpha_0.6 | +0.000000 | +0.000000 | +0.000000 | +0.311710 | +0.010462 | +0.018574 |
| case5 | adaptive_texture | -0.183795 | -0.000199 | +0.016605 | +0.127915 | +0.010263 | +0.035179 |
| case5 | adaptive_reverse | -0.020906 | -0.000823 | +0.000872 | +0.290803 | +0.009639 | +0.019445 |

## Visual Sanity Check

The saved `texture`, forward-alpha, and reverse-alpha maps share each source image's native dimensions. Visual inspection confirms that texture-map variation follows LQ image detail and that the forward/reverse alpha maps invert that variation without blank regions or tiling. Map values are finite RGB uint8 visualizations; forward and reverse maps are complementary (`alpha_texture + alpha_reverse = 1.0` before PNG quantization), and their map statistics obey [0.30,0.70]. The visual check is limited to whether LQ texture complexity is a useful proxy for spatial HYPIR contribution; it does not make a claim about hallucination-proneness.

## Conclusion

`adaptive_texture` does not improve all three average metrics over `global_alpha_0.6`; the fixed global blend remains the stronger composite result under this five-case check. Relative to `adaptive_texture`, `adaptive_reverse` changes average PSNR/SSIM/LPIPS by +0.024279/-0.001884/-0.013498.
