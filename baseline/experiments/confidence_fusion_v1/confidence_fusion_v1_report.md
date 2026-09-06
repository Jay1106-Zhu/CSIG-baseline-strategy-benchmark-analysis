# Confidence-guided HYPIR Fusion v1 Report

Date: 2026-09-01  
Data: validation `case1`-`case5` only  
Source outputs: existing HYPIR `model_t=200`, `coeff_t=200` and `coeff_t=50` PNGs, verified from their metadata and reused without new diffusion inference.

## Short conclusion

**Weak success.** The image-derived map produces smooth, nontrivial spatial weights and changes are measurably concentrated in its high-confidence half, while the conservative HYPIR/LQ blend keeps low-confidence regions close to LQ. However, no confidence-guided method matches the global HYPIR-50 average PSNR, and the data do not show that high-confidence changes are true detail recovery rather than safer-looking generation.

## Average full-image metrics

| Method | PSNR | SSIM | LPIPS-Alex | Delta PSNR vs LQ | Delta SSIM vs LQ | Delta LPIPS vs LQ |
|---|---:|---:|---:|---:|---:|---:|
| LQ | 28.034420 | 0.777699 | 0.356532 | 0.000000 | 0.000000 | 0.000000 |
| HYPIR-200 | 24.528587 | 0.684926 | 0.461005 | -3.505833 | -0.092773 | +0.104473 |
| HYPIR-50 | **28.257533** | **0.776935** | 0.338965 | **+0.223113** | -0.000764 | -0.017567 |
| fixed_50_50 | 26.858639 | 0.747600 | 0.365744 | -1.175781 | -0.030098 | +0.009211 |
| confidence_200_50 | 27.291166 | 0.766543 | 0.332414 | -0.743254 | -0.011156 | -0.024118 |
| confidence_200_LQ | 28.048711 | 0.776696 | **0.317280** | +0.014292 | -0.001002 | **-0.039252** |

HYPIR-50 is the best global baseline on PSNR/SSIM and confidence_200_LQ is closest to LQ on PSNR/SSIM while having the lowest LPIPS. The latter should be read as conservative similarity, not proof of restoration.

## Per-case metric observations

- `case1` text: HYPIR-200 is substantially worse (29.34 dB) while HYPIR-50 is near/slightly above LQ (32.14 dB). confidence_200_LQ is more conservative (31.56 dB), and confidence_200_50 improves SSIM over fixed blending but remains below LQ.
- `case2` book spines: HYPIR-50 and confidence_200_LQ exceed LQ PSNR (28.98 and 29.11 dB). This does not establish character identity; small strokes remain a visual risk.
- `case3` bird: every generated blend remains below LQ PSNR; confidence_200_LQ is the least damaging generated option but does not recover the full GT texture.
- `case4` foliage: HYPIR-50 is only marginally above LQ on PSNR/SSIM, while all blends lose PSNR. The images show less aggressive object-like texture than HYPIR-200, but no semantic label is assigned; the previously noted animal-like appearance is treated only as a visual structure.
- `case5` clock: HYPIR-50 and both confidence variants improve PSNR over LQ; confidence_200_50 also improves SSIM. Digits, pointers, and bezel geometry still require identity-level inspection beyond these aggregate metrics.

## Region analysis

Masks are the per-case median of the smoothed map, so each region contains approximately half the pixels. Average error to GT:

| Method | High L1 | Low L1 | High gradient diff | Low gradient diff |
|---|---:|---:|---:|---:|
| LQ | 11.717 | 5.260 | 4.099 | 1.951 |
| HYPIR-200 | 14.761 | 6.436 | 6.107 | 2.934 |
| HYPIR-50 | 11.278 | 5.400 | 4.052 | 2.010 |
| fixed_50_50 | 12.274 | 5.710 | 4.637 | 2.255 |
| confidence_200_50 | 11.858 | 5.427 | 4.379 | 1.992 |
| confidence_200_LQ | 11.358 | 5.261 | 4.138 | 1.914 |

Average change from LQ (not error to GT) is the more direct measure of spatial rewriting:

| Method | High change L1 | Low change L1 | High gradient change | Low gradient change |
|---|---:|---:|---:|---:|
| HYPIR-200 | 12.333 | 3.491 | 5.645 | 2.359 |
| HYPIR-50 | 4.260 | 1.134 | 1.708 | 0.547 |
| confidence_200_50 | 6.949 | 1.378 | 2.851 | 0.663 |
| confidence_200_LQ | 4.588 | 0.448 | 2.084 | 0.267 |

Thus HYPIR-200 changes the high-confidence half more than the low-confidence half, and the confidence blends attenuate low-confidence changes. This supports spatial control of *change amount*, but not the stronger claim that the map identifies where HYPIR-200 is correct.

## Visual evidence

Each `comparison/caseN.png` contains LQ, HYPIR-200, HYPIR-50, fixed 50/50, confidence 200/50, confidence 200/LQ, and GT. `caseN_error_maps.png` contains absolute-difference panels. `confidence_maps/caseN.png` and the two masks show the spatial weighting.

- **case1/case2 text:** HYPIR-200 shows the most stroke and band/texture rewriting. HYPIR-50 reduces it; confidence/LQ is more stable but can simply preserve blurred strokes. No claim of fewer incorrect characters is made without OCR or pixel-aligned identity testing.
- **case5 clock:** confidence blends keep bezel/digit/pointer changes between the two global baselines. The aggregate gains are compatible with stability, but do not prove exact pointer or numeral recovery.
- **case3 bird:** HYPIR-200 introduces the largest high-frequency and gradient changes. The confidence/LQ result reduces those changes, yet all generated methods remain below LQ PSNR, so this is risk reduction rather than demonstrated feather/water recovery.
- **case4 foliage:** HYPIR-200 is the most altered. The confidence/LQ and confidence 200/50 panels are visually less aggressive in blurred regions and show no obvious hard seam. The map does not establish whether any object-like/animal-like pattern is hallucinated; comparison is intentionally neutral about semantics.

## Q1-Q10

1. **Q1:** Yes, weakly. All five maps are smooth and nonconstant, with edge/contrast-correlated islands and broad low-response areas. They are heuristic structural reliability maps only.
2. **Q2:** Partly, but not as expected. HYPIR-200 changes more in high-confidence than low-confidence regions; its GT error is worse in both. The map does not isolate low-confidence error regions.
3. **Q3:** No. Fixed 50/50 loses 1.18 dB PSNR and 0.030 SSIM versus LQ on average.
4. **Q4:** No on aggregate. confidence_200_50 is below HYPIR-50 by 0.97 dB PSNR and 0.0104 SSIM, although its LPIPS is lower.
5. **Q5:** Yes as a conservative negative-control signal, not as restoration proof. confidence_200_LQ has the smallest low-region change and lowest LPIPS, but it largely stays near LQ.
6. **Q6:** Visually fewer large text rewrites occur than HYPIR-200, especially with HYPIR-50/LQ blending; exact stroke identity is unverified.
7. **Q7:** The confidence variants are more stable by PSNR/SSIM than HYPIR-200, with case5 PSNR gains over LQ. Exact digits/pointer correctness is not proven.
8. **Q8:** HYPIR-200 has the largest bird-region gradient changes; conservative blends reduce non-GT high-frequency changes, but do not restore GT reliably.
9. **Q9:** Yes, weakly. Case4 confidence blends reduce aggressive changes in low-response regions and avoid a hard seam; the map cannot certify that every reduced change is correct.
10. **Q10:** **WEAK EVIDENCE.** Spatially adaptive control is measurable and visually plausible, but the current heuristic does not beat HYPIR-50 globally or demonstrate true detail recovery in high-confidence areas.

## Restoration versus conservative fidelity

HYPIR-50's PSNR gain is small but real on the five-image average; it may include genuine recovery in some regions (notably case2/case5), while its low change from LQ also shows a conservative component. confidence_200_LQ has nearly LQ PSNR/SSIM and the lowest LPIPS, but its low-confidence L1 change is only 0.45, so much of its score comes from not changing the input. The present experiment therefore cannot claim that the fusion recovered LQ-missing GT details. A missing edge that remains missing is conservative fidelity, not restoration gain.

## Decision

Do not expand to a parameter sweep or new model in this phase. The result justifies a small follow-up focused on better reliability validation and targeted edge/detail checks, while retaining HYPIR-50 as the global reference and confidence/LQ as a safety control. If a future map cannot correlate with GT error or creates seams, stop the direction rather than adding complexity.
