# Structure-Anchored HYPIR local diagnosis

## Executive conclusion

Across 3565 native-resolution 256x256 (stride 128) patches, H200 has 15 Type-A useful-change patches, 865 Type-B/Type-D harmful-or-mismatch patches, and 10 Type-C needs-restoration-but-unchanged patches. These are quantile-defined diagnostic categories, not semantic labels.
The mean patch L1 improvement is H200=-2.1084 and H50=0.1556; the global metrics already show H50 is the safer average choice. H200 nevertheless has localized positive changes, so the evidence supports studying structure-anchored fusion, not starting LoRA training.
LQ-only correlations are exploratory. The strongest observed feature-to-H200-improvement Spearman association is blur_proxy (Spearman=0.149); this does not establish a reliable spatial controller without held-out cases or seeds.

## 1. Experiment purpose

Determine where HYPIR changes pixels, whether those changes align with GT-required changes, and whether simple LQ structure proxies predict useful versus harmful changes. GT is used only after inference for oracle analysis.

## 2. Data and existing outputs

Five validation pairs are compared at their original dimensions. H200 and H50 are reused from the completed coefficient experiments; no diffusion inference is run by this phase.

## 3. Patch analysis method

Each image is tiled with edge-aligned 256x256 patches at stride 128. L1/MSE/PSNR/SSIM, gradient and edge-density differences, LQ-to-output change, improvement, and change-direction cosine are recorded. Patch visualizations are downsampled only for display, never for metrics.

## 4. HYPIR-200 change analysis

H200 change magnitude and required change are compared directly in `patch_metrics.csv`; `generated_change_200_l1` is not treated as recovery by itself. Type A requires high required change, high generated change, and positive improvement; Type D flags high generated change with negative improvement or direction disagreement.

## 5. HYPIR-50 change analysis

H50 change maps and improvements are reported beside H200. The conservative setting generally reduces generated-change magnitude; where H50 remains near LQ and error does not worsen, that is recorded as conservative preservation rather than proof of correctness.

## 6. Useful versus harmful changes

The category counts above are the direct evidence: useful=15/3565, harmful-or-mismatch=865/3565, unchanged-but-needs-restoration=10/3565. Read the top/bottom 10% patch CSVs with the maps; full q10/q25/q50/q75/q90 values are in `patch_quantiles.csv`, and decision thresholds are in `metadata.md`.

### Per-case evidence

| case | H200 change L1 | H50 change L1 | H200 improvement L1 | H50 improvement L1 | A | B | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case1 | 3.623992 | 1.332490 | -1.300269 | -0.018551 | 1 | 0 | 0 | 176 | 271 | 35 |
| case2 | 8.811982 | 3.153747 | -2.248708 | 0.379698 | 0 | 11 | 0 | 168 | 503 | 21 |
| case3 | 5.090462 | 1.744698 | -3.139209 | -0.579550 | 0 | 2 | 0 | 177 | 5 | 8 |
| case4 | 13.592241 | 2.984115 | -2.606109 | 0.240049 | 0 | 1 | 10 | 178 | 690 | 0 |
| case5 | 8.680095 | 4.350329 | -1.247856 | 0.756578 | 14 | 4 | 0 | 148 | 551 | 78 |

## 7. Case1 text

H200 case1 mean improvement is -1.300269 L1 with 1 Type-A and 176 Type-D patches; H50 is -0.018551 L1. Text candidates are selected by high LQ-GT gradient difference and shown in `crops/case1_text_crops.png` with saved coordinates. Image-space evidence can distinguish edge/blur changes, but this study does not claim OCR or character correctness. Any apparent stroke changes must be judged against GT, not against sharpness alone.

## 8. Case2 book-spine text

H200 case2 mean improvement is -2.248708 L1 with 11 Type-B and 168 Type-D patches; H50 is 0.379698 L1. The same coordinate-audited crop protocol is used for the vertical spine-text case. The relevant question is whether H200's change lowers GT error and follows existing strokes; no OCR rate is asserted.

## 9. Case3 bird

H200 case3 mean improvement is -3.139209 L1 with 177 Type-D patches and no Type-A patches; H50 is -0.579550 L1. Bird, feather, leg, and water/background regions are not semantically detected. Crops are image-space candidates. Positive H200 changes count as evidence only where GT error decreases; extra high-frequency change with negative improvement is recorded as mismatch/harm.

## 10. Case4 foliage

H200 case4 mean improvement is -2.606109 L1 with 10 Type-C and 178 Type-D patches; H50 is 0.240049 L1. Ambiguous blurred foliage is analyzed as an image-space region. If H200 amplifies a low-frequency cue but diverges from GT, the appropriate description is ambiguous-structure amplification or an error interpretation; the report does not use emotional or semantic hallucination language.

## 11. Case5 clock

H200 case5 mean improvement is -1.247856 L1 with 14 Type-A and 148 Type-D patches; H50 is 0.756578 L1. Clock candidates are selected without OCR or object detection. Ring, numeral, tick, and pointer judgments require GT-aligned map/crop inspection. New edge energy alone is not evidence of a correct geometric reconstruction.

## 12. LQ-only feature correlation

Correlation rows are in `feature_correlation.csv`. Features are computed solely from LQ. A correlation with edge density means the patch has edges, not that diffusion should be allowed to generate there; stable LQ-driven control is not established by this five-image sample.

## 13. Restoration versus conservative fidelity

H50's lower generated-change magnitude and better global metrics are consistent with fewer harmful changes. That is not equivalent to recovering every missing detail. H200's Type-A patches provide localized evidence of beneficial changes, but average and per-case behavior must remain in the CSV evidence.

## 14. Support for Structure-Anchored HYPIR

Support is weak and conditional: post-hoc maps justify a structure-anchored fusion experiment, while the feature correlations are not strong enough to claim a deployable LQ-only reliability predictor.

## Direct answers Q1-Q10

Q1. No at the aggregate level: only 15/3565 patches meet the strict Type-A useful-change rule, while H200 mean improvement is -2.1084 L1 (negative).
Q2. 865/3565 patches are Type-B or Type-D harmful/mismatch candidates under the per-case 75th/25th-percentile thresholds; these are the measured image-space candidates for ‘changed too much or in the wrong direction,’ not semantic hallucination counts.
Q3. 10/3565 patches are Type-C high-required-change but low-generated-change candidates, so missed restoration exists but is less frequent than harmful/mismatch candidates under this rule.
Q4. Yes, the evidence is consistent with H50 gaining fidelity mainly by reducing changes: mean generated-change L1 is 2.7131 for H50 versus 7.9598 for H200, while mean improvement is 0.1556 versus -2.1084; this does not prove complete recovery.
Q5. Yes, but sparsely: 15 Type-A patches (case5 contributes 14, case1 contributes 1) show high required and generated change with positive GT-aligned L1 improvement.
Q6. Not reliably for improvement. LQ features correlate moderately with required change (for example gradient/entropy Spearman values are 0.730/0.752), but the strongest feature-to-H200-improvement Spearman is only 0.149 (blur_proxy).
Q7. The text crop sheets show LQ blur and GT stroke clarification; H200 adds high-frequency changes that must be checked against GT. The evidence favors structure-preserving deblur/edge restoration over unconstrained generation, without claiming OCR correctness.
Q8. The clock crop sheet shows geometry-sensitive edge changes; GT-aligned errors and the H200/H50 maps support adding geometric/edge constraints before permitting large changes.
Q9. Bird and foliage cases contain many Type-D candidates (177 and 178 respectively), consistent with an ambiguity-to-generation risk. The correct description is image-space mismatch or ambiguous-structure amplification, not a semantic claim.
Q10. Recommendation: A and B as small, validation-only next steps; do not start D (LoRA) or another broad C sweep yet. Keep E as a stop criterion if a held-out/seed check cannot reproduce the localized Type-A evidence.

## 15. Next-step recommendation

A: proceed with a minimal structure-anchored fusion prototype using only LQ-derived weights, with H50/LQ as the conservative fallback and GT used only for validation. B: edge-preserving local restoration is a useful control for text/clock. C: do not prioritize another broad coeff_t sweep before spatial evidence is understood. D: do not start LoRA. E: do not pause the direction yet, but stop if a held-out/seed check fails to reproduce useful localized changes.

## 16. Limitations

Only five validation pairs and one seed are available. Quantile categories are relative within each case; they are not calibrated probabilities. Crops are gradient-ranked candidates, not semantic detections. SSIM and gradient proxies are image-space measures and cannot establish OCR, object identity, or physical correctness. No GT-derived mask is used for inference.
