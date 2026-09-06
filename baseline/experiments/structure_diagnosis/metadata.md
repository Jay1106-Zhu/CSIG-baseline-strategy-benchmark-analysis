# Structure-Anchored HYPIR diagnosis metadata

- date: 2026-09-01
- HYPIR commit: b61d107c6cef38f01a93c7833558869731cfa8c1
- base repo/path: `sd-research/stable-diffusion-2-1-base` / `HYPIR/models/stable-diffusion-2-1-base`
- LoRA SHA-256: `d538a2cb925451fab1f75adfe715ac2b1c8bb12fa32a0851ba01be2347b354d6`
- model_t: 200
- coeff_t: H200=200, H50=50
- seed: 231
- patch size: 256x256
- stride: 128
- upscale: 1
- scale_by: factor
- explicit constraint: This experiment performs no new diffusion inference; it reuses completed coeff_t=200 and coeff_t=50 outputs.
- GT is used only for post-hoc diagnostics and oracle maps; it is never used to construct inference masks.
- analyzer: `D:/MyProjects/CSIG/baseline/experiments/structure_diagnosis/analyze_structure.py`
- analyzer SHA-256: `4cbd76e34673939afe3fced555ccdcbbddefc65a2fd3c53994060938b026ca99`
- lora_rank: 256
- lora_modules: to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj
- captioner: empty

## Inputs and outputs

- LQ: `baseline/input/case*_lq.jpg`
- GT: `csig_dataset/验证集/case*_gt.jpg`
- H200: `baseline/experiments/coeff_t_200/output/result/case*_lq.png`
- H50: `baseline/experiments/coeff_t_50/output/result/case*_lq.png`
- output root: `baseline/experiments/structure_diagnosis/` (patch_metrics.csv, patch_summary.csv, patch_quantiles.csv, feature_correlation.csv, metadata.md, report.md, case1-case5/, oracle_maps/, change_maps/, crops/)

## Feature definitions

- `local_variance`: variance of mean-channel luminance in the patch.
- `gradient_magnitude`: mean Sobel-equivalent finite-difference gradient magnitude.
- `edge_density`: fraction of pixels whose finite-difference gradient magnitude exceeds 10 intensity units.
- `laplacian_magnitude`: mean absolute luminance Laplacian.
- `local_contrast`: luminance standard deviation divided by mean luminance plus 1e-6.
- `local_entropy`: entropy of a normalized 32-bin luminance histogram.
- `blur_proxy`: mean absolute Laplacian divided by mean gradient magnitude plus 1e-6.
- `required_change`: L1 distance |GT-LQ|; `generated_change`: |HYPIR-LQ|; `improvement`: LQ error minus HYPIR error.

## Per-case quantile thresholds

Thresholds are computed independently per case from patch values: low=25th percentile, high=75th percentile; improvement positive/negative are the 75th/25th percentiles of H200 L1 improvement.

- case1: required_low=1.022573, required_high=4.668142, generated_low=2.351964, generated_high=4.716019, improvement_positive=0.264481, improvement_negative=-1.761475
- case2: required_low=4.805735, required_high=7.564967, generated_low=5.877319, generated_high=11.420766, improvement_positive=0.519813, improvement_negative=-3.281854
- case3: required_low=0.930211, required_high=3.289119, generated_low=2.519699, generated_high=6.889826, improvement_positive=0.000000, improvement_negative=-4.071976
- case4: required_low=20.052912, required_high=27.563105, generated_low=9.821956, generated_high=17.270493, improvement_positive=0.375264, improvement_negative=-3.927858
- case5: required_low=5.663061, required_high=8.820491, generated_low=6.513779, generated_high=10.778854, improvement_positive=0.691966, improvement_negative=-1.940951
