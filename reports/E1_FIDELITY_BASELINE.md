# E1 Fidelity Baseline

E1 runs one independent, pretrained fidelity-oriented restoration baseline on
exactly the five validation cases. E0/HYPIR artifacts and source code were not
modified. The selected model is a reasonable pretrained fidelity baseline; the
JPEG40 training degradation is not assumed to exactly match this validation set.

## Environment

- GPU: NVIDIA GeForce RTX 5080 Laptop GPU (15.92 GiB reported device memory).
- CUDA: 12.8 runtime; `torch.cuda.is_available()` was `True`.
- PyTorch: `2.11.0+cu128`.
- Python: `3.11.16` (project `.conda` environment).
- BasicSR: `1.4.2`; scikit-image: `0.26.0`; LPIPS-Alex available.

## 1. Model

- Model: SwinIR-M color JPEG compression artifact reduction (`color_jpeg_car`).
- Official task definition: SwinIR release task 006, color JPEG artifact
  reduction at JPEG quality 40.
- Official sources used for verification:
  - README: `https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/README.md`
  - inference: `https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/main_test_swinir.py`
- The README lists `006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth` for the
  color JPEG40 task. The official inference code selects `color_jpeg_car`,
  sets `upscale=1`, and uses the configuration recorded below. This task and
  configuration were verified from code/source, not inferred from the filename.

## 2. Checkpoint

- Official release filename/URL (the `colorCAR` prefix is part of the official
  filename; it is the color JPEG40 CAR checkpoint corresponding to the requested
  CAR task):
  `https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth`
- Local path:
  `baseline/experiments/E1_fidelity/swinir_car_jpeg40/checkpoint/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth`
- File size: `102,873,665` bytes.
- SHA-256:
  `265c18d8809aaca0cd97a6283bee0ed1883ab88395e456381264cac2bb7b5867`
- Download date/source: 2026-09-03, official GitHub release URL.
- Checkpoint structure: top-level key `params`; 544 float32 tensors.
- Strict `SwinIR(**MODEL_CONFIG).load_state_dict(params, strict=True)` passed
  with all 544 keys matched.
- Detailed metadata is saved in
  `baseline/experiments/E1_fidelity/swinir_car_jpeg40/checkpoint_metadata.md`.

## 3. Inference Configuration

Official model configuration used:

```text
upscale=1, in_chans=3, img_size=126, window_size=7, img_range=255
depths=[6,6,6,6,6,6], embed_dim=180, num_heads=[6,6,6,6,6,6]
mlp_ratio=2, upsampler='', resi_connection='1conv'
```

Implementation and flow:

- Entrypoint: `baseline/experiments/e1_swinir_inference.py`.
- BasicSR `SwinIR` is loaded on CUDA and the checkpoint is loaded once.
- LQ uint8 RGB is converted to the official `[0,1]` tensor convention before
  model input; model output is converted back to `[0,255]` before PNG saving.
- `window_size=7`; tile size `504` and overlap `32`. `512` was not used because
  `512 % 7 != 0`. Each 4K case uses 63 tiles after padding.
- Padding is by reflected flips to a window-size multiple; output is cropped
  back to the original dimensions. Overlapping predictions are averaged.
- No TTA, ensemble, parameter sweep, finetuning, LoRA, SAM, or OCR.
- Smoke test: strict checkpoint compatibility, finite same-resolution output,
  and tiled shape preservation passed before validation inference.

## 4. Input / Output

- Input directory: `baseline/input` (`case1_lq.jpg` ... `case5_lq.jpg`).
- Ground truth directory: `csig_dataset/验证集` (`case*_gt.jpg`).
- Output directory:
  `baseline/experiments/E1_fidelity/swinir_car_jpeg40/output/`.
- Outputs are five independent RGB PNGs: `case1.png` through `case5.png`.

| Case | Input / GT / Output resolution | Actual LQ storage |
|---|---:|---|
| case1 | 4096 x 3072 | JPEG (effective quality approximately 75) |
| case2 | 3072 x 4096 | PNG data with `.jpg` extension |
| case3 | 4096 x 3072 | PNG data with `.jpg` extension |
| case4 | 4096 x 3072 | JPEG (effective quality approximately 75) |
| case5 | 4096 x 3072 | JPEG (effective quality approximately 75) |

The JPEG40 checkpoint is therefore treated as a reasonable pretrained fidelity
baseline, not as an exact match or an optimal model for these files.

## 5. Metrics

The evaluator follows the E0 definitions:

- PSNR: `skimage.metrics.peak_signal_noise_ratio(gt, pred_u8,
  data_range=255)` after rounding/clipping prediction to uint8.
- SSIM: `skimage.metrics.structural_similarity(gt, pred_u8,
  channel_axis=2, data_range=255)`.
- LPIPS: `lpips.LPIPS(net="alex")`; RGB tensors are mapped to `[-1,1]` and
  resized to maximum side 1024 before batched evaluation.
- Metric implementation reused from
  `control_conditioned_v1/control_conditioned_hypir_v1.py` (`_metric` and
  `_lpips_scores`). HYPIR-50 values are read from the existing E0
  `coeff_t_50` outputs through the same functions.
- Delta is `Fidelity - HYPIR-50`; lower LPIPS is better.

## 6. Per-case Results

### PSNR

| Case | HYPIR-50 PSNR | Fidelity PSNR | Delta |
|---|---:|---:|---:|
| case1 | 32.143319 | 32.105050 | -0.038270 |
| case2 | 28.976278 | 28.111951 | -0.864326 |
| case3 | 34.688713 | 35.404210 | +0.715497 |
| case4 | 18.084599 | 18.074936 | -0.009663 |
| case5 | 27.394758 | 26.446996 | -0.947762 |

### SSIM

| Case | HYPIR-50 SSIM | Fidelity SSIM | Delta |
|---|---:|---:|---:|
| case1 | 0.945300 | 0.951272 | +0.005972 |
| case2 | 0.825432 | 0.826285 | +0.000853 |
| case3 | 0.920412 | 0.932529 | +0.012116 |
| case4 | 0.310380 | 0.312053 | +0.001673 |
| case5 | 0.883153 | 0.878676 | -0.004477 |

### LPIPS-Alex

| Case | HYPIR-50 LPIPS | Fidelity LPIPS | Delta |
|---|---:|---:|---:|
| case1 | 0.064668 | 0.084784 | +0.020116 |
| case2 | 0.079454 | 0.109086 | +0.029633 |
| case3 | 0.074222 | 0.102273 | +0.028051 |
| case4 | 0.543473 | 0.663786 | +0.120313 |
| case5 | 0.048376 | 0.124192 | +0.075817 |

The complete machine-readable table is
`baseline/experiments/E1_fidelity/swinir_car_jpeg40/metrics.csv`.

## 7. Average Results

| Model | PSNR | SSIM | LPIPS-Alex |
|---|---:|---:|---:|
| HYPIR-50 | 28.257533 | 0.776935 | 0.162039 |
| Fidelity (SwinIR-M JPEG40) | 28.028629 | 0.780163 | 0.216824 |
| Delta (Fidelity - HYPIR-50) | -0.228905 | +0.003227 | +0.054786 |

Fidelity is higher on PSNR only for case3 and lower on the other four cases.
It is higher on SSIM for cases1-4 and lower for case5. LPIPS is higher for
all five cases (worse under the lower-is-better convention).

## 8. Runtime

- Command:
  `& .\\.conda\\python.exe baseline\\experiments\\e1_swinir_inference.py --device cuda --tile 504 --tile-overlap 32`
- Model load time: `0.547554 s`.
- SwinIR inference time per case:

| Case | Inference time (s) |
|---|---:|
| case1 | 156.498147 |
| case2 | 181.486418 |
| case3 | 186.059771 |
| case4 | 190.324722 |
| case5 | 197.396713 |
| Average | 182.353154 |

- End-to-end command wall time (including metric/LPIPS work): approximately
  `1112.3 s`.

## 9. GPU Memory

- Device: NVIDIA GeForce RTX 5080 Laptop GPU (CUDA available).
- Peak allocated during E1: `3.335263 GiB`.
- Peak reserved during E1: `4.683594 GiB`.
- Per-case peak values are in `metrics.csv`; no out-of-memory event occurred.

## 10. Comparison with HYPIR-50

The per-case tables above provide the requested PSNR, SSIM and LPIPS
comparisons. On the five-case arithmetic mean, Fidelity trades `-0.228905 dB`
PSNR for `+0.003227` SSIM and `+0.054786` LPIPS relative to HYPIR-50. The
average result is therefore mixed rather than a uniform improvement.

## 11. Qualitative Observations

The saved PNGs were visually inspected at the available display scale:

- All five outputs preserve the original composition, colors and geometry, and
  have the expected original resolution; no blank/invalid image was observed.
- case1 text layout remains aligned with the input/GT, with a modest change in
  text edge contrast; it is not a semantic OCR assessment.
- case2 book spines and text remain recognizable but visibly soft.
- case3 bird silhouette and water/background structure remain intact; fine
  texture is smooth at the inspected scale.
- case4 foliage remains smooth/soft with no claim that JPEG artifact removal
  solved the underlying difficult degradation.
- case5 clock geometry and building lines remain intact, with a smooth output.

These are visual observations only and do not establish or reject
hallucination risk.

## 12. Recommendation

SwinIR-M JPEG40 provides an independent, conservative reference with mixed
trade-offs: it improves average SSIM, loses average PSNR, and has worse average
LPIPS than HYPIR-50. Fidelity is better on PSNR for case3 only; HYPIR-50 is
better on PSNR for cases1, 2, 4 and 5. The next experiment decision should be
made by ChatGPT; no E2 experiment is started by this report.
