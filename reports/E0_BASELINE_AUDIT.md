# E0 Baseline Audit

审计范围严格限定为 Repository Audit + E0 Baseline Verification。未执行 E1，未安装 Fidelity Model，未修改 HYPIR，未训练或进行参数 sweep。

## 1. Environment

- GPU: NVIDIA GeForce RTX 5080 Laptop GPU; 1 device; 15.92 GiB total (audit-time free/total: 14.714/15.92 GiB)
- CUDA: 12.8 (PyTorch CUDA runtime)
- PyTorch: 2.11.0+cu128
- Python: 3.11.16 (Anaconda, `.conda` environment)
- LPIPS: installed and usable; `LPIPS(net="alex")`
- HYPIR repository commit: `b61d107c6cef38f01a93c7833558869731cfa8c1`

GPU memory above is an audit-time probe, not a recorded peak for the prior inference runs. Peak allocation was NOT FOUND in the existing logs.

## 2. Repository Structure

关键目录和文件（实际存在）：

- `HYPIR/test.py`: 批量 HYPIR inference entrypoint。
- `HYPIR/HYPIR/enhancer/base.py`: tiled VAE encode, generator forward, decode and wavelet reconstruction flow。
- `HYPIR/HYPIR/enhancer/sd2.py`: SD2 scheduler/text encoder/UNet-LoRA setup and generator forward。
- `HYPIR/configs/sd2_gradio.yaml`, `HYPIR/configs/sd2_train.yaml`: upstream configs; `sd2_gradio.yaml` 的 `weight_path` 仍为 `TODO`，不是本 E0 的实际运行配置。
- `HYPIR/models/stable-diffusion-2-1-base`: local Stable Diffusion 2.1 base model (25 files, approximately 5.16 GB total)。
- `HYPIR/weights/HYPIR_sd2.pth`: HYPIR LoRA checkpoint, 1,038,090,392 bytes; SHA-256 `d538a2cb925451fab1f75adfe715ac2b1c8bb12fa32a0851ba01be2347b354d`。
- `baseline/input`: five LQ inputs。
- `baseline/experiments/coeff_t_50`: source HYPIR-50 run, outputs/log/metadata/metrics。
- `baseline/experiments/coeff_t_200`: source HYPIR-200 run, outputs/log/metadata/metrics。
- `baseline/experiments/texture_weight_sweep_v2/fusion/texture_selective`: existing texture-selective reference outputs。
- `csig_dataset/验证集`: five GT files plus five LQ-named copies。
- `control_conditioned_v1`: Control v1 source, metrics, region diagnostics, alpha maps and comparison images。
- `baseline/evaluate_metrics.py`: generic LQ/GT/output PSNR, SSIM and LPIPS evaluator。
- `tests/test_control_conditioned_hypir_v1.py`, `tests/test_evaluate_metrics.py`: relevant unit tests。

当前主要输出目录：

- Control v1 artifacts: `control_conditioned_v1/`
- HYPIR-50 PNGs: `baseline/experiments/coeff_t_50/output/result/`
- HYPIR-200 PNGs: `baseline/experiments/coeff_t_200/output/result/`
- Existing original baseline output: `baseline/output/result/`

## 3. HYPIR Inference

批量入口为 `HYPIR/test.py`。实际调用流程是：设置 seed -> 构造 `SD2Enhancer` -> 加载 scheduler、CLIP tokenizer/text encoder、VAE、SD2 UNet 和 HYPIR LoRA -> 对每个 LQ 图像执行 `model.enhance(...)` -> 保存 PNG 和 prompt 文本。

`BaseEnhancer.enhance()` 使用 tiled VAE encoding、tiled generator forward、tiled VAE decoding，最后执行 wavelet reconstruction，并恢复到输入分辨率。`SD2Enhancer.forward_generator()` 在 `model_t` timestep 上运行 LoRA UNet，再以 `coeff_t` 调用 scheduler step。

实际入口和代码位置：

- CLI/inference loop: `HYPIR/test.py:14-147`
- tiled enhance flow: `HYPIR/HYPIR/enhancer/base.py:66-160`
- SD2/LoRA loading and forward: `HYPIR/HYPIR/enhancer/sd2.py:9-64`

## 4. Control v1

- 修改内容: 新增 `control_conditioned_v1/control_conditioned_hypir_v1.py` 离线 MVP；未发现 HYPIR tracked source 修改。
- Control 信号进入位置: HYPIR 推理完成后，在图像空间计算 LQ-only alpha map，然后应用 `F = H50 + alpha * (H200 - H50)`；没有进入 UNet、VAE 或 scheduler 内部。
- LQ 特征: Sobel gradient、local variance、inverse-sharpness blur proxy、high-frequency energy、local contrast。alpha 上限为 0.35；代码包含 moderate-gradient support、strong-edge penalty 和 unsupported-detail penalty。
- Checkpoint: 复用 `HYPIR/weights/HYPIR_sd2.pth`（见 SHA-256 above）；Control v1 本身不加载新模型。
- Config: `control_conditioned_v1/experiment_metadata.md` 记录目录、公式和参数；脚本默认根目录和参数位于 `control_conditioned_v1/control_conditioned_hypir_v1.py:436-454`。没有单独的 Control YAML。
- Inference parameters inherited by source HYPIR outputs: `base_model_type=sd2`, `model_t=200`, `coeff_t=50` or `200`, `lora_rank=256`, modules `to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj`, `patch_size=512`, `stride=256`, `scale_by=factor`, `upscale=1`, `captioner=empty`, `seed=231`, `device=cuda`。

现有 Control v1 目录保存 alpha maps、comparison images、CSV 和 Markdown；没有单独保存 `control_conditioned_v1` fused PNG。融合图像在脚本运行时生成并用于指标计算。

## 5. Dataset

Validation LQ directory is `baseline/input`; GT directory is `csig_dataset/验证集`。实际 case 为 `case1` 至 `case5`，每个 case 均有 LQ 和 GT，且 H50/H200/reference outputs 全部存在。

| Case | LQ / GT / H50 / H200 resolution | Files |
|---|---|---|
| case1 | 4096x3072, RGB | `case1_lq.jpg`, `case1_gt.jpg`, `case1_lq.png` (H50/H200 outputs) |
| case2 | 3072x4096, RGB | `case2_lq.jpg`, `case2_gt.jpg`, `case2_lq.png` (H50/H200 outputs) |
| case3 | 4096x3072, RGB | `case3_lq.jpg`, `case3_gt.jpg`, `case3_lq.png` (H50/H200 outputs) |
| case4 | 4096x3072, RGB | `case4_lq.jpg`, `case4_gt.jpg`, `case4_lq.png` (H50/H200 outputs) |
| case5 | 4096x3072, RGB | `case5_lq.jpg`, `case5_gt.jpg`, `case5_lq.png` (H50/H200 outputs) |

All checked LQ/GT/H50/H200/reference images are RGB and have matching dimensions per case. The GT directory also contains `case*_lq.jpg` copies; Control v1 selects the explicit `_gt` file for evaluation.

## 6. Metrics

E0 values below are read from `control_conditioned_v1/metrics.csv`, generated by `control_conditioned_v1/control_conditioned_hypir_v1.py`.

- PSNR: `skimage.metrics.peak_signal_noise_ratio(gt, pred_u8, data_range=255)`; predictions are rounded/clipped to uint8 first。
- SSIM: `skimage.metrics.structural_similarity(gt, pred_u8, channel_axis=2, data_range=255)`。
- LPIPS: `lpips.LPIPS(net="alex")`, prediction and GT converted to [-1, 1]; max-side resize to 1024 px before batching. `LPIPS-Alex enabled=True`。
- Generic evaluator location: `baseline/evaluate_metrics.py:169-218` (same PSNR/SSIM/LPIPS family and explicit dimension checks)。

## 7. E0 Results

已有 Control v1 结果可确认，无需重复运行。每个 case 的 Control v1 指标：

| Case | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| case1 | 32.140579 | 0.945184 | 0.063836 |
| case2 | 28.970673 | 0.825091 | 0.078362 |
| case3 | 34.593669 | 0.919747 | 0.076098 |
| case4 | 18.071665 | 0.309548 | 0.525369 |
| case5 | 27.382457 | 0.883552 | 0.047795 |

平均及对照：

| Model | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| LQ | 28.034420 | 0.777699 | 0.204318 |
| HYPIR-50 | 28.257533 | 0.776935 | 0.162039 |
| HYPIR-200 | 24.528587 | 0.684926 | 0.159664 |
| texture_selective_h200 | 28.480280 | 0.781421 | 0.164548 |
| Control v1 | **28.231809** | 0.776625 | 0.158292 |

Control v1 PSNR 与已知值 `28.231809` 一致（差值 0）。相对 `texture_selective_h200`，Control v1 平均 PSNR 较低 `0.248471 dB`，SSIM 较低 `0.004796`，LPIPS 较低 `0.006256`；没有同时超过该参考的三项门槛。

源 HYPIR inference metadata 中记录的时间（5-case 总耗时，不是单 case）：

| Source run | Total elapsed | Model load |
|---|---:|---:|
| coeff_t=50 | 342.774 s | 11.78 s |
| coeff_t=200 | 316.423 s | 11.95 s |

Control v1 离线融合耗时未记录；本次审计未重跑以免改变已保存 baseline。

## 8. Reproducibility

- HYPIR source commands are recorded verbatim in `baseline/experiments/coeff_t_50/experiment_metadata.md` and `baseline/experiments/coeff_t_200/experiment_metadata.md`。两次运行均使用 `.conda/python.exe HYPIR/test.py` 等价命令、seed 231 和上述固定参数。
- Control v1 entrypoint: `.conda/python.exe control_conditioned_v1/control_conditioned_hypir_v1.py`，默认读取 `baseline/input`、`csig_dataset/验证集`、`coeff_t_50`、`coeff_t_200` 和 `texture_selective` 目录。
- To preserve existing artifacts, the Control v1 script refuses to overwrite a non-empty output directory; a rerun requires a new empty `--output-dir`.
- Existing output path: `control_conditioned_v1/`。Source HYPIR output paths are listed in Sections 2 and 4。
- Seed: source HYPIR `231`; Control v1 fusion code has no random operation/seed parameter。
- Checkpoint: local `HYPIR/weights/HYPIR_sd2.pth`, SHA-256 recorded above。

相关单元测试：`python -m unittest tests.test_control_conditioned_hypir_v1 tests.test_evaluate_metrics` -> **8 tests passed**。`pytest` 未安装（`No module named pytest`），未新增依赖。

## 9. Problems / Risks

1. **REPORTING INCONSISTENCY**: `control_conditioned_v1/summary.md` reports Control v1 average SSIM as `0.776624`, while `metrics.csv` reports `0.776625`。`summary.md` reports reference LPIPS `0.164546`, while the CSV average for `texture_selective_h200` is `0.164548`; the script constants use the former threshold. Differences are at 1e-6 to 2e-6 scale and do not affect the known Control v1 PSNR value, but the CSV and hard-coded reference should be distinguished when comparing thresholds.
2. Control v1 has no standalone fused output PNG; its fused image is generated in memory and only represented in comparison panels and metrics. Independent re-evaluation therefore requires rerunning the script into a new output directory.
3. Peak GPU memory and Control v1 offline fusion wall-clock time were NOT FOUND in existing artifacts.
4. `pytest` is NOT FOUND in the current environment; the equivalent relevant `unittest` suite passed 8/8.

未发现 HYPIR source code bug that changes the existing E0 values；未执行修复。

## 10. Recommendation for Next Experiment

建议交由 ChatGPT 判断下一步。

