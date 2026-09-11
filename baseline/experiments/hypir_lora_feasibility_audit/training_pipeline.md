# HYPIR training pipeline（以本仓库代码为准）

入口：`HYPIR/train.py` → `SD2Trainer(config).run()`。配置：`HYPIR/configs/sd2_train.yaml`。

## 数据流

```text
parquet/txt 里的 GT 路径
        ↓
RealESRGANDataset.__getitem__     # HYPIR/HYPIR/dataset/realesrgan.py
  只读 GT（hq），另采样 blur kernel
  返回 {hq, kernel1, kernel2, sinc_kernel, txt}
        ↓
RealESRGANBatchTransform          # HYPIR/HYPIR/dataset/batch_transform.py
  USM sharp GT
  两段 blur / resize / noise / JPEG，stage2_scale=4
  再 bicubic 拉回 GT 尺寸
  返回 {GT, LQ, txt}
        ↓
BaseTrainer.prepare_batch_inputs  # trainer/base.py:256
  z_lq = VAE.encode(LQ).latent_dist.sample()
  timesteps = 全 batch 填 config.model_t（默认 200）
```

**不是**读磁盘上的 (LQ, GT) 对。  
**是** `GT → 随机 Real-ESRGAN 退化 → LQ`。  
每次 `__getitem__` 退化不同，不能稳定复现同一张 LQ，除非固定全部 RNG。

`load_file_meta`（`dataset/utils.py:14`）只接受 `.parquet` / `.txt` / `.list`。当前 yaml 的 `file_list: TODO`，脚本不能原样启动。

## LoRA 加载

`SD2Trainer.init_generator`（`trainer/sd2.py:37-58`）：

1. `UNet2DConditionModel.from_pretrained(SD2.1)`
2. `G.eval().requires_grad_(False)` 冻结整网
3. `LoraConfig(r=lora_rank, lora_alpha=lora_rank, init_lora_weights="gaussian", target_modules=config.lora_modules)`
4. `G.add_adapter(...)`；仅 LoRA 参数 `requires_grad`
5. LoRA 权重转 fp32

**不读取 `HYPIR/weights/HYPIR_sd2.pth`。** 官方训练是从高斯初始化新 LoRA，不是对现有 HYPIR mapping 做 continuation。

推理侧 `enhancer/sd2.py:20-35` 才 `torch.load(HYPIR_sd2.pth)` 并断言所有 lora key 匹配。

## 官方超参（yaml，不是 invent）

| 项 | 值 | 位置 |
|---|---|---|
| lora_rank | 256 | sd2_train.yaml:76 |
| lora_modules | to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj | :77 |
| model_t | 200（训练时固定，不是随机 t） | :74, trainer/base.py:264 |
| coeff_t | 200，用于 `scheduler.step(eps, coeff_t, z_in).pred_original_sample` | :75, trainer/sd2.py:90 |
| lr_G / lr_D | 1e-5 | :85-86 |
| batch_size | 6 | :4 |
| max_train_steps | 30000 | :93 |
| gradient_accumulation_steps | 1 | :94 |
| mixed_precision | bf16 | :91 |
| checkpointing_steps | 500 | :99 |
| lambda_l2 / lpips / gan | 1 / 5 / 0.5 | :82-84 |
| out_size | 512 | :17 |
| crop_type | none（要求图像已经是 512） | :18 |
| p_empty_prompt | 0.0 | :40 |
| use_ema | true, decay 0.999 | :78-79 |

本地 checkpoint `HYPIR_sd2.pth`：514 tensors，257 个 `lora_A` + 257 个 `lora_B`，259,474,432 参数，约 1.04 GB。

## Forward 与 loss

`forward_generator`（`trainer/sd2.py:83-92`）与推理同结构：单步 ε → x0 → VAE decode。训练 decode **不做** tile，也 **不做** `wavelet_reconstruction`（wavelet 只在 `enhancer/base.py:153` 推理末尾）。

Generator loss（`optimize_generator`，`trainer/base.py:277-285`）：

```text
L_G = 1 * MSE(x, GT) + 5 * LPIPS_VGG(x, GT) + 0.5 * D(x, for_G=True)
```

Discriminator：ConvNeXt `ImageConvNextDiscriminator`，真 GT / 假 G(x)，逐步与 G 交替（偶步 G、奇步 D，`run()` :327）。

有 reconstruction（L2）、perceptual（LPIPS VGG）、adversarial（ConvNeXt）。

## 冻结

| 模块 | 行为 |
|---|---|
| VAE | `eval(); requires_grad_(False)` |
| text encoder | 同上 |
| UNet 主干 | 先全部 freeze，再只解冻 LoRA |
| Discriminator | 可训练 |
| LPIPS VGG | freeze |

## Prompt

训练：parquet 的 `prompt_key`。README 示例用空字符串。yaml `p_empty_prompt: 0.0`。  
推理比赛：`EmptyCaptioner` → `""`。  
Prompt 进入 CLIP cross-attn，但当前比赛 mapping 是空 prompt 锁死的。

## 官方脚本能否原样复用？

`accelerate launch train.py --config configs/sd2_train.yaml` **现在不能跑**：`output_dir` 和 `file_list` 是 TODO。即使填了路径，它也是「新 LoRA + 合成退化」，不是「adapt HYPIR_sd2.pth 到 CSIG LQ」。没有 val loader、没有 early stopping、没有 CSIG 5-case hook。
