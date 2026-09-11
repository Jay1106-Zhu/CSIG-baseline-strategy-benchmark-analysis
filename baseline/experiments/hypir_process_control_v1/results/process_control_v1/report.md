# HYPIR process control v1 裁决（2026-09-11）

离线。未改 HYPIR，未重跑扩散，未覆盖 fusion_v1 / fusion_v2 / texture_selective。  
LPIPS = Alex，最长边 1024（与 fusion 锚同一口径）。

## 1. 当前到底有哪些真正能控制生成自由度的参数？

以 `HYPIR/HYPIR/enhancer/sd2.py` `forward_generator` 为准，不是论文：

```text
z_in = Encode(LQ) * scaling_factor
eps  = UNet(z_in, t=model_t, text)          # 一次前向
x0   = scheduler.step(eps, coeff_t, z_in).pred_original_sample
RGB  = wavelet(Decode(x0) 高频, LQ 低频)
```

| 参数 | 定义 | 使用处 | 实际改变 | 能否控制生成自由度 | 本轮是否适合 |
|---|---|---|---|---|---|
| **coeff_t** | `test.py --coeff_t` | `scheduler.step(eps, coeff_t, z_in)` | x0 公式里 `sqrt(1-α_t)`：t 越大，离 LQ 越远 | **是** | **是（已扫完）** |
| model_t | `test.py --model_t` | UNet timestep 嵌入 | 网络以为自己在哪个噪声档 | 否（降它 PSNR 更差） | 否 |
| sampling steps | 无 | 无 | 没有 denoising loop | 否 | 否 |
| noise injection | 无 | 无 | z_in 就是 LQ latent，不加噪 | 否 | 否 |
| start/end timestep | 无 | 无 | 单次 (model_t, coeff_t) | 否 | 否 |
| CFG / guidance | 无 | 无 | 没有 uncond 分支 | 否 | 否 |
| LoRA scale | 无 | 无 | 只 load adapter，没有 `set_adapter_scale` | 否 | 否（本轮不发明） |
| prompt | captioner | CLIP cross-attn | 当前是空字符串；E4 已锁死 | 否 | 否 |
| seed | `--seed` | VAE `.sample()` | E4 四 seed 两两 PSNR 39.5–39.9 | 否 | 否 |
| wavelet | `enhance()` 末尾 | 始终开启 | 生成高频 + LQ 低频；无 CLI | 否 | 否 |
| patch/stride/upscale | CLI | 分块 / 分辨率 | 不是强度 | 否 | 否 |

`coeff_t` 的 eps 尺度（`sqrt((1-α)/α)`）：50=0.226，75=0.287，100=0.344，150=0.456，200=0.572。这是连续的「离 LQ 走多远」，不是多步 denoise strength。

**不是「当前完全没有入口」。** 入口就是 `coeff_t`，而且 50/75/100/150/200 早已在同一 5 case 上跑完。本轮不重推理。

## 2. 哪一个参数最值得控制？

只选 **`coeff_t`**。

`model_t` 已经单独扫过（固定 coeff_t=200）：200→50 均 PSNR 24.53→22.25，视觉更不稳。它不是生成强度。

本轮档位：弱 50（H50）、中弱 75、中 100、中强 150、强 200（H200）。

## 3. 控制以后，Case 4 的错误植物结构是否真的减少？

**没有在「结构正确」的意义上减少。** 只是把同一套错误中频 prior 画得浅一些或深一些。

`crops/case4_fish.png`：

- LQ：粉色 blob
- t=50：仍是 blob，轮廓略清楚
- t=75：眼睛开始出现
- t=100–150：鱼头成型
- t=200：带眼睛的完整鱼头
- GT：褐色荚果

鱼头不是 t=200 才发明的。它在 mapping 里，`coeff_t` 只决定渲染到哪一步。t=50 不是「正确复叶」，只是没画完的鱼头。

`crops/case4_leaf.png`：锯齿单叶随 t 增大出现；GT 是复叶+黄花序。没有任何一档变成 GT。

`crops/case4_yellow_flower.png`：LQ 的黄条/粉团在 t↑ 后变成错误的柔荑花序+刺果；GT 是黄花+灯笼果。同样是 prior 被画实，不是结构被纠正。

粉色区域位置大体还在，颜色还在；变的是语义。

## 4. 是否比简单 α fusion 更有效？

**否。** 没有超过锚，也没有超过 fusion_v1。

| method | PSNR | SSIM | LPIPS_1024 | vs 28.48 |
|---|---:|---:|---:|---|
| LQ | 28.034 | 0.778 | 0.204 | ref |
| coeff_t_50 = H50 | 28.258 | 0.777 | 0.162 | 否 |
| coeff_t_75 | 27.958 | 0.771 | 0.155 | 否 |
| coeff_t_100 | 27.505 | 0.763 | 0.151 | 否 |
| coeff_t_150 | 26.213 | 0.734 | **0.149** | 否 |
| coeff_t_200 = H200 | 24.529 | 0.685 | 0.160 | 否 |
| fusion_v1_A | 28.460 | 0.782 | 0.189 | 否 |
| **texture_selective_h200** | **28.480** | 0.781 | 0.165 | 锚 |

PSNR/SSIM 随 `coeff_t` 单调下降。LPIPS 在 t=150 最好，那是「看起来更像纹理」的陷阱：case4 错植物会拉低 LPIPS。

逐 case PSNR：没有任何一档同时打过 texture 和 fusion_A。case3 水面 crop：t=50 已 42.15 vs LQ 46.64，假波纹随 t 增加；fusion_A 46.08 更接近 LQ。

简单 α / scene fusion 是在 **已经生成错结构之后** 往回拉。`coeff_t` 是在 **生成过程中少走几步**。两条路都到不了「高频对 + 中频对」。fusion_v1 的 28.46 仍然高于所有纯 `coeff_t` 档。

## 5. 下一步

情况判定：**C**（并带 B 的成分）。

- 视觉上更保守 = 把 H200 往 LQ/H50 拉，不是纠正 prior。
- 无论 50/75/100/150/200，HYPIR 只在 H50 ↔ H200 之间滑动。
- 无法同时得到「有用高频 + 正确中频」。
- 当前 inference-level 控制已经到上限。没有第二个未使用的强度旋钮值得再扫。

### 明确结论

**暂停过程控制进入 LoRA / Adapter。**

不要再扫 `coeff_t`、不要发明 LoRA scale / wavelet 混合权重当「过程控制」、不要再堆 mask。  
本轮不实施 LoRA。下一阶段如果做，必须改 mapping 本身，而不是再拧 `coeff_t`。

## 其他 case（简记）

- **case3 鸟：** 身份保持。t↑ 羽毛/水面假纹理增加；t=50 仍不如 LQ/fusion 干净。
- **case1 文字：** 「整」字形保持；t↑ 笔画更脏、更生成。fusion_A / texture 更稳。
- **case2 书脊：** t=50 PSNR 28.98 好于 LQ，但仍低于 texture 29.24；t=200 红标过锐并偏色。
- **case5 钟表：** 指针/罗马数字在各档保持；t=50 全图 PSNR 27.39 略高于 texture 27.36，但平均被其他 case 拉下来。`crops/case5_clock.png` 256 框落到窗户上，表盘以 `comparison/case5.png` 为准。

## 产物

- `parameter_map.md`、`metrics.csv`、`crop_metrics.csv`、`summary.md`
- `comparison/case{1-5}.png`
- `crops/case4_{fish,leaf,yellow_flower}.png` 等
