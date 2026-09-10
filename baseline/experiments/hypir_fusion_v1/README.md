# HYPIR fusion v1

独立、离线的 **output-space controlled fusion** 实验。不修改 HYPIR 源码，不训练 LoRA / 分类器 / Adapter，不引入新模型，不跑扩散。

## 实验目的

验证：在已有 HYPIR-50 / HYPIR-200 输出上做可控融合，是否优于裸 HYPIR-200。

HYPIR-200 不能直接作为最终输出：它会写入确定性中频结构错误（错叶型、鱼头幻觉、水面假纹理）。换 seed、调 `coeff_t` 无效。本实验把 H200 当成 **detail candidate**，把 LQ / H50 当成 **structure anchor**，用结构 mask 抑制 hallucination。

## 方法

```text
out_Y = base_Y + alpha * mask * (H200_Y - base_Y)
Cb, Cr = LQ
```

| 方案 | base | detail |
|---|---|---|
| A | LQ | H200 − LQ |
| B | H50 | H200 − H50 |

H200 = detail candidate。LQ / H50 = structure anchor。Structure mask = hallucination suppression。

融合只在 **Y** 上进行。**Cb/Cr 始终来自 LQ**，避免 H200 色调漂移。不要做 RGB 直混。

### Structure mask（传统图像处理）

输入：LQ Y、H200 Y。无语义模型。

1. resize 到 1/4
2. Sobel 梯度
3. 方向一致性：`cos = (g1·g2) / (||g1|| ||g2|| + eps)`，再 `clip(cos, 0, 1)`
4. 幅值一致性：`gradient_similarity = 1 - |m1-m2| / (m1+m2+eps)`
5. `mask = clip(cos, 0, 1) * gradient_similarity`
6. Gaussian blur，再 resize 回原尺寸

LQ 与 H200 结构一致 → mask 接近 1（允许细节）。  
H200 突然出现新轮廓 → mask 降低（抑制幻觉）。

平坦区梯度方向无定义，不把 `0/eps` 当成结构冲突；若 LQ 平坦而 H200 长出新边，则仍降权。

### Scene-specific alpha（手动，不是分类器）

| scene | case | alpha |
|---|---|---|
| text | case1 文字 | 0.25 |
| book | case2 书脊 | 0.30 |
| bird | case3 鸟 | 0.12（范围 0.10–0.15） |
| plant | case4 绿植 | 0.08（范围 0.05–0.10） |
| clock | case5 钟表 | 0.30 |

未知 case 回退 `--default-alpha 0.15`。

## 输入 / 输出

通用接口：

```text
data/LQ/
data/H50/
data/H200/
results/fusion_v1/
```

五个验证 case 若 `data/` 为空，则自动使用项目已有路径：

- LQ: `baseline/input/`
- H50: `baseline/experiments/coeff_t_50/output/result/`
- H200: `baseline/experiments/coeff_t_200/output/result/`
- GT（只评测）: `csig_dataset/验证集/`

输出：

- `fusion/A/`、`fusion/B/`：融合结果
- `masks/`：结构 mask（彩色 / 灰度 / α·mask）
- `heatmaps/`：`|H200-LQ|` 与被抑制残差
- `comparison/`：LQ | H50 | H200 | Fusion | GT
- `crops/`：case3 水面、case4 鱼头等诊断块
- `metrics.csv`、`summary.md`

## 运行

先不要跑 100 张。默认只跑 case1–case5。

```powershell
& .\.conda\python.exe baseline\experiments\hypir_fusion_v1\experiment.py --root .
```

自定义三个文件夹：

```powershell
& .\.conda\python.exe baseline\experiments\hypir_fusion_v1\experiment.py `
  --lq-dir path\to\LQ `
  --h50-dir path\to\H50 `
  --h200-dir path\to\H200 `
  --output-dir baseline\experiments\hypir_fusion_v1\results\fusion_v1
```

只跑方案 A、关掉 LPIPS、改绿植 alpha：

```powershell
& .\.conda\python.exe baseline\experiments\hypir_fusion_v1\experiment.py `
  --scheme A --no-lpips --alpha-plant 0.05
```

单元测试：

```powershell
& .\.conda\python.exe -m unittest tests.test_hypir_fusion_v1
```

指标口径与现有实验对齐：PSNR/SSIM 原分辨率；LPIPS-Alex 最长边 1024 双线性。不要和 `coeff_t` 的 native 4K LPIPS 混用。

## 观察重点

| case | 看什么 |
|---|---|
| case4 绿植 | 鱼头 / 错误叶片是否被压住 |
| case3 鸟 | 水面错误纹理是否减少 |
| case1 / 2 / 5 | 文字和钟表结构是否保持 |

对照：LQ、H50、裸 H200。当前 HYPIR 系锚是 `texture_selective_h200`（均 PSNR ≈ 28.48）。全局 α=0.2 只能打平锚，赢不了；本实验问的是 **空间 mask + 场景 α + Y 通道** 能否切开「可保留锐度」和「错误中频」。

## 明确不做（延期）

- Degradation Encoder
- Adapter
- LoRA
- Diffusion finetune
- ControlNet
- Global Attention

## 本次 5-case 结果（2026-09-10）

| method | PSNR | SSIM | LPIPS_1024 |
|---|---:|---:|---:|
| fusion_A | 28.460 | 0.782 | 0.189 |
| fusion_B | 28.256 | 0.779 | 0.156 |
| texture_selective_h200 | 28.480 | 0.781 | 0.165 |
| HYPIR-200 | 24.529 | 0.685 | 0.160 |

相对裸 H200 有效；相对当前锚未超过（差 0.02 dB，门槛是 +0.05）。  
Fusion **不是识别并删除鱼头**：plant α=0.08 把 H200 残差压回去，鱼头变回模糊粉团。Sobel mask 能压新强边缘（错叶轮廓），压不了平滑区语义。文字/钟表结构保持。详见 `results/fusion_v1/report.md`。

## 若有效，下一步（V1.1）

V1 是保守融合。下一步不是单独用「大残差=坏」（会杀掉文字），而是：

```
M_final = structure_mask × residual_penalty
```

允许提高 α，让有用细节回来，同时抑制「大残差 + 结构不一致」的无依据生成。不要开 LoRA。
