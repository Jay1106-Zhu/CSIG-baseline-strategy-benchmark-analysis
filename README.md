# CSIG 生成式图像增强：策略与基准分析

赛道是同分辨率 4K 图像增强。测试集 100 张无 GT；本地验证只有 5 对 LQ/GT。算法需采用 Diffusion 架构，输入输出分辨率不变。

## 当前结论

- **主扩散对照：HYPIR-50**（`model_t=200`，`coeff_t=50`，`upscale=1`）。五张验证图上平均 PSNR 优于 Identity。
- **DiffIR Motion Deblurring 是弱对照，不是主线。** 官方权重在 GoPro/HIDE 运动去模糊上训练，和本赛题退化不匹配。五张图上平均 PSNR 低于 Identity。
- Identity（直接复制 LQ）仍是保真度下限。任何增强若整体掉到 Identity 以下，不能当提交候选。
- 默认 HYPIR（`coeff_t=200`）会改写文字、引入幻觉，不能直接提交。

指标摘要（验证集 5 张，详见 `baseline_bakeoff/`）：

| 方法 | PSNR | SSIM | LPIPS-Alex | DISTS |
| --- | ---: | ---: | ---: | ---: |
| Identity | 28.03 | **0.778** | 0.231 | 0.180 |
| HYPIR-50 | **28.26** | 0.777 | **0.181** | **0.161** |
| DiffIR-S2（本地 runner） | 27.79 | 0.775 | 0.205 | 0.179 |

## 仓库地图

```
README.md                 本文件
requirements-cu128.txt    项目 Python 依赖（CUDA 12.8）
.gitignore
baseline/                 HYPIR 实验脚本、指标、对比图
baseline_bakeoff/         DiffIR 对照：runner、输出、CSV、报告
csig_dataset/验证集/      5 对 LQ/GT（case1–case5）
docs/                     规划、交接、赛题原文
reports/                  E0–E3 文字报告
tests/                    bakeoff 契约测试
```

不在本仓库里（体积或许可原因，只记录 hash / 路径）：

- HYPIR 源码与权重：本机 `D:\MyProjects\CSIG\HYPIR\`
- DiffIR 官方源码：本机 `D:\MyProjects\CSIG\third_party\DiffIR\`，commit `293f86cdf313914ea0ffb2457ed032e4f1bd9dd2`
- DiffIR S2 权重：本机 `model_cache/diffir_motion_deblurring/Deblurring-DiffIRS2.pth`，SHA-256 `679fe3aae09a49a31517d1f04faed93dc85456050cde0df416a7ee4fe4297031`
- 测试集 100 张 LQ：本机 `D:\MyProjects\CSIG\csig_dataset\测试集\`

## 分支

| 分支 | 内容 |
| --- | --- |
| `main` | HYPIR 实验打包快照（`50ae047`） |
| `baseline-2-diffir-motion-deblurring` | 在 `main` 上增加 DiffIR bakeoff |
| 本分支 | 目录整理 + 官方 yaml 驱动的 DiffIR 复现 |

看 DiffIR 输出请打开：

- 既有本地 runner：[`baseline_bakeoff/outputs/diffir/`](baseline_bakeoff/outputs/diffir/)
- 官方 yaml + 分块兜底：[`baseline_bakeoff/outputs/diffir_official/`](baseline_bakeoff/outputs/diffir_official/)（本分支后续提交）
- 报告：[`baseline_bakeoff/BACKBONE_SELECTION_REPORT.md`](baseline_bakeoff/BACKBONE_SELECTION_REPORT.md)

## DiffIR 复现说明

官方评测入口是 `DiffIR/test.py -opt options/test_DiffIRS2.yml`，前向只吃 LQ。`scale: 4` 是 CPEN 的 PixelUnshuffle，不是把输出放大 4 倍。`val.window_size: 8` 只是 pad 到 8 的倍数，**没有 4K tile**。

在 RTX 5080 16GB 上，4K 全图会 OOM。本仓库的官方复现：

1. 用官方 yaml 的网络结构、`params_ema`、`timesteps: 4`、`manual_seed: 0` 和 pin 过的 S2 权重。
2. 先尝试全图；OOM 后用 Hann 加权 `tile=512 / overlap=128`（与 bakeoff runner 相同的内存兜底）。
3. 配置见 `baseline_bakeoff/options/test_DiffIRS2_csig.yml`。

这不是 `sh test.sh` 的逐行复现，而是官方配置 + 显存兜底。任务不匹配仍然成立：运动去模糊权重打在轻退化 4K 增强上，改动弱、PSNR 低于 Identity 是预期。

## HYPIR 复现（摘要）

验证集默认参数已跑通，但 `coeff_t=200` 会损害文字和几何。`coeff_t=50` 是当前验证集上更稳的参考。脚本和 CSV 在 `baseline/` 与 `baseline/experiments/`。

## 文档

- 赛题原文：`docs/赛题.txt`
- 交接：`docs/PROJECT_HANDOFF.md`
- 实验记录：`docs/findings.md`、`docs/progress.md`、`docs/task_plan.md`
- 策略报告：`reports/`
