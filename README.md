# CSIG 生成式图像增强：策略与基准分析

赛道一：同分辨率 4K 图像增强（Diffusion）。验证集 5 对 LQ/GT；测试集 100 张无 GT。

**一眼看懂：** HYPIR-200 更清晰，但 PSNR 全面掉。诊断结论是 **确定性错误中频映射**（错叶型、锁死鱼头），不是随机抽样。当前提交锚 **冻结** `texture_selective_h200`（均 PSNR **28.48**）。F1 / process control / LoRA / multi-band 全部 NO-GO。100 张 test **全部**用 texture_selective_h200，不按类别 routing。

---

## 当前结论（2026-09-11）

| 阶段 | 结果 |
|---|---|
| fusion_v1 | 28.46，不是识别删除鱼头 |
| fusion_v2 F1 residual conf | 方案 A 28.20 / 28.11，未过锚 |
| process control `coeff_t` | 只在 H50↔H200 滑动，最好 28.26 |
| LoRA feasibility | **NO-GO**（无配对训练数据） |
| fusion_v3 multi-band | **NO-GO**（鱼头进入高频，最好 26.60） |
| 最终策略 | 100 张全部 `texture_selective_h200`；dry-run case1 已过 |

完整计划：[`docs/CURRENT_PLAN.md`](docs/CURRENT_PLAN.md)  
跑图封装：[`baseline/experiments/final_test_inference/`](baseline/experiments/final_test_inference/)  
策略审计：[`baseline/experiments/final_strategy_audit/`](baseline/experiments/final_strategy_audit/)

## 此前结论（2026-09-09 / 10）

| 事实 | 证据 |
|---|---|
| case4 主因是中频植物结构错（A=7/8），不是合理叶脉重采样 | E1 分层 256 crop |
| 全图 1/16 仍是同一棵树；局部 256 已换叶型 | E2 |
| 全局混合 H200 最多打平锚（α=0.2 → 28.45） | E3 |
| 四 seed 几乎同一张图（两两 PSNR 39.5–39.9），鱼头不变 | E4 |
| 块级：case3 平坦水面 ΔPSNR −8.9；case4 三层均匀约 −1.6 dB，LPIPS 更好 | 960 块网格 |
| fusion_A 均 PSNR 28.46；降权压回模糊，不是识别删除鱼头 | hypir_fusion_v1 |

完整报告：[`baseline/experiments/error_decomposition_v1/report.md`](baseline/experiments/error_decomposition_v1/report.md)  
现行计划：[`docs/CURRENT_PLAN.md`](docs/CURRENT_PLAN.md)（取代已废止的 `DAS_HYPIR_final_plan.md`）

### 全图对照（验证集 5 张）

| 方法 | PSNR | SSIM | 备注 |
| ---: | ---: | ---: | --- |
| Identity / LQ | 28.03 | **0.778** | 下限 |
| HYPIR-50 | 28.26 | 0.777 | 少改 |
| **texture_selective_h200** | **28.48** | **0.781** | 当前 HYPIR 系锚 |
| fusion_A（LQ + α·mask·H200，Y 通道） | 28.46 | 0.782 | 保守降权，未过门槛 |
| fusion_B（H50 底板） | 28.26 | 0.779 | LPIPS 最好（0.156） |
| H50+SwinIR α=0.6 | 28.51 | 0.784 | PSNR 最高；主提交需 Diffusion |
| HYPIR-200 | 24.53 | 0.685 | 不可直接交 |
| DiffIR-S2 | 27.79 | 0.775 | 运动去模糊权重，弱对照 |

---

## 先看这些图

四个诊断实验的图（含局部）都在 `baseline/experiments/error_decomposition_v1/`：

| 实验 | 看什么 | 路径 |
|---|---|---|
| **E1 局部 crop** | LQ / H50 / H200 / GT，含 1/4。case4 高需求 7/8 是错叶型；`mid04` 是鱼头 | [`e1_panels/case4/`](baseline/experiments/error_decomposition_v1/e1_panels/case4/) |
| E1 case3 | 鸟还在，水面被造纹理 | [`e1_panels/case3/`](baseline/experiments/error_decomposition_v1/e1_panels/case3/) |
| **E2 多尺度** | case4 1×…1/16：粗布局对、黄花层缺 | [`e2_multiscale/case4/`](baseline/experiments/error_decomposition_v1/e2_multiscale/case4/) |
| E3 混合预览 | case4 α=0 / 0.3 / 0.5 / 1.0 | [`e3_previews/`](baseline/experiments/error_decomposition_v1/e3_previews/) |
| **E4 四 seed 局部** | 同一套锯齿叶 / 柔荑花序 / 鱼头 | [`e4_analysis/key_crops.png`](baseline/experiments/error_decomposition_v1/e4_analysis/key_crops.png) |
| **fusion v1 鱼头** | 不是删除鱼头：α=0.08 把 H200 生成压回模糊粉团 | [`crops/case4_mid04_fish.png`](baseline/experiments/hypir_fusion_v1/results/fusion_v1/crops/case4_mid04_fish.png) |
| fusion v1 水面 | 假波纹变弱（小 α，不是 mask 切出） | [`crops/case3_low02_water.png`](baseline/experiments/hypir_fusion_v1/results/fusion_v1/crops/case3_low02_water.png) |
| fusion v1 五图面板 | LQ \| H50 \| H200 \| Fusion-A/B \| GT | [`comparison/`](baseline/experiments/hypir_fusion_v1/results/fusion_v1/comparison/) |

脚本与 CSV 同目录：`run_error_decomposition.py`、`e1_labels.csv`、`e3_blend_curve.csv`、`patch_metrics_summary.csv`。  
fusion v1：[`hypir_fusion_v1/`](baseline/experiments/hypir_fusion_v1/)（`fusion.py`、`experiment.py`、`metrics.csv`、`report.md`）。

未上传：4K 全分辨率 H200/H50 PNG、fusion 的 4K 输出/mask/heatmap（约 300 MB，见 `LOCAL_ONLY.txt`）、E4 的 22MB mean/median 全图、HYPIR/DiffIR 权重、`.conda`、测试集 100 张。

---

## 仓库地图

```
README.md                          本文件
docs/CURRENT_PLAN.md               现行比赛计划（锚已冻结）
docs/赛题.txt                      赛题原文
baseline/experiments/error_decomposition_v1/   E1–E4 诊断
baseline/experiments/hypir_fusion_v1/          输出空间融合 v1
baseline/experiments/hypir_fusion_v2/          F1 residual confidence（NO-GO）
baseline/experiments/hypir_process_control_v1/ coeff_t 过程控制裁决
baseline/experiments/hypir_lora_feasibility_audit/  LoRA NO-GO
baseline/experiments/hypir_fusion_v3/          Laplacian multi-band（NO-GO）
baseline/experiments/final_strategy_audit/     跑图前策略
baseline/experiments/final_test_inference/     100 张封装（dry-run 已过）
baseline/                          更早的 HYPIR 实验、指标、三联图
baseline_bakeoff/                  DiffIR 对照
csig_dataset/验证集/               5 对 LQ/GT
reports/                           E0–E3 早期文字报告
tests/                             契约测试
```

| 看什么 | 路径 |
|---|---|
| 问题定义与下一步 | [`docs/CURRENT_PLAN.md`](docs/CURRENT_PLAN.md) |
| E1–E4 报告 | [`error_decomposition_v1/report.md`](baseline/experiments/error_decomposition_v1/report.md) |
| fusion v1 报告 | [`hypir_fusion_v1/results/fusion_v1/report.md`](baseline/experiments/hypir_fusion_v1/results/fusion_v1/report.md) |
| HYPIR-50 输出 | [`baseline/experiments/coeff_t_50/output/result/`](baseline/experiments/coeff_t_50/output/result/) |
| DiffIR 对照报告 | [`baseline_bakeoff/BACKBONE_SELECTION_REPORT.md`](baseline_bakeoff/BACKBONE_SELECTION_REPORT.md) |
| 验证集 | [`csig_dataset/验证集/`](csig_dataset/验证集/) |

本机、不在 GitHub：

- HYPIR 源码与权重：`D:\MyProjects\CSIG\HYPIR\`
- DiffIR 源码：`D:\MyProjects\CSIG\third_party\DiffIR\`
- DiffIR S2 权重 SHA-256 `679fe3aae09a49a31517d1f04faed93dc85456050cde0df416a7ee4fe4297031`
- 测试集 100 张：`D:\MyProjects\CSIG\csig_dataset\测试集\`

---

## 下一步（不要再扩大研究）

1. **Exp-F1**：残差置信融合 \(I=(1-\alpha(x))LQ+\alpha(x)H200\)，\(\alpha\) 由 \(|H200-LQ|\) 决定（残差大 → 不信 H200）。不用 blur map。
2. 不过门槛 → 提交 `texture_selective_h200`。
3. 禁止：更多 seed、blur-up Adapter、未冻结就跑 100 张。

---

## DiffIR / HYPIR 早期基准（仍有效）

DiffIR Motion Deblurring 是弱对照：官方权重在 GoPro/HIDE 上训练，五张平均 PSNR 27.79，低于 Identity。官方 yaml + 4K tile 兜底见 `baseline_bakeoff/`。

HYPIR 默认 `coeff_t=200` 损害文字和几何；`coeff_t=50` 是更稳的强度参考，但 E3 表明其 PSNR 优势主要来自少改。
