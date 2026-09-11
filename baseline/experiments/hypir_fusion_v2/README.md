# HYPIR fusion v2 (F1 residual confidence)

独立、离线的 **residual-confidence fusion**。不修改 HYPIR，不覆盖 `hypir_fusion_v1`，不训练，不重跑扩散，不加 LoRA / Adapter / 分类器 / 深度模型。

## 相对 v1 多了什么

v1 只有 Sobel 结构一致性 `M_struct`。平滑区内的语义幻觉（case4 鱼头内部）没有新边缘，`M_struct` 仍接近 1，H200 信息会漏进来。

v2 增加 H200–LQ 残差置信：

```text
conf    = 1 - normalize(mean_c |H200 - LQ|)
M_final = M_struct                         # A / B
M_final = M_struct * (conf ** gamma)       # gamma = 0.5, 1, 2
```

`normalize` = Gaussian blur σ=8，再按图像 1–99 百分位拉到 [0, 1]。残差空间上均匀（含 LQ≡H200）时 conf=1。

大残差且没有结构证据 → 当 hallucination，降权。  
只是补细节且与 LQ 一致 → 保留。

## 四组 mask

| 组 | 公式 | 说明 |
|---|---|---|
| A | `M_struct` | 原 fusion_v1 |
| B | `M_struct` | 与 A 同一公式，不重复写盘 |
| C | `M_struct * conf` | gamma=1 |
| D | `M_struct * conf^2` | gamma=2 |

额外写出 `M_struct * conf^0.5`（用户要求可测 gamma=0.5）。

其余与 v1 相同：scene alpha、Y 通道融合、Cb/Cr 锁 LQ、方案 A 底板 LQ、方案 B 底板 H50。

```text
out_Y = base_Y + alpha * M_final * (H200_Y - base_Y)
Cb, Cr = LQ
```

## 运行

默认只跑验证集 case1–case5，复用已有 H50/H200 PNG。

```powershell
& .\.conda\python.exe baseline\experiments\hypir_fusion_v2\experiment.py --root .
```

```powershell
& .\.conda\python.exe -m unittest tests.test_hypir_fusion_v2
```

## 输出

```text
results/fusion_v2/
  fusion/A_struct, B_struct, A_conf, B_conf, A_conf2, B_conf2, A_conf05, B_conf05
  masks/     struct, conf, final_* （彩色 / 灰度 / α·mask）
  heatmaps/  residual, residual_blur, residual_norm, suppressed_*
  comparison/
  crops/
  metrics.csv
  crop_metrics.csv
  summary.md
  report.md
```

## 明确不做

- 修改 HYPIR 源码
- 调用 HYPIR 重新推理
- 改 diffusion 参数
- LoRA / Adapter / 分类器 / 新深度模型
- 覆盖 fusion_v1
- 跑 100 张测试集

## 本次 5-case 结果（2026-09-11）

| method | PSNR | SSIM | LPIPS_1024 |
|---|---:|---:|---:|
| A/B fusion_A (`M_struct`) | 28.460 | 0.782 | 0.189 |
| C fusion_A_conf | 28.198 | 0.780 | 0.198 |
| D fusion_A_conf2 | 28.109 | 0.779 | 0.201 |
| texture_selective_h200 | **28.480** | 0.781 | 0.165 |

未超过 28.48。方案 A 乘 conf 单调掉向 LQ。鱼头置信图变暗，但 α=0.08 下输出仍是粉团，没有真绿植。详见 `results/fusion_v2/report.md`。
