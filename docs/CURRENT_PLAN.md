# CSIG 赛题一：现行计划（2026-09-09）

**本文件取代 `DAS_HYPIR_final_plan.md`。** 旧 DAS-V1（blur 越大 → 生成越强）已废止。

依据：`baseline/experiments/error_decomposition_v1/report.md`（Phase0 + E1–E4）。

时间有限。目标不是把 HYPIR 研究完整，而是尽快得到一个可提交、可答辩的方法。

---

## Current problem definition

HYPIR-200 在当前推理设置下是 **确定性映射** \(LQ \mapsto \hat x\)（E4：四 seed 两两 PSNR 39.5–39.9 dB，鱼头/错叶型不变）。

| 区域 | 事实 |
|---|---|
| 低频（全图 1/16） | 布局/色块与 GT 一致 |
| 中频（1/8–1/4，局部 256） | **错误植物结构**（E1 高需求 A=7/8） |
| 高频 | 跟在错误中频后面，不是独立问题 |
| 局部 D | 鱼头等，同样锁死，不是随机抽样 |

因此：

- 不是 posterior sampling variance
- 不是「合理但不等于 GT 的叶脉」（C=0）
- 不是「模糊区生成过多高频」

H200 的价值：锐度、case4 LPIPS（0.377 vs LQ 0.654）。  
H200 的伤害：错误中频语义 + 大残差处的固定幻觉。  
H50 的 PSNR 优势主要来自 **少改**（E3），不是更好的 restoration。

**方法目标（替换旧目标）：**

> 控制 HYPIR-200 对 restoration 的贡献：保留其纹理/锐度/感知收益，抑制错误中频和大残差幻觉。  
> 不是增强 generation，也不是在模糊处加强 Adapter。

现行对照（必须超过的才值得继续；**全图** PSNR/SSIM）：

| 方法 | 均 PSNR | 均 SSIM | 备注 |
|---|---:|---:|---|
| LQ | 28.03 | 0.778 | 下限 |
| H50 | 28.26 | 0.777 | 少改 |
| **texture_selective_h200** | **28.48** | **0.781** | 当前 HYPIR 系锚 |
| H50+SwinIR α=0.6 | 28.51 | 0.784 | PSNR 最高；决赛需确认非纯扩散是否可交 |
| H200 | 24.53 | 0.685 | 不可直接交 |

分块对照（非重叠 256，每张 192 块，共 960；块均值 PSNR ≠ 全图 PSNR）：

| case | 全图 H200−LQ | 192 块均值 ΔPSNR | 块 LPIPS H200 vs LQ |
|---|---:|---:|---|
| 1 文字 | −2.69 | −4.66 | 更差（平坦块被造纹理） |
| 2 书脊 | −3.98 | −3.83 | 几乎持平 |
| 3 鸟 | −6.76 | **−8.93** | 更差；low/mid 水面最狠 |
| 4 绿植 | −1.70 | −1.64 | **更好**；三层 ΔPSNR 几乎一样 |
| 5 钟表 | −2.40 | −2.52 | 更差 |

CSV：`error_decomposition_v1/patch_metrics_full.csv`、`patch_metrics_summary.csv`、`e1_patch_metrics.csv`。块 LPIPS 是 256 原生 Alex，不可与全图 1024 协议混用。

---

## Discarded directions

现在不做，除非 fusion 已经超过 `texture_selective` 且仍有余量。

| 方向 | 原因 |
|---|---|
| 更多 seed | E4 已证确定性，边际信息为零 |
| 更多人工 patch 标注 | 主因已定位；24+12 足够决策。全网格 960 块的 PSNR/SSIM/LPIPS 已算完，不必再标 |
| Prompt 扫描 | 空 prompt 下映射已锁死；换 prompt 是另一条研究，赶不上 |
| 独立 P0 退化拟合研究 | 方向对，但是几天内的主路径太慢；不作为本阶段实验 |
| blur map 越大 → 生成越强（旧 DAS-V1） | 与 E1 相反：糊且大残差处正是错植物/鱼头 |
| 限制高频生成 | 错误在中频；压高频不治 A |
| Mixture-of-LoRA / GAN / Global Attention | 归因不清，赶不上 |
| reblur 训练损失 | 无可靠 \(D\)；A 类错叶型糊完仍可能对上 LQ |
| 100 张 test | 方法未冻结前禁止 |

---

## Highest priority experiments

**只有 1 个尚未做、必须做的实验。** 全局融合已经做过。

### Exp-F0（已完成，禁止重跑）

全图 \(I=(1-\alpha)LQ+\alpha H200\)，α∈{0.1,0.2,0.3,0.5}。  
数据：`error_decomposition_v1/e3_blend_curve.csv`。

五张平均 PSNR：

| α | PSNR | vs texture_selective 28.48 |
|---:|---:|---|
| 0.1 | 28.34 | 低 |
| **0.2** | **28.45** | **几乎打平** |
| 0.3 | 28.35 | 低 |
| 0.5 | 27.60 | 明显低 |
| H50 | 28.26 | 低 |

**结论：只降低 H200 全局比例，最多打平现有锚，赢不了。** 纯全局融合作为「最终方法」停止。α=0.2 保留为 fusion 下限对照。

若 fusion 无效（已经对全局成立）：说明 **H200 的错误中频不能靠均匀稀释消除**——要么空间上丢掉大残差区，要么改映射。

### Exp-F1（下一件唯一的新实验）

**残差置信融合（不用 blur map）。**

原因：E1 显示大改动区域就是 A/D；\(|H200-LQ|\) 是最便宜的「这里不可信」信号。Blur 大 ≠ 该生成。

最小实现（离线，复用现有 PNG，不跑 HYPIR）：

```text
R(x) = mean_c |H200 - LQ|          # [0,255]
R     = GaussianBlur(R, σ=8)
R     = percentile_norm(R, 1–99)
conf  = 1 - R                      # 残差大 → 不信 H200
α(x)  = α_min + (α_max-α_min)*conf
I     = (1-α) LQ + α H200
```

固定两档，禁止按 case 调：

- F1a：`α_min=0.00, α_max=0.30`
- F1b：`α_min=0.05, α_max=0.25`（贴近全局 0.2 的空间版）

可选 F1c（仅当 F1a/b 有正信号）：再乘 `edge_agree = clip(cos(∇H200, ∇LQ), 0, 1)`。  
第一版 **不做 reblur**（没有 \(D\)）。

**门槛（必须同时）：**

- 均 PSNR ≥ texture_selective **+0.05 dB**（28.53）
- SSIM 不掉 > 0.001
- LPIPS_1024 不差于 texture_selective 0.003 以上
- case1/2/5 无新的文字/指针灾难
- case4 `mid04` 鱼头必须被压掉（目视）

**若 F1 失败：** 说明用「H200 改了多少」当置信仍选不出可保留的 H200 区域 → **停止一切 H200 空间融合**，提交锚点方法。  
**若 F1 成功：** 说明伤害集中在大残差区，H200 的感知收益可以空间上留下来。

### Exp-F2（仅 F1 通过后）

同一套 conf，把 LQ 换成 H50：

```text
I = (1-α) H50 + α H200
```

问：保真锚用 H50 是否比 LQ 更好。失败则保持 F1。

---

## Candidate method

**主候选（按序，后者替换前者）：**

1. **提交锚：** `texture_selective_h200`（或全局 α=0.2，二者几乎打平，先用已有锚）。
2. **若 F1 过门槛：** residual-confidence fusion（H200 只在低残差处留下）。
3. **若 F1+F2 过门槛：** H50 作保真底、H200 作受限纹理。
4. **Adapter 不进主路径。** 只有 F1 成功且仍有 ≥2 天，才允许一个 **反向 DAS**：在大残差处把 H200 **拉回** LQ/H50，而不是加强生成。禁止 blur-up Adapter。

**H50+SwinIR** 仅作 PSNR 参考。赛题要求 Diffusion；主提交必须 HYPIR 在环路上。SwinIR 融合最多当备份 zip，不作为故事主线。

---

## DAS-V1 重新判决

### 为什么旧方向是错的

旧式：\(h' = h + \alpha B \odot A(h,d)\)，\(B\) 大则 Adapter 强。

E1–E4：\(B\) 大的地方正是 **确定性错植物和鱼头**。把生成开大 = 放大锁死的错误映射。E4 证明这不是抽到的坏样本，换 seed 无效。

### 若还要动 Adapter，只能这样改

不是 blur → 生成↑。

而是：

```text
conf = 1 - norm(|H200 - LQ|)     # 或学出来的 conf，但不能用 GT
I    = LQ + conf * (HYPIR_adapter(LQ) - LQ)
```

Adapter 的任务是：**减小不可信残差**（identity / 拉回观测），不是补高频。  
Zero-conv 初始化必须等于当前提交锚。  
无 CSIG 匹配退化、无 F1 正信号，不准开训。

---

## Implementation order

```text
Current problem definition
        ↓
Discarded directions（本文上表，不再做）
        ↓
Exp-F0 已完成：全局融合打平锚、赢不了
        ↓
Exp-F1  residual-confidence fusion     ← 现在唯一要做的
        ↓
        失败 → 冻结 texture_selective / α=0.2，进入提交工程
        成功 → Exp-F2（H50 底板）
        ↓
        仍有时间且 F1 有空间收益 → 反向 Adapter（可选，可跳过）
        ↓
冻结一个方法 → 100 张 test → zip
```

| 阶段 | 输入 | 改哪里 | 预期 | 风险 |
|---|---|---|---|---|
| F1 | 现成 LQ/H200 PNG | 只改像素融合，不改 HYPIR | case4 鱼头消失；均 PSNR ≥28.53 | 大残差全图都大 → 退回 LQ，PSNR 不涨 |
| F2 | 现成 H50/H200 | 同上，底板换 H50 | 比 F1 再涨一点 | 与 texture_selective 重复 |
| 提交 | 冻结方法 | 推理+命名+jpg | 可交 | 过早跑 100 张 |

F1 预期工作量：半天写脚本 + 五张指标 + 目视 case4 `mid04`/`high02`。

---

## Competition submission plan

1. **现在：** 实现 F1a/F1b，和 `texture_selective_h200`、全局 α=0.2、H50、LQ 同一口径比。
2. **F1 不过：** 主提交 = `texture_selective_h200`（或 α=0.2，选五张均 PSNR 更高者）。不再开新方法。
3. **F1 过：** 主提交 = F1（或 F2）。答辩故事：HYPIR 提供感知锐度；用残差置信抑制确定性中频错误和幻觉。
4. **100 张：** 只跑冻结的那一套。输出 `output_dir/case{k}.jpg`。
5. **决赛 PPT：** 用 E1 的 A=7/8、E4 的鱼头四 seed 不变、F1 把鱼压掉（若发生）作为可控性证据。不要讲 blur-up Adapter。

---

## 科学停机条件（即使赶时间也保留）

| 结果 | 含义 | 下一步 |
|---|---|---|
| 全局融合打平、赢不了（已发生） | 均匀稀释去不掉错误中频 | 做空间置信，不做更多 α |
| F1 不过门槛 | 大残差与「该留的锐度」分不开 | 停融合，交锚点 |
| F1 过门槛 | 伤害集中在大残差区，H200 仍有可保留部分 | 可试 F2；仍不要训 GAN/MoE |
| Adapter 若做且过门槛 | 映射本身可被小模块纠正 | 才值得谈 DAS；且必须是 **拉回** 不是加强 |

不把「融合一定有效」写成结论。Exp-F0 已经表明 **全局融合不够**。F1 是对「空间上能否切开」的一次证伪，不是默认会赢。
