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

**输出空间 H200 融合已经做完（F0、fusion_v1、F1），全部未过 28.48+0.05。** 不要再叠 mask。

**过程级控制也已经做完。** `coeff_t` 是唯一真实强度旋钮（单步 x0 尺度）。50/75/100/150/200 均 PSNR 全部低于 28.48。鱼头随 t 被画实，不是被纠正。

**LoRA feasibility（2026-09-11）：NO-GO。** 仓库没有可训配对；官方 pipeline 是 GT→4× Real-ESRGAN 合成 LQ，且不加载 `HYPIR_sd2.pth`。保持锚 `texture_selective_h200`。详见 `baseline/experiments/hypir_lora_feasibility_audit/go_no_go.md`。

**fusion_v3 multi-band（2026-09-11）：NO-GO。** 鱼头眼睛/鳞片在 H200 高频；B/C/D 均 PSNR 26.39–26.60，低于锚 28.48。冻结 `texture_selective_h200`，停止 output-space 微调。

**最终跑图策略（审计，未跑 100 张）：** 方案 U，全部 test 用 `texture_selective_h200`。测试集无类别标注；`scenario_routing_v1` 已失败（25.80）。fusion_v1 的 case→scene 表不能用测试文件名。详见 `baseline/experiments/final_strategy_audit/README.md`。

### hypir_process_control_v1（2026-09-11，已跑，到上限）

产物：`baseline/experiments/hypir_process_control_v1/results/process_control_v1/report.md`。

未重跑 HYPIR。复用 `coeff_t_*` PNG，LPIPS_1024 与 fusion 对齐。结论：暂停过程控制进入 LoRA/Adapter。

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

### hypir_fusion_v1（2026-09-10，已跑，不是 F1 本身）

用户指定的 output-space 融合：Y 通道 `base + α_scene · mask · (H200−base)`，mask = 1/4 Sobel cosine × 幅值相似度，Cb/Cr 锁 LQ。方案 A 底板 LQ，方案 B 底板 H50。

结果：`baseline/experiments/hypir_fusion_v1/results/fusion_v1/report.md`。fusion_A 均 PSNR **28.46**（锚 28.48，门槛 28.53）。Fusion 不是识别并删除鱼头，而是降低 H200 权重把生成压回模糊；Sobel mask 只切新强边缘。F1 不要单独用大残差当坏（会杀文字），应用 `structure_mask × residual_penalty`。

### Exp-F1（2026-09-11，已跑，未过门槛，禁止重跑）

用户指定实现：`M_final = M_struct * conf^gamma`，`conf = 1 - percentile_norm(|H200−LQ|)`，保留 v1 的 scene α / YCbCr / 双底板。不是 CURRENT_PLAN 里更早的全局 α(x) 配方。

产物：`baseline/experiments/hypir_fusion_v2/results/fusion_v2/report.md`。

| 组 | 公式 | 方案 A PSNR | 方案 B PSNR |
|---|---|---:|---:|
| A/B | `M_struct`（原 v1） | **28.460** | 28.256 |
| C | `M_struct * conf` | 28.198 | 28.366 |
| D | `M_struct * conf^2` | 28.109 | 28.368 |
| extra | `M_struct * conf^0.5` | 28.281 | 28.350 |
| 锚 | texture_selective | **28.480** | — |

门槛 28.53。未过。方案 A 随 gamma 单调掉向 LQ；方案 B+conf 相对 fusion_B 最多 +0.11 dB。置信图能在鱼头内部变暗，但 plant α=0.08 已把输出压成粉团；水面 conf≈0.91，假波纹不是大残差。文字/钟表因大残差被 conf 误伤。

**若 F1 失败（已发生）：** 用「H200 改了多少」当置信仍选不出可保留的 H200 区域 → **停止一切 H200 空间融合**，提交锚点方法。

当前瓶颈不是结构 mask，而是 H200 输出本身与 LQ 信息差异过大，需要进一步降低生成强度。

### Exp-F2（仅 F1 通过后；F1 已失败，不要做）

同一套 conf，把 LQ 换成 H50：

```text
I = (1-α) H50 + α H200
```

问：保真锚用 H50 是否比 LQ 更好。失败则保持 F1。

---

## Candidate method

**主候选（按序，后者替换前者）：**

1. **提交锚（F1 未替换）：** `texture_selective_h200`（或全局 α=0.2，二者几乎打平，先用已有锚）。
2. **F1 未过门槛：** 不要用 residual-confidence fusion 换锚。
3. **F2 不做。**
4. **Adapter 不进主路径。** F1 无正信号，不准开训。禁止 blur-up Adapter。

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
Exp-F1  residual-confidence fusion     ← 已跑，未过 28.48
        ↓
        失败 → 冻结 texture_selective / α=0.2，进入提交工程
        成功 → Exp-F2（H50 底板）
        ↓
冻结一个方法 → 100 张 test → zip
```

| 阶段 | 输入 | 改哪里 | 预期 | 实际 |
|---|---|---|---|---|
| F1 | 现成 LQ/H200 PNG | 只改像素融合，不改 HYPIR | case4 鱼头消失；均 PSNR ≥28.53 | 方案 A 28.46→28.20→28.11；未过锚。鱼头变暗但未变真叶 |
| F2 | 现成 H50/H200 | 同上，底板换 H50 | 比 F1 再涨一点 | **不要做**（F1 失败） |
| 提交 | 冻结方法 | 推理+命名+jpg | 可交 | 主提交仍是 texture_selective |

---

## Competition submission plan

1. **现在：** F1 已跑且不过门槛。停止 H200 空间融合。
2. **F1 不过（已发生）：** 主提交 = `texture_selective_h200`（或 α=0.2，选五张均 PSNR 更高者）。不再开新 mask。
3. **100 张：** 只跑冻结的那一套。输出 `output_dir/case{k}.jpg`。
4. **决赛 PPT：** 用 E1 的 A=7/8、E4 的鱼头四 seed 不变、fusion 把生成压回模糊（不是检测删除）作为可控性证据。不要讲 blur-up Adapter，也不要讲 F1 过了门槛。

---

## 科学停机条件（即使赶时间也保留）

| 结果 | 含义 | 下一步 |
|---|---|---|
| 全局融合打平、赢不了（已发生） | 均匀稀释去不掉错误中频 | 做空间置信，不做更多 α |
| F1 不过门槛 | 大残差与「该留的锐度」分不开 | 停融合，交锚点 |
| F1 过门槛 | 伤害集中在大残差区，H200 仍有可保留部分 | 可试 F2；仍不要训 GAN/MoE |
| Adapter 若做且过门槛 | 映射本身可被小模块纠正 | 才值得谈 DAS；且必须是 **拉回** 不是加强 |

不把「融合一定有效」写成结论。Exp-F0 已经表明 **全局融合不够**。F1 是对「空间上能否切开」的一次证伪，不是默认会赢。
