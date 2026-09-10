# DAS-HYPIR：面向 CSIG 赛题的退化感知空间自适应 HYPIR 方案

> **已废止（2026-09-09）。** 现行计划见 [`CURRENT_PLAN.md`](CURRENT_PLAN.md)。  
> 诊断见 `baseline/experiments/error_decomposition_v1/report.md`（含 2026-09-10 分块全网格指标）。  
> E1–E4 证明 HYPIR-200 是确定性错误中频映射，不是「模糊处该加强生成」。Blur map → Adapter 增强 与诊断相反。

## 1. 方案概述

本方案面向 CSIG 赛题一的 4K 同分辨率图像增强任务，目标不是简单增强 HYPIR 的生成能力，而是解决 HYPIR 在当前赛题上的三个核心问题：

1. **训练退化分布与比赛退化分布不匹配**
2. **同一张图中不同位置退化程度不同**
3. **Diffusion 生成先验过强，容易在高频缺失区域产生幻觉**

因此，我们将最终方案定义为：

> **DAS-HYPIR：Degradation-Aware Spatial HYPIR**

核心思想是：

> **先让 HYPIR 学对比赛里的退化，再告诉它“这张图是什么退化、哪里退化严重”，最后用物理一致性约束它不要乱生成。**

最终 V1 只保留四个核心模块：

- CSIG-specific degradation modeling
- Continuous degradation representation
- Spatial blur map
- Lightweight spatial adapter

同时引入一个关键训练约束：

- Reblur consistency

第一阶段暂不加入：

- 多专家 LoRA / Mixture-of-LoRA
- 大型 Global Attention
- GAN
- 复杂语义检测器
- 硬分类路由

---

## 2. 问题重新定义

传统 HYPIR 可以近似理解为学习：

\[
p(x|y)
\]

其中：

- \(x\)：真实高清图像
- \(y\)：低质量输入图像

但 CSIG 的真实退化过程更合理地写成：

\[
y = D_{\theta,B}(x) + \epsilon
\]

其中：

- \(\theta\)：全局退化参数，如 blur、resize、JPEG、noise 等
- \(B(p)\)：不同空间位置的局部模糊程度
- \(\epsilon\)：残余噪声与编码误差

因此，我们希望模型学习的是：

\[
\boxed{p(x|y,\theta,B)}
\]

而不是只学习：

\[
p(x|y)
\]

理论依据是：当模型知道“发生了什么退化”和“退化发生在哪里”时，恢复问题的不确定性会降低：

\[
H(X|Y,\theta,B) \le H(X|Y)
\]

这意味着模型不需要过度依赖 diffusion prior 去猜测缺失细节。

---

## 3. 当前 HYPIR 在 CSIG 上的主要问题

### 3.1 退化域不匹配

HYPIR 原始训练依赖 Real-ESRGAN 风格的混合退化链路，包含：

- blur
- resize
- noise
- JPEG
- second-stage blur
- second-stage resize
- second-stage noise
- second-stage JPEG
- sinc filter
- resize back

其训练假设更接近通用 blind SR / ×4 restoration。

而 CSIG 赛题的输入输出保持同分辨率 4K，更接近：

\[
\text{HQ}
\rightarrow
\text{blur}
\rightarrow
\text{resampling}
\rightarrow
\text{mild noise / JPEG / ISP}
\rightarrow
\text{LQ}
\]

因此存在明显的 degradation domain gap。

### 3.2 原始 HYPIR 的生成先验过强

现有实验已经说明：

- 较强 HYPIR 生成会显著降低 PSNR / SSIM
- 降低生成强度后，指标反而提升
- 密集绿植等场景容易出现额外纹理甚至对象级 hallucination
- 文字、钟表、鸟类等结构敏感区域容易被重绘

因此当前瓶颈不是：

> HYPIR 不会生成高清细节

而是：

> **HYPIR 不知道什么时候该生成、什么时候不该生成。**

### 3.3 手工空间特征不够可靠

此前尝试过：

- Sobel edge
- local variance
- Laplacian blur proxy
- high-frequency residual

用于决定 HYPIR 在不同区域的增强强度。

这些方法能够一定程度保护边缘，但它们不能稳定预测：

> 某个区域是否真的需要更强 restoration。

因此，问题不是“空间控制”这个思想错误，而是：

> **手工 blur proxy 不够准确，需要 learned degradation representation。**

---

## 4. 最终网络结构

整体结构如下：

```text
                     ┌─────────────────────┐
                     │ Degradation Encoder │
                     └─────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
       Degradation Token d             Blur Map B(x,y)
       “是什么退化”                    “哪里退化”
                │                             │
                └──────────────┬──────────────┘
                               ▼
                    Spatial Residual Adapter
                               │
                               ▼
4K LQ ── VAE ── Frozen HYPIR / SD2 U-Net ── VAE ── HQ
                                                     │
                                                     ▼
                                           Known Degradation
                                                     │
                                                     ▼
                                               Reblur LQ
                                                     │
                                                     ▼
                                             Consistency Loss
```

---

## 5. 模块一：CSIG-Specific Degradation Simulator

### 5.1 目的

重新构造更贴近比赛分布的训练退化数据。

不再直接使用原始 HYPIR 的强 Real-ESRGAN ×4 风格退化，而改成：

```text
HQ
 │
 ├── Optical blur
 │     ├── Gaussian
 │     ├── anisotropic
 │     ├── defocus
 │     └── 少量 motion blur
 │
 ├── Spatially varying blur
 │
 ├── Same-resolution resampling
 │     └── 主要模拟 1× ~ 2× 范围的信息损失
 │
 ├── Optional mild noise
 │
 ├── Optional mild JPEG
 │
 └── Mild color / tone perturbation
       ↓
      LQ
```

### 5.2 空间变化模糊

传统训练常对整张 patch 使用同一个模糊核：

\[
y = K*x
\]

我们改为：

\[
y(p)=\sum_i w_i(p)(K_i*x)(p)
\]

其中：

- \(K_i\)：不同 blur kernel
- \(w_i(p)\)：空间权重
- \(B(p)\)：由这些权重定义出的 blur severity map

例如同一张 512×512 patch：

```text
左侧           中间           右侧
清晰           中度模糊       强模糊
B≈0.1          B≈0.4          B≈0.8
```

这样每一对合成训练数据天然拥有：

- HQ GT
- synthetic LQ
- degradation type
- degradation severity
- blur map GT
- forward degradation operator

无需人工标注。

---

## 6. Stage 0：Degradation Matching

这是整个方案真正开始训练前最重要的一步。

不能直接拍脑袋设置：

```text
motion = 20%
defocus = 30%
jpeg = 40%
```

而应该先使用官方 5 对 validation：

\[
(GT_i, LQ_i)
\]

对不同退化参数进行拟合：

\[
D_\theta(GT_i)
\]

然后比较 synthetic LQ 与真实 LQ 的接近程度。

建议统计：

- PSNR
- SSIM
- radial frequency attenuation
- gradient attenuation
- edge spread
- JPEG blockiness
- noise residual
- color shift
- local frequency distribution

最终得到一个较合理的：

\[
p_{\text{CSIG}}(\theta)
\]

Stage 0 不训练 HYPIR，只回答：

> **我们模拟的退化到底像不像比赛数据？**

---

## 7. 模块二：Degradation Encoder

### 7.1 为什么不用硬分类器

不采用：

```text
motion / defocus / jpeg / resize
五选一
```

因为实际退化往往是混合的：

\[
\text{defocus}
+
\text{resize}
+
\text{JPEG}
+
\text{mild noise}
\]

因此模型应该输出连续退化表征：

\[
d=E_g(y)
\]

其中：

\[
d\in\mathbb{R}^{C}
\]

它可以包含：

- blur type
- blur severity
- anisotropy
- resize severity
- JPEG severity
- noise severity
- optical degradation strength

分类标签仍然可以作为辅助监督，但真正送入 HYPIR 的不是 one-hot label，而是连续 degradation token。

---

## 8. 模块三：Spatial Blur Map

Degradation Encoder 的第二个输出：

\[
B=E_s(y)
\]

其中：

\[
B(p)\in[0,1]
\]

表示每个位置的信息损失程度。

多尺度形式：

\[
B_{64},B_{32},B_{16},B_8
\]

分别送入 U-Net 不同层。

直观理解：

- \(d\)：告诉模型“怎么坏的”
- \(B(p)\)：告诉模型“哪里坏了”

---

## 9. 模块四：Spatial Residual Adapter

第一版不重训整个 HYPIR。

保持：

```text
SD2.1 backbone       Frozen
HYPIR official LoRA  Frozen
```

仅训练：

```text
Degradation Encoder
Spatial Residual Adapter
```

Adapter 可写成：

\[
h_l'
=
h_l
+
\alpha B_l\odot A_l(h_l,d)
\]

其中：

- \(h_l\)：HYPIR 某层 feature
- \(B_l\)：对应尺度的 blur map
- \(d\)：全局 degradation token
- \(A_l\)：轻量 Adapter

这样：

### 清晰区域

\[
B\approx0
\]

Adapter 几乎不介入。

### 严重模糊区域

\[
B\approx1
\]

Adapter 强介入。

因此模型实现：

> **只对真正需要恢复的位置进行特化。**

---

## 10. 为什么 V1 不直接训练 LoRA

同学提出的 LoRA 思路保留，但后移。

原因：

1. HYPIR 官方 LoRA 本身规模已经较大
2. 同时训练 Encoder + Blur Map + Adapter + LoRA 会导致归因困难
3. 小验证集很容易出现过拟合
4. 我们首先需要验证“退化感知 + 空间控制”是否成立

因此：

### V1

```text
HYPIR backbone      Frozen
HYPIR LoRA          Frozen
Degradation Encoder Trainable
Spatial Adapter     Trainable
```

只有当 V1 证明：

> Blur Map 学对了，但恢复能力仍然不足

才进入 V2。

---

## 11. V2：Small Δ-LoRA

V2 增加：

\[
\Delta W_{\text{CSIG}}
\]

推荐 rank：

```text
8 / 16
```

而不是直接重新训练 rank-256 大 LoRA。

最终：

\[
W
=
W_{\text{HYPIR}}
+
\Delta W_{\text{CSIG}}
\]

它只学习：

> HYPIR 原 restoration prior 到 CSIG restoration prior 的差异。

---

## 12. Mixture-of-LoRA 作为后续可选扩展

如果实验进一步证明：

> 不同 degradation type 的最优参数确实明显不同

才考虑：

\[
\Delta W
=
\sum_i p_i\Delta W_i
\]

其中：

\[
p_i
\]

来自 degradation encoder。

例如：

```text
Optical LoRA      0.55
Resize LoRA       0.35
Compression LoRA  0.10
```

该方案不进入 V1。

---

## 13. Reblur Consistency

这是降低 hallucination 的关键约束。

模型输出：

\[
\hat x=F(y,d,B)
\]

synthetic training 中我们知道真实 forward degradation：

\[
D_{\theta,B}
\]

于是重新退化：

\[
\hat y=D_{\theta,B}(\hat x)
\]

要求：

\[
\boxed{
L_{\text{reblur}}
=
\|\hat y-y\|_1
}
\]

理论上可以理解为：

\[
x^*
=
\arg\max_x
p(x)\,p(y|x,D)
\]

其中：

- HYPIR 的 diffusion prior 主要提供 \(p(x)\)
- Reblur consistency 强化 \(p(y|x,D)\)

因此生成出来的高频细节不仅要“像真实图像”，还必须：

> **能够解释原始低质量观测。**

这可以针对密集绿植中的无依据纹理和对象级 hallucination。

---

## 14. 第一版 Loss

V1 不堆复杂损失。

建议：

\[
L=
\lambda_1L_{\text{Char}}
+
\lambda_2L_{\text{LPIPS}}
+
\lambda_3L_{\text{grad}}
+
\lambda_4L_{\text{deg}}
+
\lambda_5L_{\text{blur}}
+
\lambda_6L_{\text{reblur}}
\]

### 14.1 Charbonnier Loss

负责整体像素保真：

\[
L_{\text{Char}}
=
\sqrt{(\hat x-x)^2+\epsilon^2}
\]

### 14.2 Moderate LPIPS

负责一定程度的感知质量。

第一版权重不宜过高。

目标不是：

> 生成最锐利的纹理

而是：

> 在保真的前提下恢复合理细节。

### 14.3 Gradient Loss

\[
L_{\text{grad}}
=
\|
\nabla\hat x-\nabla x
\|_1
\]

主要保护：

- 中文文字
- 书脊
- 钟表数字
- 指针
- 鸟类轮廓
- 建筑边缘

### 14.4 Degradation Supervision

监督全局退化向量：

\[
L_{\text{deg}}
\]

用于学习：

> 怎么坏的。

### 14.5 Blur Map Supervision

\[
L_{\text{blur}}
=
\|
\hat B-B
\|_1
\]

用于学习：

> 哪里坏了。

### 14.6 Reblur Loss

\[
L_{\text{reblur}}
=
\|
D_{\theta,B}(\hat x)-y
\|_1
\]

用于限制无依据生成。

---

## 15. 为什么第一阶段不用 GAN

当前主要问题不是：

> 输出不够真实

而是：

> 模型生成得太自由。

因此第一版 GAN 建议直接关闭。

先验证：

> restoration 是否更真实、更保真

再考虑：

> perception 是否还需要增强。

---

## 16. 为什么暂时不做 Global Attention

密集绿植场景中，512 tile 确实可能缺少全局上下文。

但是 Global Attention 不一定天然降低 hallucination。

如果模型获得更强的“森林”语义，它也可能变得更敢生成：

- 树枝
- 花
- 鸟
- 其他自然纹理

因此 Global Context 不作为第一阶段主因。

只有当 E4 后仍能证明：

> 剩余错误与 tiled inference 的全局上下文缺失明显相关

才加入：

```text
Full LQ
  ↓
thumbnail
  ↓
small global encoder
  ↓
global LQ tokens
  ↓
Spatial Adapter
```

并且 global context 必须来自真实 LQ，而不是自由文本 prompt。

---

## 17. 核心理论：确定性恢复与生成式恢复应该解耦

模糊越严重，并不意味着 diffusion 应该越强。

合理逻辑应该是：

### Deterministic restoration strength

\[
G_{\text{det}}(p)
\uparrow
\quad
\text{when }
B(p)\uparrow
\]

即：

> 越模糊，越需要恢复。

但 generative strength 应该同时考虑：

\[
G_{\text{gen}}(p)
=
B(p)
\cdot
C(p)
\cdot
(1-S(p))
\]

其中：

- \(B(p)\)：退化严重度
- \(C(p)\)：恢复置信度
- \(S(p)\)：结构敏感程度

例如：

| 区域 | Blur | Structure sensitivity | Generative freedom |
|---|---:|---:|---:|
| 模糊绿植 | 高 | 中 | 中高 |
| 中文文字 | 高 | 极高 | 很低 |
| 钟表指针 | 高 | 极高 | 很低 |
| 鸟羽毛 | 中 | 中高 | 中 |
| 天空 | 中 | 低 | 很低 |

因此：

> **越模糊，不等于越应该自由生成。**

---

## 18. 实验路线

最终实验只按照以下顺序推进。

| 实验 | 内容 | 核心问题 |
|---|---|---|
| P0 | Degradation Matching | 我们是否模拟对了比赛退化？ |
| E0 | HYPIR-50 / 当前最优 baseline | 建立对照 |
| E1 | CSIG degradation + 普通小 Adapter | Domain alignment 是否有效？ |
| E2 | E1 + degradation token | “是什么退化”是否有用？ |
| E3 | E2 + spatial blur map | “哪里退化”是否有用？ |
| E4 | E3 + reblur consistency | 是否降低 hallucination？ |
| E5 | E4 + rank8/16 Δ-LoRA | Adapter 是否存在容量瓶颈？ |
| E6 | 可选 Global LQ Context | tile 全局信息是否是剩余瓶颈？ |

---

## 19. 关键消融关系

### E0 → E1

如果明显提升：

> 证明 HYPIR 和 CSIG 存在 degradation domain gap。

### E1 → E2

如果明显提升：

> 证明不同 degradation type / severity 需要条件化恢复。

### E2 → E3

如果明显提升：

> 证明 spatially varying degradation 是核心问题之一。

### E3 → E4

如果指标稳定且幻觉减少：

> 证明 physical degradation consistency 能约束 diffusion hallucination。

---

## 20. 评估标准

新模型不能只和原始 HYPIR-200 比。

需要和当前强 baseline 比较，包括：

- HYPIR-50
- texture-selective HYPIR
- current global fusion baseline

主要指标：

- PSNR
- SSIM
- LPIPS
- sharp-region PSNR
- mild-blur PSNR
- heavy-blur PSNR
- edge / gradient consistency

### 20.1 Synthetic Held-out 区域指标

根据真实 blur map：

#### Sharp

\[
B<0.2
\]

#### Mild

\[
0.2\le B<0.5
\]

#### Heavy

\[
B\ge0.5
\]

分别计算：

\[
PSNR_{\text{sharp}}
\]

\[
PSNR_{\text{mild}}
\]

\[
PSNR_{\text{heavy}}
\]

理想情况：

\[
\boxed{
PSNR_{\text{heavy}}\uparrow
\quad
\text{同时}
\quad
PSNR_{\text{sharp}}\approx\text{不变}
}
\]

这才真正说明：

> 模型学会了优化模糊位置，而不是对整张图强行锐化。

---

## 21. 官方 5 对验证集的定性硬门槛

### case1 / case2

重点检查：

- 中文文字
- 书脊文字
- 细边缘

不能出现明显字符重绘。

### case3

重点检查：

- 鸟轮廓
- 羽毛
- 水面背景

不能产生无依据高频。

### case4

重点检查：

- 密集绿植
- 虚假纹理
- 彩色异常
- 动物样结构
- 对象级 hallucination

### case5

重点检查：

- 钟表数字
- 指针
- 刻度

不能改变结构关系。

---

## 22. 最终研究逻辑

整个方案的研究链路可以概括为：

```text
Domain Alignment
      ↓
Degradation Understanding
      ↓
Spatial Conditioning
      ↓
Physical Consistency
```

即：

### 第一步

让训练数据像比赛。

### 第二步

让模型知道：

> 怎么坏的。

### 第三步

让模型知道：

> 哪里坏了。

### 第四步

限制模型：

> 只能生成能够解释原始输入的细节。

---

## 23. 最终主线总结

最终方案不再追求：

> 给 HYPIR 堆更多生成能力。

而是：

\[
\boxed{
\text{让 HYPIR 的强生成 prior 受到正确退化信息和观测一致性的约束}
}
\]

完整逻辑是：

\[
\boxed{
\text{What degradation}
+
\text{Where degraded}
+
\text{How to restore}
+
\text{Whether restoration explains the observation}
}
\]

对应具体模块：

```text
What degradation
→ Degradation Token

Where degraded
→ Spatial Blur Map

How to restore
→ Spatial Adapter / optional Δ-LoRA

Whether valid
→ Reblur Consistency
```

最终模型：

# DAS-HYPIR

> **Degradation-Aware Spatial HYPIR**

它保留 HYPIR 已有的自然图像生成先验，同时通过 CSIG-specific degradation、连续退化表征、空间 blur map 和物理一致性约束，让模型从“自由生成高清纹理”转向“有依据地恢复真实细节”。

---

## 24. 当前执行建议

第一阶段只做：

1. P0：Degradation Matching
2. E1：CSIG-specific degradation
3. E2：Degradation Token
4. E3：Spatial Blur Map + Adapter

先不要实现：

- Mixture-of-LoRA
- Global Attention
- GAN
- Semantic Detector
- 大规模 HYPIR LoRA 重训

只有当 E1-E3 出现稳定收益后，才进入：

- E4 Reblur
- E5 small Δ-LoRA
- E6 Global LQ Context

这样可以保证：

- 理论假设清晰
- 工程量可控
- 每一步都能做可靠消融
- 不会因为模块过多而无法判断收益来源
- 更适合后续形成论文和比赛答辩逻辑
