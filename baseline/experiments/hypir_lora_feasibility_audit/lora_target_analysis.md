# LoRA target analysis

目标：把 mapping 从「CSIG LQ → 错误植物 prior」改成「CSIG LQ → 更接近 CSIG GT」。不是训一个普通超分 LoRA。

## 官方已经选了什么

`configs/sd2_train.yaml:77` 与 `test.py` / `sd2_gradio.yaml` 一致：

```text
to_k, to_q, to_v, to_out.0,
conv, conv1, conv2, conv_shortcut, conv_out,
proj_in, proj_out,
ff.net.2, ff.net.0.proj
```

`HYPIR_sd2.pth` key 分组（514 tensors）：

| 组 | tensor 数 |
|---|---:|
| attn（to_q/k/v, to_out.0） | 256 |
| conv* | 130 |
| proj_in / proj_out | 64 |
| ff.net | 64 |

包含 down/mid/up 和 `conv_out`。**官方不是只训 cross-attention。**

## A. 只训 UNet cross-attention？

不合理，且违背「优先复用官方设计」。

Cross-attn 主要吃空 prompt。比赛推理 caption 是 `""`。只动 attn 很难改 case4 那种局部中频叶型。

## B. 只训 attention 够不够改中频结构？

不够作为默认策略。错误锯齿叶 / 鱼头是空间中频形态，残差块 `conv1/conv2/conv_shortcut` 和 `conv_out` 更直接。官方已经把这些放进 target modules。

## C. 是否应覆盖 attention + projection + conv？

若将来真的训，应 **原样使用官方列表**，不要自造模块名。不要只 attn，也不要再扩到全部 UNet。

## D. 官方实现已经覆盖哪些？

见上表。rank=256，`lora_alpha=rank`，高斯初始化（训练）或加载 `HYPIR_sd2.pth`（推理）。

## 和「mapping adaptation」的缺口

要 adapt **已经训好的** HYPIR LoRA，需要：

1. `add_adapter` 后 `load_state_dict(HYPIR_sd2.pth)`，而不是 gaussian init
2. 继续在这些 259M 参数上小 lr 更新

官方 `train.py` 不做第 1 步。即使补上，没有 CSIG 匹配配对数据，target modules 选对也没有监督信号去纠正鱼头。
