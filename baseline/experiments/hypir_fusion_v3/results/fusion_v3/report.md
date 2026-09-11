# NO-GO multi-band

Laplacian 三带融合。未改 HYPIR，未覆盖 fusion_v1/v2/texture_selective。LPIPS = Alex 1024。

## 五个问题

1. **错误结构主要在哪个频带？** 中频和高频都有，高频并不干净。鱼头眼睛/鳞片/外轮廓在 `H200_high`（L0，~1–2 px）；头型和叶片锯齿在 `H200_mid`。鱼头 crop：mean |H200−LQ| mid=8.46，high=7.27。黄花 crop high=8.60 > mid=7.09。
2. **有价值细节主要在哪个频带？** 无法从 H200 里单独拿出「干净高频」。H200 的高频就是错误结构的边缘。
3. **是否实现了分离？** 没有。B/C/D 的 case4 仍是鱼头，只是眼睛没那么蓝。锯齿叶缘跟着 high 进来。
4. **是否超过 28.480？** 没有。最好的 multi_B = **26.604**，低于锚 1.88 dB，也低于 LQ 28.03。
5. **是否用于 100 张测试集？** 否。

## 平均指标

| method | PSNR | SSIM | LPIPS_1024 |
|---|---:|---:|---:|
| LQ | 28.034 | 0.778 | 0.204 |
| H50 | 28.258 | 0.777 | 0.162 |
| H200 | 24.529 | 0.685 | 0.160 |
| **texture_selective** | **28.480** | 0.781 | 0.165 |
| fusion_v1_A | 28.460 | 0.782 | 0.189 |
| multi_B (LQ/LQ/H200) | 26.604 | 0.707 | 0.171 |
| multi_C (LQ/H50/H200) | 26.579 | 0.710 | 0.151 |
| multi_D01 α_mid=0.1 | 26.493 | 0.709 | 0.149 |
| multi_D02 α_mid=0.2 | 26.389 | 0.707 | 0.147 |

门槛 28.53。全部 multi-band 远低于。Case4 PSNR：LQ 18.06，texture 18.02，B 17.36，H200 16.35。结构变差。

## 视觉（case4 fish）

`crops/case4_fish.png`：B/C/D01/D02 仍是鱼头。  
`crops/case4_fish_bands.png`：H200_high 里眼睛和鳞片边缘非常清楚。dHigh 在鱼头区域发亮。

水面假波纹：dMid 很强，dHigh 也有颗粒。`band_energy` case3 water：dHigh 2.12 > dMid 1.20。

## 裁决

错误已经大量进入高频。把 High 换成 H200 就是把鱼头边缘和锯齿叶缘贴回去。理论基础不成立。

**冻结 texture_selective_h200，停止继续 output-space 微调，进入最终测试集推理/工程整理。**

不要再调频带数、α、mask、CLIP、DINO、ControlNet、Adapter。
