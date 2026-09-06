# model_t 单变量 sweep 实验报告

实验日期：2026-09-01  
实验范围：evaluation/验证集 `case1` 至 `case5`，共 5 张 LQ/GT 对。  
唯一变量：`model_t ∈ {200, 150, 100, 75, 50}`。

## 实验设计

固定条件如下，除 `model_t` 外没有增加或改变参数：

| 参数 | 固定值 |
|---|---|
| base model | 当前本地 `HYPIR/models/stable-diffusion-2-1-base`，来源 `sd-research/stable-diffusion-2-1-base` |
| `coeff_t` | `200` |
| `seed` | `231` |
| `lora_rank` | `256` |
| `lora_modules` | `to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj` |
| `patch_size / stride` | `512 / 256` |
| `scale_by / upscale` | `factor / 1` |
| `captioner` | `empty`，空 prompt |
| `device` | `cuda` |
| 输入 | `baseline/input` |
| 推理入口 | 未修改的官方 `HYPIR/test.py` |

每个组合独立保存于 `baseline/experiments/model_t_<value>/`，包含 `output/result/` 五张 PNG、`comparison/` 五张三联图、`evaluation_metrics.csv` 和 `experiment_metadata.md`。原始 `baseline/output/`、`baseline/comparison/`、`baseline/evaluation_metrics.csv` 未覆盖。

## 平均指标比较

数值直接来自各实验目录中的 `evaluation_metrics.csv`，LPIPS 为越低越好。

| `model_t` | Average Output PSNR | Average ΔPSNR | Average Output SSIM | Average ΔSSIM | Average Output LPIPS | Average ΔLPIPS | 总耗时(s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 24.528587 | -3.505833 | 0.684926 | -0.092773 | 0.461005 | +0.104473 | 320.999 |
| 150 | 24.079372 | -3.955048 | 0.673582 | -0.104117 | 0.475143 | +0.118611 | 330.210 |
| 100 | 23.255397 | -4.779023 | 0.643493 | -0.134206 | 0.494161 | +0.137629 | 326.382 |
| 75 | 22.772125 | -5.262295 | 0.626244 | -0.151454 | 0.498991 | +0.142459 | 331.231 |
| 50 | 22.252461 | -5.781959 | 0.606825 | -0.170874 | 0.507663 | +0.151131 | 319.874 |

### 结论

- 按五组平均 PSNR，`model_t=200` 最好（24.528587 dB）。
- 按五组平均 SSIM，`model_t=200` 最好（0.684926）。
- 按五组平均 LPIPS，`model_t=200` 也最好（0.461005，最低）。
- 因此在本次固定条件和五张 evaluation 上，综合平均指标最佳的是 `model_t=200`；降低 `model_t` 没有改善平均保真度。
- 五个 `model_t` 的平均 PSNR、SSIM 均低于原始 LQ；平均 LPIPS 均高于原始 LQ。case4 的 LPIPS 改善和其他 case 的变化不能抵消整体下降。

完整的逐 case、Average 和输出路径对照见 `baseline/experiments/model_t_sweep_summary.csv`。

## 逐场景视觉变化

观察依据为每个实验目录的 `comparison/case*_lq_compare.png`，面板顺序为 Input/LQ、HYPIR Output、Ground Truth。

- **case1 文字/文档：** `model_t=200` 已出现文字软化、字符重绘和横向带状纹理。降到 150、100、75、50 后，输出没有变得更忠实；文字笔画对比度和局部纹理继续变化，背景上更容易看到生成纹理/噪声，字符身份仍不可靠。指标随降低明显恶化（PSNR 29.335323 → 24.869841，SSIM 0.846634 → 0.684048）。没有出现“低 `model_t` 文字更稳定”的视觉证据。
- **case2 书脊文字：** 所有设置都保持书本构图和大标题位置。较低 `model_t` 使书脊、细字、编号和边缘看起来更锐或更高对比，但细小笔画发生重绘的风险没有消失；`model_t=50` 的锐度不等于与 GT 更接近。LPIPS 随降低略有改善（0.374753 → 0.338133），但 PSNR/SSIM 继续下降，不能据此认为文字内容更正确。
- **case3 鸟类：** 鸟的主体位置和姿态总体保持，但水面波纹、羽毛边缘和鸟体局部纹理被生成式增强。降低 `model_t` 后水面高频纹理和羽毛重绘更明显，鸟眼、喙、腿等身份细节存在更高漂移风险；同时可见鸟的边缘轮廓不自然，和 LQ/GT 的边界过渡不一致。背景滩涂/水面也被改成更密集、更锐的纹理；这不是简单的“恢复原有模糊”，而是输出与 GT 纹理组织不同的生成式变化。PSNR/SSIM/LPIPS 均随降低恶化，未观察到更稳定的鸟体结构。
- **case4 密集绿植：** 所有设置都比 LQ 更清晰、更高频，但叶片形状、枝条细节、花簇亮点和颜色被重新生成。更严重的是，`model_t=200` baseline 输出中出现了 LQ/GT 均不存在的对象级 hallucination：右中部有一个鹦鹉/蜥蜴样的彩色动物结构，右侧还有疑似动物头部，左中部可见脸样结构。这些不是原有叶片的正常锐化，属于把叶片纹理错误解释成动物/面部并生成新对象。降低 `model_t` 并未证明能消除该问题；输出仍与 GT 的真实叶片结构有差异。LPIPS 在低 `model_t` 下反而更低（`model_t=50` 为 0.578716），而 PSNR/SSIM 仍低于 LQ，说明感知“像纹理”与逐像素/结构保真冲突。
- **case5 钟表：** 表圈和罗马数字轮廓在各组都被锐化/重绘。低至 `model_t=50` 时可见更强的边缘、竖向纹理和局部 ringing 风险；数字笔画、刻度和指针端点仍需按原图级检查，不能因为变锐就断言正确。没有出现低 `model_t` 使钟表数字或指针明显更稳定的现象；PSNR/SSIM 继续下降，LPIPS 变差。

## 是否出现低生成强度后更稳定

**没有。** 在本实现和固定 `coeff_t=200` 的条件下，`model_t` 从 200 降到 50 时，五组平均 PSNR 从 24.528587 降到 22.252461，平均 SSIM 从 0.684926 降到 0.606825，平均 LPIPS 从 0.461005 升到 0.507663。case1 文字、case5 钟表以及 case3 鸟的视觉检查都没有显示低值更稳定；低值组反而更容易出现高频纹理、边缘重绘或结构偏离。case4 的对象级 hallucination 在 baseline 中已经明确存在，不能把它归类为普通锐化。

需要谨慎解释“生成强度”：本实验只改变了 HYPIR 的 `model_t`，而 `coeff_t` 保持 200。不能把观察结果外推成所有 `model_t/coeff_t` 组合的普遍规律。

## 当前实验能证明什么

在同一镜像 base、同一 seed、同一空 prompt、同一 patch/stride 和同一 `coeff_t=200` 下，对当前五张 evaluation 图，所测试的五个 `model_t` 中 `200` 的平均 PSNR、SSIM、LPIPS 均最佳；降低 `model_t` 没有带来平均指标改善，也没有在文字/钟表上观察到更稳定的内容保持。所有组合都成功完成 4K 同分辨率推理，输出均通过 RGB、尺寸、uint8 和 finite 健康检查。

## 当前实验不能证明什么

- 不能证明 `model_t=200` 是全局最优值；只测试了五个离散点，且没有测试高于 200 或更密集的区间。
- 不能单独归因 `model_t` 的理论“生成强度”，因为 `coeff_t` 固定为 200，二者的耦合关系没有被拆开验证。
- 不能证明官方 `stabilityai/stable-diffusion-2-1-base` 会得到相同结果；本实验仍使用 `sd-research` 镜像。
- 不能证明五张 evaluation 上的趋势能泛化到 100 张 test；也不能据此选择最终提交参数。
- 不能证明视觉上更锐的输出更接近 GT；case4 的 LPIPS 与 PSNR/SSIM 已显示相反方向。
- 不能证明字符、数字或钟表指针具体发生了哪些语义改变；当前报告只记录图像级可见风险，没有 OCR/几何标注实验。

## 产物清单

| 产物 | 路径 |
|---|---|
| 汇总 CSV | `baseline/experiments/model_t_sweep_summary.csv` |
| 本报告 | `baseline/experiments/model_t_sweep_report.md` |
| 汇总脚本 | `baseline/experiments/build_model_t_summary.py` |
| 每组参数/耗时 | `baseline/experiments/model_t_{200,150,100,75,50}/experiment_metadata.md` |
| 每组指标 | `baseline/experiments/model_t_{200,150,100,75,50}/evaluation_metrics.csv` |
| 每组输出 | `baseline/experiments/model_t_{200,150,100,75,50}/output/result/` |
| 每组对比图 | `baseline/experiments/model_t_{200,150,100,75,50}/comparison/` |
