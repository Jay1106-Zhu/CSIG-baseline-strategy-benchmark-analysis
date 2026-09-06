# coeff_t 单变量 Sweep 实验报告

实验日期：2026-09-01  
实验集：`csig_dataset/验证集/` 的 `case1`--`case5`，每组均有 LQ/GT。  
唯一变量：`coeff_t ∈ {200, 150, 100, 75, 50}`；固定 `model_t=200`。

## 实验设计与代码路径

`HYPIR/test.py` 的 `--coeff_t` 在构造 `SD2Enhancer` 时直接传入（`test.py` 参数解析和构造调用）；`HYPIR/HYPIR/enhancer/sd2.py` 的 `forward_generator()` 使用 `self.scheduler.step(eps, self.coeff_t, z_in)`，因此它是 scheduler 还原 `pred_original_sample` 所用的 timestep。`model_t` 则在同一文件中作为 UNet 输入 timestep。两者作用不同，本实验只改变前者。

固定条件：当前本地 `sd-research/stable-diffusion-2-1-base` 镜像、`HYPIR/weights/HYPIR_sd2.pth`、LoRA rank 256 与既有模块列表、seed 231、空 prompt、`patch_size=512`、`stride=256`、`upscale=1`、`scale_by=factor`、CUDA。每组独立目录 `baseline/experiments/coeff_t_<value>/`，未写入 `baseline/output/`、`baseline/comparison/` 或 `baseline/evaluation_metrics.csv`，未修改 HYPIR 官方源码。

## 平均指标

LPIPS 越低越好，PSNR/SSIM 越高越好。LQ 平均基线为 PSNR 28.034420、SSIM 0.777699、LPIPS 0.356532。

| coeff_t | Average PSNR | 相对 LQ ΔPSNR | Average SSIM | 相对 LQ ΔSSIM | Average LPIPS | 相对 LQ ΔLPIPS | 推理耗时(s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 24.528587 | -3.505833 | 0.684926 | -0.092773 | 0.461005 | +0.104473 | 316.423 |
| 150 | 26.213323 | -1.821097 | 0.734088 | -0.043611 | 0.400391 | +0.043859 | 350.165 |
| 100 | 27.504504 | -0.529916 | 0.762503 | -0.015196 | 0.362717 | +0.006185 | 369.286 |
| 75 | 27.957945 | -0.076475 | 0.771081 | -0.006618 | 0.350124 | -0.006408 | 355.261 |
| 50 | **28.257533** | **+0.223113** | **0.776935** | **-0.000764** | **0.338965** | **-0.017567** | 342.774 |

逐 case 的完整数值（包括每个 coeff_t 的 PSNR/SSIM/LPIPS、相对 LQ delta、结论和输出路径）见 [`coeff_t_sweep_summary.csv`](coeff_t_sweep_summary.csv)。

## 结论 A--H

### A. coeff_t 降低是否真的改善结果？

在本次固定条件和五个候选值内，是。平均 PSNR 从 24.528587（200）连续升至 28.257533（50），平均 SSIM 从 0.684926 升至 0.776935，平均 LPIPS 从 0.461005 降至 0.338965。`coeff_t=50` 的平均 PSNR 已比 LQ 高 0.223113 dB，平均 LPIPS 比 LQ 低 0.017567；但平均 SSIM 仍低 0.000764，因此不是三项指标全部超过 LQ。

这只是当前候选范围、当前镜像 base、单 seed 和五张 evaluation 的事实，不能外推到更低值或其他数据。

### B. 最好的 coeff_t

候选集合中是 **`coeff_t=50`**：平均 PSNR、SSIM、LPIPS 均最佳。由于 50 是搜索下界，不能声称全局最优。

### C. 与 baseline `coeff_t=200` 的差异

`coeff_t=50` 相对 baseline 的平均变化为：PSNR **+3.728946 dB**、SSIM **+0.092009**、LPIPS **-0.122040**。逐 case 的输出都比 `coeff_t=200` 更接近 GT；但是否超过 LQ 取决于场景：case3 仍三项低于 LQ，case1 的 SSIM/LPIPS 仍未超过 LQ。

### D. 五个场景发生了什么？

- **case1 文字/文档：** `coeff_t=50` 相对 200 的 PSNR/SSIM/LPIPS 为 32.143319/0.945300/0.135358，对比 29.335323/0.846634/0.452658。三联图中输出的横向带状伪影和大范围生成纹理明显减弱，整体更接近 LQ/GT。小字在缩略图中不能逐字符核验，不能声称每个字符都正确。
- **case2 书脊文字：** `coeff_t=50` 为 28.976278/0.825432/0.331810，均优于 200 的 24.213958/0.731045/0.374753；PSNR 也比 LQ 高 0.786896。书脊和编号区域的高频重绘减少，布局保持；小号竖排字仍不适合仅凭缩略图判定逐字一致。
- **case3 鸟类：** `coeff_t=50` 为 34.688713/0.920412/0.198108，较 200 提升 5.823315 dB PSNR、0.108315 SSIM，LPIPS 降 0.207743，但仍比 LQ 低 0.933369 dB、低 0.014546 SSIM。输出水面高频纹理和羽毛/边缘重绘明显减弱，鸟的主体姿态保持；仍存在与 GT 纹理组织不同的生成变化。
- **case4 密集绿植：** `coeff_t=50` 为 18.084599/0.310380/0.785311，较 200 提升 1.732160 dB PSNR、0.052775 SSIM，LPIPS 降 0.159237，并略高于 LQ 的三项指标。输出更接近 LQ 的低频叶片结构；在缩略三联图中未见 baseline 中那种明显的彩色动物样对象，但不能据此证明局部不存在幻觉。纹理仍比 GT 模糊，不能把“更平滑”当成完整恢复。
- **case5 钟表：** `coeff_t=50` 为 27.394758/0.883153/0.244238，较 200 提升 3.518942 dB PSNR、0.105904 SSIM，LPIPS 降 0.201450，且三项均超过 LQ。表圈、罗马数字和指针整体几何位置在三联图中保持；降低后边缘重绘/过度锐化减轻。没有在缩略图中确认确定的数字位移或指针错位，因此不做更强语义断言。

### E. case1/case2 文字和 case5 钟表是否出现更少重绘？

从三联图的可见高频变化看，**是，重绘倾向明显减少**：`coeff_t=200` 的文字背景带状纹理、书脊细字增强和钟表边缘过度锐化，在 `coeff_t=50` 减弱，输出更接近输入低频结构。这个判断针对“可见纹理/边缘重绘”而不是 OCR 级字符正确率。由于展示图是缩略图，不能确认每个汉字、数字或指针端点是否逐一与 GT 一致。

### F. case3 鸟类和 case4 绿植纹理变化

case3 从密集水面波纹、羽毛边缘生成，转向较低频、较接近 LQ 的纹理；主体轮廓更稳定，但仍未完全达到 GT。case4 从 `coeff_t=200` 的高频叶片重绘和对象样结构，转向更平滑、保守的叶片/花簇外观；这解释了 PSNR/SSIM 的改善，也意味着不能仅按视觉锐度评价。

### G. 与之前 model_t sweep 是否一致？

不一致，趋势相反。在固定 `coeff_t=200` 时，之前的 `model_t` 从 200 降到 50 使平均 PSNR/SSIM 下降、LPIPS 上升；本实验固定 `model_t=200` 后，`coeff_t` 从 200 降到 50 使三项平均指标改善。说明 `model_t` 与 `coeff_t` 不是可以互换的“强度旋钮”，必须分开归因。

### H. 当前最值得继续研究的变量

优先做 **`coeff_t` 在 25--75 附近的更密集单变量 sweep**（例如 60/50/40/30/20），因为当前最佳点落在下界，尚未找到拐点；先保持 `model_t=200` 和全部其他条件不变。随后再用少量不同 seed 检查 case1/case2/case5 的字符、数字、指针稳定性，并保留当前镜像 base 与官方 `stabilityai` base 的来源差异记录。当前不建议直接修改 HYPIR 网络结构或开始 100 张测试集批量提交。

## 产物与复现

- 每组实验：`baseline/experiments/coeff_t_<value>/output/result/`（5 张输出）、`comparison/`（5 张三联图）、`evaluation_metrics.csv`、`experiment_metadata.md`、`inference.log`。
- 汇总：[`coeff_t_sweep_summary.csv`](coeff_t_sweep_summary.csv)。
- 编排/评测脚本：[`run_coeff_t_sweep.ps1`](run_coeff_t_sweep.ps1)、[`build_coeff_t_artifacts.py`](build_coeff_t_artifacts.py)、[`build_coeff_t_summary.py`](build_coeff_t_summary.py)。
- 每组输出均通过 RGB、finite、尺寸与输入一致性检查；`git -C HYPIR status` 未显示 tracked 源码变更。
