# HYPIR process control v1

检查 **现有 inference 代码** 里有没有真正的生成过程控制入口，并用已经跑完的 `coeff_t` 五档做裁决。不修改 HYPIR，不覆盖 fusion，不重跑扩散。

## 代码事实（不以论文为准）

`HYPIR/HYPIR/enhancer/sd2.py` `forward_generator`：

```text
z_in = Encode(LQ) * scaling_factor
eps  = UNet(z_in, t=model_t, text)
x0   = DDPMScheduler.step(eps, coeff_t, z_in).pred_original_sample
RGB  = wavelet(Decode(x0) high-freq, LQ low-freq)
```

一次 UNet 前向。没有采样步数、没有 CFG、没有对 LQ latent 加噪。

唯一真正改变「离 LQ 走多远」的已有参数是 **`coeff_t`**：它进入 x0 公式的 `sqrt(1-α_t)`。`model_t` 只改 UNet timestep 嵌入，已证明降它会变差，不当生成强度。

## 本轮实验

复用已有 PNG，不推理：

| 档位 | 含义 | 已有目录 |
|---|---|---|
| 50 | 弱 / HYPIR-50 | `baseline/experiments/coeff_t_50/` |
| 75 | 中弱 | `coeff_t_75/` |
| 100 | 中 | `coeff_t_100/` |
| 150 | 中强 | `coeff_t_150/` |
| 200 | 强 / HYPIR-200 | `coeff_t_200/` |

对照：LQ、`texture_selective_h200`（锚 28.48）、`fusion_v1` 方案 A。

```powershell
& .\.conda\python.exe -m unittest tests.test_hypir_process_control_v1
& .\.conda\python.exe baseline\experiments\hypir_process_control_v1\experiment.py --root .
```

## 明确不做

- 修改 HYPIR 源码 / 网络
- 重新推理、训练、LoRA、Adapter、ControlNet
- 覆盖 fusion_v1 / fusion_v2 / texture_selective
- 再堆 Sobel / residual confidence / Gaussian mask
- 100 张测试集

## 本次结论（2026-09-11）

唯一真实强度旋钮是 `coeff_t`（单步 x0 尺度）。五档均 PSNR：50=28.26，75=27.96，100=27.50，150=26.21，200=24.53。全部低于锚 28.48。

Case4 鱼头随 t 增大被画实，不是被纠正。t=50 只是没画完的鱼头。

**暂停过程控制进入 LoRA / Adapter。** 详见 `results/process_control_v1/report.md`。
