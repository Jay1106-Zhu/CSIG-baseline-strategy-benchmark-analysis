# CSIG 项目交接报告

报告日期：2026-09-01  
工作目录：`D:\MyProjects\CSIG`  
范围：仅基于当前文件、代码、CSV、输出图片、模型元数据和 Git 状态检查；本次未修改 HYPIR 源码、未下载模型、未运行新的推理实验。

## Executive status

当前项目已完成 HYPIR-SD2 的可复现 baseline 闭环：本地环境、LoRA、一个测试集 evaluation 侦察样本、五张带 GT 的 evaluation/验证集样本、对比图和指标 CSV 都已存在。baseline 使用官方 HYPIR `test.py`，但使用的是公开镜像 base，而不是 README 指定的 `stabilityai/stable-diffusion-2-1-base`。在固定 seed=231、`model_t=200`、`coeff_t=200` 的当前设置下，五张验证图的平均 PSNR/SSIM 下降，平均 LPIPS 变差；因此当前结果是跑通和风险侦察结果，不是可直接提交的参数结论。

## 证据索引（Evidence -> Finding -> Path）

| Evidence | Finding | Path |
|---|---|---|
| 赛题原文、验证集/测试集文件数量 | 任务要求、输入输出和 evaluation/test 区别 | `赛题.txt`, `csig_dataset/验证集/`, `csig_dataset/测试集/` |
| HYPIR Git HEAD 与工作树状态 | HYPIR commit 为 `b61d107c...`; 跟踪源码无 diff，模型目录/权重为未跟踪文件 | `HYPIR/.git/`, `git -C HYPIR status` |
| HYPIR README、`test.py`、enhancer 实现 | 当前推理参数、模型数据流和同分辨率适配 | `HYPIR/README.md`, `HYPIR/test.py`, `HYPIR/HYPIR/enhancer/base.py`, `HYPIR/HYPIR/enhancer/sd2.py` |
| 权重 SHA-256 与 base HF metadata | LoRA 文件和本地 base 的可追溯身份；base 来自 `sd-research` 镜像 | `HYPIR/weights/HYPIR_sd2.pth`, `HYPIR/models/stable-diffusion-2-1-base/.cache/huggingface/download/*.metadata` |
| 实际 CSV 行 | 五个 case 与 Average 的原始指标，不重新计算替换 | `baseline/evaluation_metrics.csv` |
| 五组三联图 | 每个场景的文字、书脊、鸟、绿植、钟表视觉问题 | `baseline/comparison/case1_lq_compare.png` 至 `case5_lq_compare.png` |
| 进度/调研记录 | 已完成工作、耗时、未做实验和历史决策 | `progress.md`, `findings.md`, `task_plan.md` |

## 1. 项目与比赛背景

### 比赛任务

赛道为“生成式图像增强可控性挑战”。需要用 Diffusion/AIGC 模型增强 4K 低质量图像，在提升清晰度的同时保持内容、结构、色彩和真实性，重点场景包括小人脸、文字、密集绿植、钟表和鸟类。综合评分同时包含有参考和无参考指标，具体权重未在当前赛题文件中公布。

### 输入与输出

- 初赛测试集：100 张低质量图像，文件名为 `case1.jpg` 至 `case100.jpg`，Ground Truth 不公开。
- 输出必须与输入严格同名、JPG 格式，放在 `output_dir` 中，再按要求打包为 zip；赛题示例要求 zip 内有 `output_dir/case1.jpg` 等文件。
- 处理前后分辨率必须不变。当前本地测试集图像为 RGB，分辨率为 `4096x3072` 或 `3072x4096`。

### Evaluation 与 test 的区别

- `evaluation`（当前目录也称验证集）：5 组 LQ/GT 对，共 10 张图，仅供本地验证，不提交结果。
- `test`：100 张只有 LQ 的测试图，GT 不公开，才是初赛实际提交对象。
- 当前已完成的是五张带 GT 的 evaluation/验证集 baseline，以及测试集 `case1` 的单张 evaluation 侦察；尚未批量处理 100 张 test，也没有生成最终提交 zip。

### 当前目标

先理解官方 HYPIR baseline 的内容保持能力，建立可复现指标和视觉证据；在不修改网络的前提下，用单变量实验寻找较低生成强度或场景适配策略，再决定是否需要区域控制或模型改动。

## 2. 当前环境

| 项目 | 当前实际值 |
|---|---|
| OS | Microsoft Windows 11 家庭版中文版；版本 `10.0.26200`，64 位（`Get-CimInstance`）；`Get-ComputerInfo` 同时报告 Windows 10 Home China/Build 26200，属于系统命名显示差异 |
| Python | `3.11.16`，Conda Python，解释器 `\.conda\python.exe` |
| PyTorch | `2.11.0+cu128` |
| torchvision | `0.26.0+cu128` |
| CUDA runtime | `12.8`；`torch.cuda.is_available()` 为 `True` |
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU，约 `16303 MiB`，计算能力 `12.0` |
| NVIDIA driver | `610.78` |
| HYPIR commit | `b61d107c6cef38f01a93c7833558869731cfa8c1`，`main`，提交信息为合并 tiled inference 更新（2025-10-16） |
| 关键依赖 | `diffusers 0.32.2`、`transformers 4.49.0`、`accelerate 1.4.0`、`peft 0.14.0`、`lpips 0.1.4`、`open_clip_torch 2.31.0`、`timm 1.0.15`、`einops 0.8.1`、`numpy 2.4.6`、`Pillow 11.3.0`、`scikit-image 0.26.0`、`safetensors 0.8.0`、`torchmetrics 1.9.0`、`basicsr 1.4.2`、`gfpgan 1.3.8`、`realesrgan 0.3.0`、`facexlib 0.3.0`、`tensorboard 2.19.0`、`tb-nightly 2.21.0a20251023` |

注意：仓库根目录 `D:\MyProjects\CSIG` 没有顶层 `.git`，因此根目录 `git status` 不可用。`HYPIR` 自身是 Git 工作树，跟踪源码没有修改，但 `models/` 和 `weights/` 在其状态中显示为未跟踪文件。

## 3. 当前模型与权重

### HYPIR

- 版本/commit：`b61d107c6cef38f01a93c7833558869731cfa8c1`。
- LoRA 权重：`HYPIR/weights/HYPIR_sd2.pth`。
- 文件大小：`1,038,090,392` bytes。
- SHA-256：`D538A2CB925451FAB1F75ADFE715AC2B1C8BB12FA32A0851BA01BE2347B354D6`。

### Stable Diffusion base

- 实际来源/repo id：`sd-research/stable-diffusion-2-1-base`。
- 本地路径：`HYPIR/models/stable-diffusion-2-1-base`。
- 本地 Hugging Face metadata commit：`0708cecd370b4d1c3a6ff3f7332f5e9aea78896f`。
- 目录含 `model_index.json`、scheduler、tokenizer、text_encoder、UNet 和 VAE；本地 safetensors 与 metadata 存在且推理已成功加载。
- HYPIR README 指定的官方 base 是 `stabilityai/stable-diffusion-2-1-base`。
- **实际使用的 base 不是 README 指定的官方 repo。** 当时官方 repo API 返回 401/404，项目改用结构匹配的公开 `sd-research` 镜像；运行时 LoRA key 断言通过，说明结构和权重键兼容，但没有证据证明两个 repo 的权重逐字节相同。
- 来源差异风险：base 权重、配置或版本差异可能改变生成结果，且可能是指标下降或内容漂移的贡献因素；官方 `stabilityai` base 尚未在本项目中测试。

## 4. 当前推理配置

以下是五张验证集 baseline 和测试集 `case1` 侦察实际使用的配置；验证集输出由 `HYPIR/test.py` 生成，prompt 文件均为空。

| 参数 | 实际值 |
|---|---|
| `base_model_type` | `sd2` |
| `base_model_path` | `HYPIR/models/stable-diffusion-2-1-base`（实际为 `sd-research` 镜像） |
| `model_t` | `200` |
| `coeff_t` | `200` |
| `lora_rank` | `256` |
| `lora_modules` | `to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj` |
| `patch_size` | `512` |
| `stride` | `256` |
| `scale_by` | `factor` |
| `upscale` | `1`（适配赛题同分辨率要求；README 示例的 `4` 未采用） |
| `captioner` | `empty`；没有 `txt_dir`，生成的 prompt `.txt` 文件为 0 字节 |
| `seed` | `231` |
| `device` | `cuda` |
| 验证输入目录 | `baseline/input` |
| 验证输出目录 | `baseline/output/result` |
| 测试侦察输入目录 | `baseline/evaluation_input` |
| 测试侦察输出目录 | `baseline/evaluation_output/result` |

HYPIR 的 tiled VAE 编码、Generator forward 和 VAE 解码均使用上述 patch/stride；`upscale=1` 后输出裁剪回输入分辨率。

## 5. 当前已经完成的工作

- 环境配置：建立项目 `.conda` Python 3.11 环境，安装 CUDA 12.8 PyTorch，并补齐 HYPIR 运行依赖；历史记录中的 `pip check` 为通过。
- 模型下载：已下载 HYPIR LoRA 和 `sd-research` base 的必要 Diffusers 文件；未下载官方 `stabilityai` base。
- baseline 推理：用未修改的官方 `HYPIR/test.py` 完成测试集 `case1` 单张 evaluation 侦察和验证集 `case1_lq` 至 `case5_lq` 五张推理；五张验证输出均成功、无 OOM、RGB/uint8、尺寸保持。
- 5 张 evaluation/验证推理：结果在 `baseline/output/result/case1_lq.png` 至 `case5_lq.png`；总耗时历史记录约 348.9 秒，模型加载约 11.59 秒。
- 性能记录：测试集 `case1` 单图总耗时约 58.13 秒（模型初始化约 10.75 秒）；验证集 `case1_lq` 约 63.15 秒；两次记录的 CUDA 峰值均约 3.994 GiB allocated、4.936 GiB reserved。
- combine 可视化：`baseline/compare_images.py` 生成测试集二联图和五张验证集三联图；脚本不导入 HYPIR，不覆盖原始输出。
- metrics 计算：`baseline/evaluate_metrics.py` 生成 `baseline/evaluation_metrics.csv`，包含 5 个 case 和 `Average`；历史记录显示用 CUDA 计算 LPIPS-Alex，约 127.4 秒。
- 本次已完成独立的 `model_t` 单变量 sweep：`200/150/100/75/50`，每组五张 evaluation 图；产物位于 `baseline/experiments/`，汇总见 `baseline/experiments/model_t_sweep_summary.csv` 和 `model_t_sweep_report.md`。当前环境没有 `pytest` 模块，因此历史测试通过记录未在本次检查中重新执行。

## 6. 当前文件结构

仅列比赛、模型、推理、可视化和指标相关的重要路径；省略 Hugging Face `.cache`、Python `__pycache__` 等缓存。

```text
CSIG/
├─ 赛题.txt
├─ requirements-cu128.txt
├─ task_plan.md
├─ findings.md
├─ progress.md
├─ PROJECT_HANDOFF.md
├─ csig_dataset/
│  ├─ 测试集/                 # 100 张 case1.jpg ... case100.jpg
│  └─ 验证集/                 # case1_lq/gt ... case5_lq/gt，共 5 对
├─ baseline/
│  ├─ input/                  # case1_lq.jpg ... case5_lq.jpg
│  ├─ output/
│  │  ├─ result/              # 五张 HYPIR PNG 输出
│  │  └─ prompt/              # 五个空 prompt 文件
│  ├─ evaluation_input/       # 测试集 case1.jpg 侦察输入
│  ├─ evaluation_output/
│  │  ├─ result/case1.png     # 测试集 case1 输出
│  │  └─ prompt/case1.txt     # 空 prompt
│  ├─ comparison/             # case1 二联图及 case1--case5 三联图
│  ├─ compare_images.py
│  ├─ evaluate_metrics.py
│  └─ evaluation_metrics.csv
├─ HYPIR/
│  ├─ HYPIR/                  # 官方源码包，当前 tracked source 无 diff
│  ├─ test.py / README.md / requirements.txt
│  ├─ configs/
│  ├─ weights/HYPIR_sd2.pth
│  └─ models/stable-diffusion-2-1-base/
└─ tests/
   ├─ test_compare_images.py
   └─ test_evaluate_metrics.py
```

## 7. Baseline 实验结果

以下数值直接抄录当前 `baseline/evaluation_metrics.csv`，没有重新计算或替换。场景类别依据验证集图片和已有调研记录。

| Case | 场景类别 | LQ PSNR | Output PSNR | ΔPSNR | LQ SSIM | Output SSIM | ΔSSIM | LQ LPIPS | Output LPIPS | ΔLPIPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case1 | 中文文字/文档 | 32.028801 | 29.335323 | -2.693478 | 0.948453 | 0.846634 | -0.101820 | 0.072517 | 0.452658 | 0.380142 |
| case2 | 书脊文字/密集小字 | 28.189381 | 24.213958 | -3.975424 | 0.831443 | 0.731045 | -0.100398 | 0.373972 | 0.374753 | 0.000781 |
| case3 | 鸟类 | 35.622082 | 28.865398 | -6.756683 | 0.934958 | 0.812097 | -0.122862 | 0.170643 | 0.405851 | 0.235208 |
| case4 | 密集绿植 | 18.056199 | 16.352439 | -1.703761 | 0.309323 | 0.257605 | -0.051718 | 0.879949 | 0.626074 | -0.253875 |
| case5 | 钟表 | 26.275637 | 23.875816 | -2.399821 | 0.864316 | 0.777249 | -0.087067 | 0.285579 | 0.445688 | 0.160109 |
| **Average** | 5 组平均 | **28.034420** | **24.528587** | **-3.505833** | **0.777699** | **0.684926** | **-0.092773** | **0.356532** | **0.461005** | **0.104473** |

CSV 中 PSNR/SSIM 五个 case 均为 `Degraded`；LPIPS 只有 case4 为 `Improved`，其余 case 和 Average 为 `Degraded`。LPIPS 是越低越好，case4 的单项改善不能抵消 PSNR/SSIM 下降。

## 8. Baseline 视觉观察

观察依据：`baseline/comparison/case1_lq_compare.png` 至 `case5_lq_compare.png` 的 Input/LQ、HYPIR Output、Ground Truth 三联图；测试集 `case1_compare.png` 另作无 GT 的文字招牌侦察。

- **case1：中文文字/文档。** 输出整体构图和粉色背景仍在，但文字笔画被明显重新生成并变软，局部出现不均匀的横向纹理/带状感；字符完整性和原文一致性不可靠。该组 PSNR、SSIM 明显下降，LPIPS 增幅最大之一，说明“更处理过”没有带来 GT 接近度。测试集 `case1_compare.png` 的餐饮招牌和价格文字大体保留，但细小中文、数字和边缘也有字符重绘风险；该图没有 GT，不能据此确认具体字符错误。
- **case2：书脊文字。** 书本位置和大标题总体保持，输出局部对比度/锐度更高；然而书脊小字、竖排笔画、细线和编号区域有重绘/笔画变化风险。缩略图下中央 `1/2/3/4` 的布局仍可辨，但不能证明数字逐字一致；输出存在过度锐化和文字 hallucination 风险。PSNR/SSIM 下降约 4 dB/0.10，LPIPS 几乎不变但 CSV 结论仍为 Degraded。
- **case3：鸟类。** 鸟的主体轮廓、姿态和场景位置大体保留，但输出边缘轮廓不自然；水面/滩涂被改成更密集、更锐且与 GT 组织不同的纹理，不能简单视为恢复模糊。鸟眼、喙、腿和羽毛细节有重绘风险。该组 PSNR 降幅最大（-6.756683 dB），视觉锐度不能代表内容保真。
- **case4：密集绿植。** 输出把叶片、枝条和花簇变得更清晰、更高频，局部叶形、亮点、颜色和纹理明显重新生成。更严重的是 baseline 中出现了 LQ/GT 均不存在的对象级 hallucination：右中部鹦鹉/蜥蜴样彩色物体、右侧疑似动物头部、左中部脸样结构。这不是普通锐化，而是把叶片纹理错误解释并生成了新对象。LPIPS 单项改善（-0.253875）与 PSNR/SSIM 下降并存，是感知指标和像素/结构指标冲突的直接例子。
- **case5：钟表。** 输出的表圈、罗马数字和指针边缘更锐，但这些几何细节经过生成式重绘。当前缩略图未确认明显的大幅指针位置或数字整体位移，不能宣称发生了确定的数字改变；不过细小刻度、数字笔画和指针端点存在变化风险，需原图级结构检查。PSNR/SSIM 下降，LPIPS 变差。

## 9. 当前已确认的结论

### 已通过现有实验确认的事实

1. HYPIR 官方 `test.py` 在当前 Windows/RTX 5080 环境可以加载 LoRA 和本地 base，并完成同分辨率 4K tiled 推理；五张验证输出均成功保存，无 OOM，输出健康检查为 RGB、uint8、finite、0--255。
2. `upscale=1` 下五张验证输出尺寸与输入完全相同；测试 `case1` 也为 `4096x3072` 输入到 `4096x3072` PNG 输出。
3. 在当前 base、seed、空 prompt、`model_t=200`、`coeff_t=200`、patch/stride 设置下，五个验证 case 的 PSNR 和 SSIM 全部低于 LQ；Average 为 PSNR `-3.505833` dB、SSIM `-0.092773`、LPIPS `+0.104473`。
4. 当前 base 是 `sd-research/stable-diffusion-2-1-base` 镜像，不是 HYPIR README 指定的 `stabilityai/stable-diffusion-2-1-base`；LoRA key 兼容性已在实际加载时通过，但来源不等价未被证明。
5. 已生成可追溯的五组三联图、测试 case1 二联图和 CSV；原始输入、GT、HYPIR 输出没有被 combine 或指标脚本静默 resize/覆盖。

### 仍只是推测的原因

- `model_t=200` 或 `coeff_t=200` 的生成强度可能过高，导致内容改写；当前没有 sweep，不能写成已证明事实。
- 镜像 base 与官方 base 的权重/配置差异可能影响结果；尚未有官方 base 对照。
- 空 prompt、tiled patch 的局部条件和分块融合可能影响文字/钟表保真；当前没有分别控制这些变量的实验。
- 不同场景可能需要不同强度或区域策略；五张 evaluation 样本只能提示方向，不能证明泛化规律。

## 10. 当前主要假设（待验证）

1. **待验证假设：** 降低 `model_t` 和/或 `coeff_t` 会减少 diffusion generation strength，提高文字、钟表和鸟类的内容保真。
2. **待验证假设：** `model_t` 与 `coeff_t` 的作用不完全相同，应该分别做单变量 sweep，而不是同时改动后归因。
3. **待验证假设：** 不同场景需要不同强度：文字/钟表偏低强度，绿植/羽毛可允许较高强度。
4. **待验证假设：** 对文字、书脊和钟表增加区域内容约束或低强度原图融合，能减少字符、数字和指针重绘。
5. **待验证假设：** 使用 README 指定的官方 `stabilityai` base 后，结果可能与当前镜像不同；差异大小和方向未知。

## 11. 当前还没有做的实验

- `model_t` sweep 已完成（`200/150/100/75/50`，固定 `coeff_t=200`）；结果显示本次候选中 `model_t=200` 的平均 PSNR/SSIM/LPIPS 均最好。详见 `baseline/experiments/model_t_sweep_report.md`。
- 尚未做 `coeff_t` sweep，也没有做两者的严格单变量对照。
- 尚未测试不同 seed 的稳定性。
- 尚未做区域控制、OCR/文字掩膜、钟表/鸟类专用约束或原图融合候选。
- 尚未训练 LoRA；现有 LoRA 是下载的 HYPIR 预训练权重。
- 尚未修改 HYPIR 网络结构或官方源码；`HYPIR` tracked source 当前无 diff。
- 尚未测试 HYPIR README 指定的官方 `stabilityai/stable-diffusion-2-1-base`。
- 尚未批量推理 100 张 test、生成最终 JPG 输出目录或提交 zip。
- 尚未验证比赛完整综合评分权重，也没有无参考指标的本地复现结果。

## 12. 下一步建议

按“先理解 baseline，再做单变量实验，再决定模型改动”的原则，近期只建议：

1. 保留本次 `model_t` sweep 结果，下一步在固定 `model_t=200` 及其余条件的前提下，单独对 `coeff_t` 做小范围 sweep；每个组合继续写入独立 experiment 目录。
2. 在明确的候选参数上测试少量不同 seed，检查文字/书脊/钟表是否出现不稳定的字符、数字或指针变化；同时记录 PSNR、SSIM、LPIPS 和三联图。
3. 只有当单变量结果仍不能保住文字/钟表，再做区域低强度/原图融合与纹理区较高强度的对照；完成前不改 HYPIR 网络。

## 13. 给下一位 AI 的注意事项

- 不要随意修改 HYPIR 官方源码；先通过独立脚本或参数入口验证假设。
- 不要改变 baseline 参数后覆盖 `baseline/output` 或 `baseline/evaluation_metrics.csv`；新实验必须建立独立 `experiment_*` 目录。
- 每次新实验必须保留 seed、base repo/commit、LoRA SHA-256、全部推理参数、输出路径和指标。
- 不要把视觉上更锐、更高频误认为更接近 GT；当前 case4 已显示 LPIPS 改善仍伴随 PSNR/SSIM 下降。
- evaluation 只有 5 张，不能针对单张图片过拟合后直接推广到 100 张 test。
- 文字、书脊、数字、钟表指针优先按内容身份和几何一致性检查；缩略图看不清时不要下确定结论。
- 保留当前镜像 base 的来源差异记录；若获得官方 base，必须在独立目录做对照，不替换现有 baseline。
- 根目录没有顶层 Git 工作树；需要查看版本时使用 `git -C HYPIR ...`，并注意 `models/`、`weights/` 是未跟踪大文件。
- `requirements-cu128.txt` 与实际环境存在版本差异：文件记录 `diffusers 0.40.0`、`transformers 5.16.1`、`accelerate 1.14.0`、`einops 0.8.2`、`open_clip_torch 3.3.0`、`timm 1.0.29`、Pillow `12.3.0` 等，而实际运行环境为 HYPIR pin/已验证组合的 `diffusers 0.32.2`、`transformers 4.49.0`、`accelerate 1.4.0`、`einops 0.8.1`、`open_clip_torch 2.31.0`、`timm 1.0.15`、Pillow `11.3.0`；重建环境前先决定是否要保持当前可运行组合。

## EXECUTIVE SUMMARY

1. 我们在 CSIG“生成式图像增强可控性挑战”上，目标是 4K 同分辨率增强并保持文字、鸟、绿植、钟表等内容真实。
2. 本地已有 100 张 test 和 5 对 evaluation/验证集 LQ/GT。
3. HYPIR 仓库已固定在 commit `b61d107c6cef38f01a93c7833558869731cfa8c1`。
4. HYPIR LoRA 已下载，SHA-256 为 `D538A2CB...7B354D6`。
5. 当前环境是 Windows 11、Python 3.11、PyTorch 2.11.0+cu128、RTX 5080 Laptop 16 GB。
6. 官方 HYPIR `test.py` 已跑通，采用 tiled 512/256、`upscale=1`、seed 231、空 prompt。
7. 实际 base 是 `sd-research/stable-diffusion-2-1-base` 本地镜像。
8. 该 base 不是 HYPIR README 指定的 `stabilityai/stable-diffusion-2-1-base`。
9. LoRA key 兼容性已验证，但两个 base 的权重等价性没有验证。
10. 测试集目前只跑了 `case1` 侦察；没有 100 张提交结果。
11. 五张 evaluation/验证图已经全部推理并生成三联对比图。
12. CSV 显示五组 PSNR/SSIM 全部下降，平均 PSNR 下降 3.505833 dB。
13. 平均 LPIPS 也变差；只有密集绿植 case4 的 LPIPS 单项改善。
14. 最大问题是生成式细节重绘：中文/书脊字符、鸟羽和水面纹理、绿植叶片、钟表细节都存在内容漂移风险。
15. `model_t` sweep 已完成；在当前固定 `coeff_t=200` 条件下，降低 `model_t` 未改善五组平均指标。
16. 仍没有测试不同 seed、区域控制、LoRA 训练、网络修改或官方 stabilityai base。
17. 下一步应在保留本次结果的前提下，单独研究 `coeff_t` 或其他明确的单变量；不要把本次结论外推到未测试的参数组合。
18. 只有单变量结果仍不足以保真时，再考虑文字/钟表区域约束或原图融合。
