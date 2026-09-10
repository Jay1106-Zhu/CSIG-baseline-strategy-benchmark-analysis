# 调研发现

## 赛题约束
- 初赛输入：测试集 100 张，Ground Truth 不公开；验证集 5 组 LQ/GT 仅用于本地验证。
- 测试图命名为 `case1.jpg` 到 `case100.jpg`；输出必须严格同名、JPG、放在 `output_dir` 后压缩为 zip。
- 输入输出分辨率不变；算法必须采用 Diffusion 架构；综合评分含有参考和无参考指标，具体权重未公开。
- 测试和验证图均为 RGB，分辨率只有 `(4096,3072)` 与 `(3072,4096)` 两种。

## 数据概况
- 当前目录已解压到 `csig_dataset\`：测试集 100 张、验证集 10 张（5 对），另有 `.DS_Store`。
- 验证样本可见退化类型：case1 为偏色/模糊中文文本，case2 为书脊文字与纹理，case3 为鸟类，case4 为密集绿植，case5 为钟表。
- 视觉上 LQ 与 GT 保持同一构图，GT 主要提升锐度、局部纹理、对比度和色彩；文字/钟表等几何结构不能依赖自由生成。
- 代理统计（LQ 对 GT）：case1 PSNR 32.03 dB、case2 28.19 dB、case3 35.62 dB、case4 18.06 dB、case5 26.28 dB；绿植 case4 退化最重，鸟类 case3 相对较轻。

## 建模判断（2026-09-09/10 更新）
- **现行计划：** [`CURRENT_PLAN.md`](CURRENT_PLAN.md)。旧 DAS-V1（blur↑→生成↑）已废止。
- HYPIR-200 是确定性映射（四 seed 两两 PSNR 39.5–39.9 dB）。case4 主因是中频错误植物（E1 高需求 A=7/8），不是合理叶脉重采样，也不是抽样方差。
- 全图融合 α=0.2 均 PSNR 28.45，打平但赢不了 `texture_selective` 28.48。下一件唯一实验是残差置信融合（大残差不信 H200）。
- 分块（非重叠 256，960 块）：case3 块均值 ΔPSNR −8.93（平坦水面被造纹理）；case4 三层 ΔPSNR 几乎一样（约 −1.6 dB）且块 LPIPS 更好。块 LPIPS ≠ 全图 1024 LPIPS。
- 不建议在模糊区加强生成。文字/钟表优先少改；绿植要压错误中频和大残差幻觉，不是追 GT 叶脉。

## Scenario Routing v1 现有结果来源
- `baseline/experiments/structure_local_restoration_v1/metrics.csv` 已包含 case1-case5 的 LQ、HYPIR-50、HYPIR-200、`texture_selective_h200` 整图 PSNR/SSIM/LPIPS-Alex；其 Average 行给出当前基线 `texture_selective_h200` = 28.480280 / 0.781421 / 0.164546。
- `baseline/experiments/structure_local_restoration_v1/local_analysis.csv` 已包含四种目标方法在 LQ-only 区域 `strong_edge`、`textured_non_edge`、`blurred_texture` 的 `change_L1` 与 `error_to_GT_L1`（case4 才有非空 blurred_texture 区域）。这些结果可直接汇总，无需 diffusion 重算。
- 逐 case 的全局 PSNR 最优并非统一方法：case1 LQ-only fusion 中 `blurred_texture_h200` 略高于其他；case2 `texture_selective_h200`；case3 `texture_selective_h50`；case4 HYPIR-50；case5 HYPIR-50。四目标方法子集内，case1/2/3 为 texture-selective 或 H50，case4 为 H50，case5 为 H50。
- `structure_diagnosis/feature_correlation.csv` 显示 LQ 特征与 GT 所需变化的相关性有限（例如 local_variance 对 required_change Spearman 0.608，但对 H200 improvement Spearman 0.033；blur_proxy 对 improvement Spearman 0.149），说明 routing 规则应保持固定、解释性强，并在报告中明确验证集样本不足与过拟合风险。

## Scenario Routing v1 结论
- 固定规则实际将 case1（small face/text）、case3（bird）、case4（dense foliage）判为 `blurred/low-frequency` 并选择 HYPIR-200，将 case2（book spine text）和 case5（clock）判为 `ambiguous` 并保留 LQ。
- Routing v1 平均 PSNR/SSIM/LPIPS-Alex = 25.803636 / 0.722419 / 0.169598；相对当前 `texture_selective_h200` 的 28.480280 / 0.781421 / 0.164546 同时损害 PSNR、SSIM 和 LPIPS，未达到继续条件。
- 因此结论为 B：当前 LQ 全局统计量和固定规则没有显示可迁移的策略选择价值；停止 routing，不训练 classifier，不调规则，不新增 diffusion inference。

## 风险
- Diffusion 生成细节可能改字、改数字、改鸟眼/钟表指针；必须做 identity/边缘/文字一致性检查。
- 分块拼接会产生接缝；使用 overlap + 加权融合，并在全图缩略图上检查颜色连续性。
- 官方综合指标未知，不能只优化感知锐度；应保留原图混合比例和多候选结果。

## 环境与凭据
- 本机 GPU 为 NVIDIA GeForce RTX 5080 Laptop GPU（计算能力 12.0，显存约 16 GB，驱动 610.78）。旧 CUDA/PyTorch 组合可能无法生成或运行 `sm_120` 内核，应选 CUDA 12.8 及较新的 PyTorch。
- `数据集ossutil.md` 包含明文临时 AK/STS（有效期说明为 1 小时）。数据已经在本地，后续不需要使用这些凭据；应在阿里云侧立即撤销/轮换，并避免提交或传播该文件。
- 环境已验证：Python 3.11、torch 2.11.0+cu128、torchvision 0.26.0+cu128；`torch.cuda.is_available()` 为 True，RTX 5080 上 2048x2048 CUDA 矩阵乘法成功。
- 核心库版本已锁定到 `requirements-cu128.txt`，普通包使用清华 PyPI 镜像，CUDA wheel 使用阿里云 pytorch-wheels 镜像；`.conda\Lib\site-packages\sitecustomize.py` 提供旧版 Basicsr 的 torchvision 兼容别名。
- Pillow 可直接读取中文路径；OpenCV 的 `imread` 在该路径失败，应使用 `np.fromfile` + `cv2.imdecode`。
- `tb-nightly==2.21.0a20251023` 已从 PyPI 官方索引补装（清华镜像未同步），现在 `pip check` 报告无断裂依赖。

## Baseline 规划进展
- 当前仓库只有数据、环境与规划文档，尚无下载权重、推理实现或既有模型结果；baseline 应先以可控、零训练、可复现的推理闭环为第一目标。
- 验证样本与赛题均要求 4K 输入输出同尺寸。它们不是典型低分辨率到高分辨率的 4x 超分任务；将图像先缩小后做 4x 复原会无端丢失输入中仍然正确的像素，尤其危及文字、钟表和人脸。
- 主 baseline 应定位为同分辨率、低噪声的条件扩散修复（latent img2img / restoration diffusion），以重叠分块运行；将扩散生成结果按置信度与原图混合。Real-ESRGAN/GFPGAN 只用于非扩散对照、退化定位或局部先验，不作为满足赛题架构要求的主提交结果。
- 16 GB 显存可支撑 fp16、attention/vae slicing、512 或 768 像素工作块及 64--128 像素重叠，但不适合整图 4K latent diffusion。必须从一开始把分块拼接、随机种子和显存控制做进推理入口。

## 推荐的第一版 Baseline

### 主模型：低噪声 SD 1.5 img2img + ControlNet Tile
- 使用 Stable Diffusion 1.5 的 img2img 管线和 Tile 条件控制，在原始 4K 像素坐标上做重叠分块修复；不进行先缩小再 4x 放大的预处理。
- 起始配置：fp16、确定性 seed、tile 768（显存不足退至 512）、overlap 128、15--20 步、`denoise_strength=0.08` 和 `guidance_scale=1.5--2.5`。每图输出扩散结果 `D` 后，再与输入 `I` 做 `O=(1-alpha)I+alpha D` 的像素级融合。
- 初始 `alpha=0.15`。对 OCR 文本、脸部及强几何边缘施加 0.05--0.10 的低融合系数；对绿植、羽毛等非规则纹理最高允许 0.20--0.25。第一版先用边缘/OCR/人脸掩膜即可，鸟和钟表的专用检测待主链路验证有效后再补。
- Tile 条件必须输入原始块，并以 Hann/cosine 权重融合相邻输出，避免拼接缝；扩散前后在 Lab 色彩空间锚定低频颜色，拒绝明显色偏。

### 必须保留的对照
- `copy`：直接复制 LQ 到输出，给出真实下限，避免把有害生成误当增强。
- `classical`：轻量反卷积/锐化或 Real-ESRGAN 后回缩至原尺寸，只作为质量和退化分析对照，不作为主提交的 Diffusion 方案。
- `diffusion-fused`：同一组扩散输出离线生成 `alpha={0.05,0.10,0.15,0.20,0.25}` 的候选；融合不需要重复扩散采样，适合 5 对验证数据的小样本情形。

### 验证和选择规则
1. 先仅运行 5 组验证图，并记录每组的 PSNR、SSIM、LPIPS、色差，以及与输入的边缘一致性；保存原图、纯扩散和每个融合候选的可视化拼图。
2. 以对 `copy` 的平均 PSNR/SSIM 提升为硬门槛，同时要求文字、钟表两组不能下降；无参考质量指标仅用于同一保真水平候选之间的排序。
3. 网格只调三个量：`denoise_strength={0.05,0.08,0.12}`、Tile 控制强度和 `alpha`。一次扩散结果可复用多个 `alpha`，因此完整首轮最多 15 次采样，而不是 75 次。
4. 固定一套全局参数后才跑 100 张测试图。若验证集中没有候选稳定优于 `copy`，主提交应降级为极低融合强度，不应以感知锐化冒险牺牲有参考指标。

### 不作为首选的模型
- SUPIR：更强但显存、耗时与内容改写风险更高，适合后续在绿植/鸟类局部作为候选，而非初始 4K 全图链路。
- DiffBIR、StableSR、SeeSR：核心假设偏向低分辨率到高分辨率复原。对本赛题等分辨率轻退化，需要先缩小输入再恢复，会丢掉文字与几何信息；只有在验证集实测获益后才纳入第二阶段比较。

## HYPIR 官方仓库核验（2026-09-01）
- 仓库已克隆到 `HYPIR/`，HEAD `b61d107c6cef38f01a93c7833558869731cfa8c1`（2025-10-16，合并 tiled inference 更新）。
- 官方开源版本是 HYPIR-SD2：Stable Diffusion 2.1 base + `HYPIR_sd2.pth` LoRA 权重；`test.py` 是批量推理入口，唯一支持 `--base_model_type sd2`。
- README 官方命令固定 `model_t=200`、`coeff_t=200`、LoRA rank 256、`patch_size=512`、`stride=256`、`scale_by=factor`、`upscale=4`。比赛同分辨率要求必须将 `upscale` 改为 1，这是参数语义上的必要适配，不改网络结构。
- `BaseEnhancer.enhance()` 数据流：输入 tensor → 可选缩放 → 4K 图像 pad 到 8 的倍数 → `make_tiled_fn` VAE encode → tiled `forward_generator` → tiled VAE decode → crop/presize 回原尺寸 → `wavelet_reconstruction` 颜色/低频重建 → PIL/PNG 输出。
- 官方分块同时覆盖 VAE 编码、生成器前向和 VAE 解码；`patch_size`/`stride` 是显存和接缝控制的主要入口。
- 当前 `.conda` 环境与官方 requirements 存在版本差异：torch 2.11.0+cu128 / torchvision 0.26.0（为 RTX 5080 保留），diffusers 0.40.0、transformers 5.16.1、accelerate 1.14.0；`peft` 尚未安装。HYPIR 官方 pin 为 diffusers 0.32.2、transformers 4.49.0、accelerate 1.4.0、peft 0.14.0，需按最小范围对齐非 CUDA 包。
- 仓库自带 `HYPIR/examples/lq` 与 prompt，可用于无比赛数据时的连通性测试；本项目已挂载真实验证集 `csig_dataset/验证集/case*_lq.jpg` 和 GT，不需要伪造图片。
- 当前最小比赛 evaluation fixture 为 `baseline/evaluation_input/case1.jpg`（复制自 `csig_dataset/测试集/case1.jpg`，4096x3072 RGB JPEG）；有 GT 的 `baseline/input/case1_lq.jpg` 仅作为后续指标核验备用。
- 临时代理 `127.0.0.1:7890` 正在监听；通过 `http://127.0.0.1:7890` 访问 Hugging Face 可达，但 `whoami` 和 SD 2.1 base API 均返回 401，说明剩余阻塞是账号/许可而非网络。后续下载命令可通过临时 `HTTP_PROXY/HTTPS_PROXY` 使用该端口。
- 用户提供的 `sd-research/stable-diffusion-2-1-base` 为公开 Diffusers 仓库（commit `0708cecd370b4d1c3a6ff3f7332f5e9aea78896f`，非 gated），具备 HYPIR 所需的 `model_index.json`、`scheduler/`、`tokenizer/`、`text_encoder/`、`unet/`、`vae/`。主要 safetensors 大小为 text encoder 1,361,597,018 B、UNet 3,463,726,498 B、VAE 334,643,276 B。
- HYPIR README 指定的是 `stabilityai/stable-diffusion-2-1-base`，该官方 repo 当前对未登录客户端返回 gated 401。`sd-research` 的模型卡声明为 Stable Diffusion v2-1-base 且接口结构匹配，可作为第一阶段的公开 base 替代；无法在没有官方仓库读取权限时证明两个仓库权重逐字节一致。LoRA 加载中的 HYPIR key 断言将作为运行时兼容性校验。

## HYPIR 首次运行实测（2026-09-01）
- 使用本地 base `HYPIR/models/stable-diffusion-2-1-base`（`sd-research` 镜像）和官方 `HYPIR/test.py`，未修改源码。
- evaluation：输入 `baseline/evaluation_input/case1.jpg`，输出 `baseline/evaluation_output/result/case1.png`；输入输出均为 RGB、4096x3072，输出 uint8 范围 0--255、finite=True，无 OOM。
- evaluation 单图总耗时 58.13 s（含模型初始化 10.75 s）；CUDA 峰值 allocated 3.994 GiB，reserved 4.936 GiB。
- validation：输入 `baseline/input/case1_lq.jpg`，输出 `baseline/output/result/case1_lq.png`；总耗时 63.15 s，CUDA 峰值 allocated 3.994 GiB、reserved 4.936 GiB。
- validation `case1`（中文文字）指标：LQ 基线 PSNR 32.0288 dB / SSIM 0.94845；HYPIR 输出 PSNR 29.3353 dB / SSIM 0.84663 / LPIPS-Alex 0.45266。该结果仅用于侦察，显示默认 HYPIR 生成强度会改写文字细节，不能直接作为比赛提交参数。
- 目视检查 evaluation 输出无黑图/花屏，但招牌文字出现明显生成式重绘/字符变化风险，符合赛题要求的主要失败模式。
- 可控性代码位置：`HYPIR/enhancer/base.py:85-105` 的 `scale_by/upscale` 决定是否改变分辨率；`:117-148` 的 `patch_size/stride` 控制三段 tiled 推理；`HYPIR/enhancer/sd2.py:50` 的 `model_t` 与 `:62` 的 `coeff_t` 控制一步 DDPM 反演强度；`:56-64` 为 UNet LoRA 生成和 scheduler `pred_original_sample`；`base.py:150-153` 裁剪、回缩和 `wavelet_reconstruction` 锚定输入低频颜色。
- 内容保持风险主要发生在 `SD2Enhancer.forward_generator()` 的 UNet+LoRA 输出及 scheduler 还原 latent；`wavelet_reconstruction` 只能保留输入低频颜色，不能保证文字/数字语义一致。

## 图像对比可视化工具（2026-09-01）

### Evidence
- `baseline/compare_images.py` 只依赖 Pillow，读取输入、HYPIR 输出和可选 GT，不调用 HYPIR 模块。
- evaluation 默认目录为 `baseline/evaluation_input/` 与 `baseline/evaluation_output/result/`；验证集命令额外指定 `--ground-truth-dir csig_dataset/验证集`。
- 实测验证命令生成 `baseline/comparison/case1_lq_compare.png`：Input/LQ、HYPIR Output、Ground Truth 三栏均为 2048x1536，统一缩放比例 0.5，原图均为 4096x3072。
- `tests/test_compare_images.py` 共 5 项测试通过；重复运行后 HYPIR 输出 SHA-256 保持不变。

### Finding
- 工具按 stem 匹配 evaluation 输入/输出；验证集将 `case1_lq`、`case1_gt` 归一到 `case1`，支持不同扩展名和后续批量 case。
- 缩放只发生在 combine 展示副本，使用同一个比例计算所有面板尺寸；不裁剪、不覆盖、不重编码 HYPIR 原始结果。
- 顶部标签包含 `Input`、`HYPIR Output`、`Ground Truth`（有 GT 时）、文件名和原始分辨率。没有 GT 的 evaluation 样本保持二联图。

### Path
- 生产脚本：`baseline/compare_images.py`
- 测试：`tests/test_compare_images.py`
- evaluation 结果：`baseline/comparison/case1_compare.png`
- validation 三联结果：`baseline/comparison/case1_lq_compare.png`

### 可复现命令
```powershell
& .\.conda\python.exe baseline\compare_images.py

& .\.conda\python.exe baseline\compare_images.py `
  --input-dir baseline\input `
  --output-dir baseline\output\result `
  --ground-truth-dir csig_dataset\验证集 `
  --comparison-dir baseline\comparison
```

## 验证集五图 baseline 批量运行（2026-09-01）

### Evidence
- 使用官方 `HYPIR/test.py`，参数保持已验证 baseline：`model_t=200`、`coeff_t=200`、`lora_rank=256`、`patch_size=512`、`stride=256`、`scale_by=factor`、`upscale=1`、空 prompt、seed 231、CUDA。
- 输入目录：`baseline/input/`，包含 `case1_lq.jpg` 至 `case5_lq.jpg`；输出目录：`baseline/output/result/`。
- 5 张输出均成功保存，无 OOM；总墙钟时间约 348.9 秒（模型加载 11.59 秒）。
- 输出健康检查：RGB、uint8、finite、0--255；分辨率与输入完全一致，case1/3/4/5 为 4096x3072，case2 为 3072x4096。

### Finding
- 5 组默认 HYPIR 输出的 PSNR/SSIM 均低于原始 LQ，尤其 case4（密集绿植）和文字/鸟类样本下降明显；这证明官方 baseline 已跑通，但默认生成强度存在内容保真风险。
- 5 个三联图已写入 `baseline/comparison/case1_lq_compare.png` 至 `case5_lq_compare.png`，可直接用于逐组检查 Input/LQ、HYPIR Output 与 GT。
- 复核 case3/case4 原图级输出后确认：case3 鸟的轮廓边界不自然，背景滩涂/水面纹理被改成与 LQ/GT 不同的密集高频模式；case4 baseline 不仅有叶片纹理过生成，还出现 LQ/GT 均不存在的鹦鹉/蜥蜴样彩色物体、疑似动物头部和脸样结构，属于对象级 hallucination。

### Metrics
| Case | LQ PSNR | HYPIR PSNR | LQ SSIM | HYPIR SSIM |
|---|---:|---:|---:|---:|
| case1 | 32.0288 | 29.3353 | 0.94845 | 0.84663 |
| case2 | 28.1894 | 24.2140 | 0.83144 | 0.73105 |
| case3 | 35.6221 | 28.8654 | 0.93496 | 0.81210 |
| case4 | 18.0562 | 16.3524 | 0.30932 | 0.25760 |
| case5 | 26.2756 | 23.8758 | 0.86432 | 0.77725 |

### Path
- 输入：`baseline/input/case*_lq.jpg`
- HYPIR 输出：`baseline/output/result/case*_lq.png`
- 三联对比：`baseline/comparison/case*_lq_compare.png`

## Evaluation 指标脚本（2026-09-01）

### Evidence
- 脚本：`baseline/evaluate_metrics.py`；测试：`tests/test_evaluate_metrics.py`。
- 默认输入为 `baseline/input/`、`csig_dataset/验证集/`、`baseline/output/result/`；CSV 为 `baseline/evaluation_metrics.csv`。
- 实际 CSV 包含 5 个 case 行和 1 个 `Average` 行，字段覆盖 PSNR、SSIM、LPIPS-Alex、Delta 和三类结论。
- 8 项测试全部通过；尺寸不一致与非 RGB 图像均被显式拒绝。HYPIR tracked files unchanged。

### Finding
- LQ、GT、Output 先按公共 case key 匹配，再逐图校验 mode/channels/size；任何 mismatch 都抛出错误，不做隐式 resize。
- PSNR/SSIM 使用原始已校验尺寸；LPIPS-Alex 使用同一原始尺寸的 RGB tensor，模型只初始化一次并在 CUDA 上复用。
- 本次 5 组平均结果：PSNR 28.0344→24.5286（Δ-3.5058）、SSIM 0.77770→0.68493（Δ-0.09277）、LPIPS-Alex 0.35653→0.46100（Δ+0.10447）。

### 可复现命令
```powershell
& .\.conda\python.exe baseline\evaluate_metrics.py `
  --lq-dir baseline\input `
  --gt-dir csig_dataset\验证集 `
  --output-dir baseline\output\result `
  --csv baseline\evaluation_metrics.csv `
  --device cuda
```

### Path
- CSV：`baseline/evaluation_metrics.csv`
- 生产脚本：`baseline/evaluate_metrics.py`
- 测试：`tests/test_evaluate_metrics.py`

## coeff_t 单变量 sweep（2026-09-01）

### Evidence
- `HYPIR/test.py` 将 CLI `coeff_t` 传入 `SD2Enhancer`；`HYPIR/HYPIR/enhancer/sd2.py:62` 将其作为 `scheduler.step(eps, self.coeff_t, z_in)` 的 timestep，`model_t` 在 `:50` 作为 UNet 输入 timestep。
- 五组独立目录 `baseline/experiments/coeff_t_{200,150,100,75,50}/` 均有 5 张输出、5 张三联图、metrics CSV、metadata 和 inference.log；输出 RGB、finite 且尺寸与输入一致。
- 汇总 CSV：`baseline/experiments/coeff_t_sweep_summary.csv`；报告：`baseline/experiments/coeff_t_sweep_report.md`。

### Metrics
| coeff_t | Avg PSNR | ΔPSNR vs LQ | Avg SSIM | ΔSSIM vs LQ | Avg LPIPS | ΔLPIPS vs LQ | elapsed(s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 24.528587 | -3.505833 | 0.684926 | -0.092773 | 0.461005 | +0.104473 | 316.423 |
| 150 | 26.213323 | -1.821097 | 0.734088 | -0.043611 | 0.400391 | +0.043859 | 350.165 |
| 100 | 27.504504 | -0.529916 | 0.762503 | -0.015196 | 0.362717 | +0.006185 | 369.286 |
| 75 | 27.957945 | -0.076475 | 0.771081 | -0.006618 | 0.350124 | -0.006408 | 355.261 |
| 50 | **28.257533** | **+0.223113** | **0.776935** | **-0.000764** | **0.338965** | **-0.017567** | 342.774 |

### Findings
- 在本次固定条件和候选集合中，降低 coeff_t 的平均指标趋势与降低 model_t 的趋势相反；coeff_t=50 是候选最佳但位于搜索下界，不能视为全局最优。
- coeff_t=50 相对 coeff_t=200 的平均差异：PSNR +3.728946 dB、SSIM +0.092009、LPIPS -0.122040。
- case1/case2/case5 三联图显示高频文字/边缘重绘减弱；缩略图无法证明逐字符、逐数字或指针端点完全一致，需避免语义级确定断言。
- case3 水面和羽毛高频生成减弱但三项仍未超过 LQ；case4 在 coeff_t=50 下三项略超过 LQ，且未见 baseline 中明显的彩色对象样结构，但局部幻觉不能由缩略图排除。

### Next hypothesis
- 先在 coeff_t=25--75 区间做更密集的单变量搜索定位拐点，再以少量不同 seed 检查文字/钟表稳定性；保留当前 model_t=200、base、LoRA 和 tiled 参数。

## Phase 2 confidence_fusion_v1（2026-09-01）

### Evidence
- 已验证并复用 `coeff_t_200` 与 `coeff_t_50` 五张输出，没有新 diffusion 推理；所有新文件位于 `baseline/experiments/confidence_fusion_v1/`。
- LQ-derived map 使用 1/4 分辨率 Sobel/Laplacian/local std 与 high-frequency residual，百分位归一化、Gaussian smoothing、median 二分 mask；它是 structural reliability heuristic，不是 semantic confidence。
- 五张图均生成 fixed_50_50、confidence_200_50、confidence_200_LQ、七面板 comparison 和 error-map；输出健康检查通过。

### Metrics
- Average: LQ 28.034420/0.777699/0.356532；HYPIR-200 24.528587/0.684926/0.461005；HYPIR-50 28.257533/0.776935/0.338965；fixed 50/50 26.858639/0.747600/0.365744；confidence 200/50 27.291166/0.766543/0.332414；confidence 200/LQ 28.048711/0.776696/0.317280（PSNR/SSIM/LPIPS）。
- HYPIR-200 相对 LQ 的平均 high/low region L1 为 14.761/6.436；confidence 200/50 为 11.858/5.427；confidence 200/LQ 为 11.358/5.261。
- 相对 LQ 的平均改写 L1 high/low：HYPIR-200 12.333/3.491，HYPIR-50 4.260/1.134，confidence 200/50 6.949/1.378，confidence 200/LQ 4.588/0.448；空间加权降低 low-confidence 改写，但未证明恢复真实细节。

### Finding
- confidence map 有合理平滑空间分布，且与 HYPIR 改写量存在可测关系；但当前 median split 只能说明控制了变化幅度，不能证明“低置信区域=错误区域”。
- fixed 50/50 没有收益；confidence 200/50 不及 global HYPIR-50；confidence 200/LQ 的 LPIPS 最低主要来自保守地贴近 LQ。
- case1/case2/case5 文字/数字/指针视觉重绘较 HYPIR-200 减少，但没有 OCR/身份级证据；case3 高响应纹理变化减少但仍低于 LQ；case4 模糊绿植区域更保守，object-like 结构保持中性描述。
- 总结：Q10=WEAK EVIDENCE；阶段结论 Weak success，不扩展大规模 sweep 或新模型。

## Phase 3 structure diagnosis (2026-09-01)

### Evidence
- Existing `coeff_t_200` and `coeff_t_50` experiment metadata and five outputs each were verified and reused. The diagnosis script performs no HYPIR import or diffusion inference.
- Native-resolution patch grid: 256x256 with stride 128, edge-aligned; 713 patches per case and 3,565 total. Metrics include L1/MSE/PSNR/SSIM, gradient/edge differences, LQ-to-output change, improvement, and change alignment.
- New artifacts are isolated under `baseline/experiments/structure_diagnosis/`: `patch_metrics.csv`, `patch_summary.csv`, `patch_quantiles.csv`, `feature_correlation.csv`, `metadata.md`, `report.md`, case-local maps/top-patch CSVs, `oracle_maps/`, `change_maps/`, and coordinate-audited crop sheets.

### Metrics
- H200 mean patch improvement L1 is -2.1084; H50 is +0.1556. Mean generated-change L1 is 7.9598 (H200) versus 2.7131 (H50).
- Strict quantile categories: 15 Type-A useful-change patches, 865 Type-B/Type-D harmful-or-mismatch candidates, and 10 Type-C high-required-change but low-generated-change candidates.
- LQ feature correlations with required change are moderate (gradient Spearman 0.730, entropy 0.752, Laplacian 0.797), but correlations with H200 improvement are weak; the strongest is blur_proxy Spearman 0.149.

### Findings
- Q1/Q2: H200 large changes do not generally correspond to GT-required changes; harmful/mismatch candidates substantially outnumber strict useful changes.
- Q3: missed-restoration candidates exist but are fewer under the chosen quantile rule; they are explicitly listed rather than inferred from low change alone.
- Q4: H50's improved global fidelity is consistent with reducing erroneous changes, not proof that every missing detail is recovered.
- Q5: Localized real recovery evidence exists, concentrated in 14 case5 and 1 case1 Type-A patches.
- Q6: LQ features predict required difference better than they predict H200 improvement; current evidence is insufficient for a reliable LQ-only controller.
- Q7/Q8: Text and clock crops support structure-preserving deblur and geometry/edge constraints; no OCR or identity claims are made.
- Q9: Bird/case3 and ambiguous foliage/case4 show many Type-D candidates, supporting an image-space ambiguity-to-generation risk description.
- Q10: Recommendation is a minimal LQ-driven structure-anchored fusion plus edge-preserving local control study; do not start LoRA or broad coeff_t sweeps yet. Stop if held-out/seed checks fail to reproduce localized benefits.

## Phase 4 structure-anchored local restoration v1 (2026-09-02)

### Method and artifacts
- Independent script: `baseline/experiments/structure_local_restoration.py`; tests: `tests/test_structure_local_restoration.py`.
- Inputs are the existing validation LQ, GT, coeff_t=200 (HYPIR-200), and coeff_t=50 (HYPIR-50) PNGs. The script does not import HYPIR or run diffusion. It also reads the existing `confidence_200_LQ` output only as a comparison when available.
- Three fixed, interpretable LQ-only strategies use quarter-resolution smoothed features: `structure_guard` (`0.04+(1-edge)*(0.28+0.08*texture)`), `texture_selective` (`0.05+0.46*texture*(0.35+0.65*(1-edge))`), and `blurred_texture` (`0.05+(0.40*blur*texture+0.08*texture)*(1-edge)`). Every map is clipped to [0, 0.60].
- All new artifacts are isolated under `baseline/experiments/structure_local_restoration_v1/`: color/grayscale weight maps, 30 fusion images, 15 comparison panels, `metrics.csv`, `local_analysis.csv`, `weight_summary.csv`, `experiment_metadata.md`, `README.md`, and `report.md`.

### Metrics and local evidence
- Native PSNR/SSIM and uniformly resized (max side 1024) LPIPS-Alex are computed for all 5 cases, LQ/HYPIR baselines, existing confidence_200_LQ, and six fusion variants.
- Average PSNR/SSIM/LPIPS: LQ `28.034420/0.777699/0.204315`; HYPIR-50 `28.257533/0.776935/0.162039`; HYPIR-200 `24.528587/0.684926/0.159666`; confidence_200_LQ `28.048711/0.776696/0.148463`; structure_guard_h200 `28.335641/0.774326/0.165913`; texture_selective_h200 `28.480280/0.781421/0.164546`; texture_selective_h50 `28.327109/0.782096/0.193797`.
- Across LQ-derived strong-edge regions, HYPIR-200 change/error L1 is `19.9579/20.9474`; structure_guard_h200 is `3.3265/15.4210`; texture_selective_h200 is `4.5728/15.2215`. Thus edge protection reduces rewriting and GT error relative to HYPIR-200, but HYPIR-50 remains a conservative reference with lower strong-edge error `15.1776`.
- In LQ-derived textured non-edge regions, HYPIR-200 error is `9.3239`; structure_guard_h200 and texture_selective_h200 are `8.2152` and `8.1972`, while their changes are only `1.4835` and `1.3618`. This supports selective texture blending as a promising but small effect, not a universal high-frequency rule.

### Answers to A-D
- **A:** Yes for controlling amount of change in an image-space sense: edge features lower the diffusion weight and measured edge-region rewriting falls sharply. No evidence yet that the features reliably predict GT-aligned improvement; Phase 3 correlations were weak.
- **B:** Text/book-spine/clock-like structures are plausibly helped by reduced edge rewriting, but semantic identity of characters, numerals, and pointers is unverified. No OCR/object claim is made.
- **C:** Texture regions justify a higher allowance only when local variance is high and strong-edge response is lower. `texture_selective_h200` is the best average PSNR/SSIM candidate, but LPIPS does not dominate HYPIR-50 and the effect is not validated beyond five cases.
- **D:** Proceed only to a small held-out/seed local-control check. Do not launch broad parameter sweeps, LoRA, or HYPIR backbone edits. Stop if the edge reduction or texture error improvement does not reproduce.

## Phase 5 texture-region HYPIR weight sweep (2026-09-02)

### Method and artifact contract

- Independent script: `baseline/experiments/texture_weight_sweep.py`; tests: `tests/test_texture_weight_sweep.py`.
- Inputs: existing validation LQ/GT and `baseline/experiments/coeff_t_200/output/result`; no HYPIR import/inference, no LoRA/training/model changes.
- The v1 `texture_selective` map is copied outside `texture >= P80 & edge < P80`. Inside that exact mask, the direct HYPIR blend weight is set to one of `0.25, 0.40, 0.55, 0.70, 0.85, 1.00`. `strong_edge` is therefore invariant by construction and by real-case audit.
- Complete artifacts are under `baseline/experiments/texture_weight_sweep_v2/`: 35 fusion PNGs, 70 maps, 5 comparison panels, `metrics.csv`, `local_analysis.csv`, `experiment_metadata.md`, and `report.md`.

### Full-image evidence

| Method | PSNR | SSIM | LPIPS-Alex |
|---|---:|---:|---:|
| v1 `texture_selective_h200` | 28.480280 | 0.781421 | 0.164549 |
| weight 0.25 | 28.479684 | 0.781403 | 0.164545 |
| weight 0.40 | 28.476294 | 0.781214 | 0.164317 |
| weight 0.55 | 28.468252 | 0.780793 | 0.164004 |
| weight 0.70 | 28.456087 | 0.780213 | 0.163652 |
| weight 0.85 | 28.439112 | 0.779462 | 0.163265 |
| weight 1.00 | 28.418456 | 0.778650 | 0.162872 |

PSNR and SSIM degrade monotonically as direct texture generation rises. LPIPS improves slightly, but that perceptual decrease is not accompanied by lower GT error in the main texture region.

### Regional evidence

- Average `strong_edge` change/error is exactly `4.5728/15.2215` for v1 and every candidate. This confirms case1/2 text and case5 clock protection is unchanged by this sweep.
- Average `textured_non_edge` GT error is v1 `8.1972`, then `8.2142/8.2505/8.3922/8.6282/8.9448/9.3239` for weights `0.25/0.40/0.55/0.70/0.85/1.00`; every candidate is worse.
- Average `blurred_texture` GT error has a small low-weight dip (`29.3456 -> 29.2994 -> 29.2481` at 0.25/0.40), then worsens to `29.7000/30.6161/32.0222/33.9444`. This is not a stable monotonic gain and includes a very small case4 mask.
- Case3 bird is a clear negative: `textured_non_edge` error increases from `2.8215` at v1 to `2.8735/3.1457/3.5038/3.9217/4.3795/4.8569`; full PSNR falls from `35.6301` to `35.6275/35.6120/35.5860/35.5519/35.5065/35.4544`.
- Case4 dense foliage is mixed but not a success: `textured_non_edge` error is `23.9475` at v1 and `23.9469/24.0023/24.1689/24.4373/24.8000/25.2500`; only the tiny `blurred_texture` subset improves at 0.25/0.40, while full PSNR declines `18.0199 -> 18.0198/18.0176/18.0135/18.0077/18.0001/17.9907`.
- Case1 text textured-region error worsens for every candidate (`1.8811 -> 1.8972 ... 2.5196`). Case2 has a narrow regional minimum at 0.40 (`5.5922` vs v1 `5.6488`) but full PSNR is best at 0.25 and no cross-case consistency follows. Case5 clock strong-edge metrics are unchanged; its textured-region error has a local minimum at 0.55 (`6.5491` vs v1 `6.6873`) while full-image PSNR also peaks at 0.55, making it a case-specific exception.

### Decision

The core hypothesis fails on the required evidence. Increasing the texture-region HYPIR weight mostly increases generated change and raises GT error in `textured_non_edge`; the bird case is consistently harmed, and the foliage case has only a narrow, tiny-subregion dip without full-image improvement. The best full-image PSNR weight is `0.25` for cases1-4 and `0.55` for case5, so the five cases are not consistent. Stop this direction: do not advance to a broader texture-weight search or local controller based on this hypothesis. Retain v1 as the stronger controlled result and only revisit local control under a different, explicitly GT-validated hypothesis.

### Audit

- Real-case map invariance: 30 candidate maps checked; no strong-edge/outside-mask mismatch.
- Output health: all 35 fusion PNGs RGB, finite, 0-255, dimension-matched; 20-test suite passes.
- HYPIR tracked source remains unmodified. The first unoptimized run timed out at the tool limit; its incomplete directory is preserved separately and is not used for conclusions.

## Structure-Anchored Residual Fusion v2 (2026-09-02)

- Implemented an independent offline experiment at `baseline/experiments/structure_anchored_residual_fusion_v2/`. It reuses LQ, existing coeff_t=50/200 outputs, and the current `texture_selective_h200` output; no new diffusion inference, training, or LoRA update was run.
- Gate behavior is fixed and LQ-anchored: strong edge 0.020 cap, weak edge 0.100, nearby structure 0.065, non-edge 0.035; H200 residual novelty suppresses the gate and aligned gradients add at most 0.035. GT is post-fusion evaluation only.
- Average metrics: LQ 28.034420/0.777699/0.204315, H50 28.257533/0.776935/0.162039, H200 24.528587/0.684926/0.159666, current texture-selective H200 28.480280/0.781421/0.164546, v2 28.158209/0.779548/0.195091 (PSNR/SSIM/LPIPS-Alex).
- Regional evidence supports structure preservation but not detail recovery: v2 strong-edge change_L1 is 0.481077, versus H50 8.794542 and texture-selective H200 5.551991; v2 textured_non_edge and blurred_texture GT errors remain above the current texture-selective baseline.
- The required visual checks cover case1 text, case2 book spine, case3 bird, case4 foliage, and case5 clock. Fusion/gate outputs are dimension-matched RGB finite PNGs; comparison panels are intentionally resized composites.
- Decision: stop this direction. v2 does not clearly improve over H50 or current texture-selective H200, so do not launch another sweep or training based on this gate.

## Adaptive H50/H200 Fusion v1 (2026-09-02)

### Method
- Independent package/script: `baseline/experiments/adaptive_fusion_v1/experiment.py`; tests: `tests/test_adaptive_fusion_v1.py`.
- Fusion formula is exactly `F(x)=(1-alpha(x))*H50(x)+alpha(x)*H200(x)`. Alpha uses only LQ-derived Sobel gradient magnitude, local variance texture strength, blur proxy, and strong-edge mask; GT is evaluation-only.
- Fixed formula: `clip(0.035 + 0.11*texture + 0.045*edge*(1-strong_edge) + 0.035*blur*texture*(1-strong_edge) - 0.18*strong_edge, 0, 0.30)`. No sweep or learned parameter.

### Metrics
- Average PSNR/SSIM/LPIPS-Alex: LQ `28.034420/0.777699/0.204318`; HYPIR-50 `28.257533/0.776935/0.162039`; HYPIR-200 `24.528587/0.684926/0.159664`; `texture_selective_h200` `28.480280/0.781421/0.164548`; adaptive `28.243879/0.776406/0.157813`.
- Adaptive is below the supplied current best in PSNR by `0.236401 dB` and SSIM by `0.005015`; LPIPS is lower by `0.006733`, but this does not compensate for the fidelity metrics.
- Per-case PSNR delta adaptive minus texture-selective: case1 `-0.011137`, case2 `-0.262750`, case3 `-0.998636`, case4 `+0.050895`, case5 `+0.039621`.
- Regional averages: strong-edge adaptive change/error `7.1654/15.1776`; textured-non-edge `2.5214/8.4063`; blurred-texture `5.5179/28.9512`. CSV includes the requested `change_L1` and `error_to_GT_L1` fields for each method and region.

### Decision
- The continuous alpha map successfully keeps strong edges at alpha zero and yields bounded, interpretable spatial variation, but the five-case global metrics do not beat the texture mask baseline. It is not demonstrated to be more effective than `texture_selective_h200`.
- Case4 foliage and case5 clock gain small PSNR; case1 text, case2 book spine, and case3 bird lose PSNR. The mixed case behavior and lower average PSNR/SSIM fail the continue criterion.
- **Stop adaptive fusion direction.** Do not perform another alpha/texture sweep, add training, or add diffusion inference under this hypothesis.

## LoRA / Adapter training feasibility audit (2026-09-02, evidence collected)

### Local code and checkpoint facts
- `HYPIR/HYPIR/enhancer/sd2.py` loads SD2.1 `UNet2DConditionModel`, calls `add_adapter()` for the official LoRA, loads `weights/HYPIR_sd2.pth`, then freezes the complete U-Net. `forward_generator()` currently has no adapter residual arguments. The installed Diffusers 0.32.2 U-Net API does accept `down_block_additional_residuals` and `mid_block_additional_residual`.
- Metadata-only checkpoint inspection found 514 LoRA tensors / 257 LoRA module pairs, with 259,474,432 fp32 parameters (1,037,897,728 serialized bytes). Checkpoint entries cover every down/up block, mid block, and `conv_out`; configured targets are attention projections, feed-forward projections, and convolution/residual paths at rank 256.
- Exact SD2.1 latent shapes for a 512x512 RGB crop were verified on the U-Net config with a meta-device forward. The latent is `[B,4,64,64]`; selected additional residual slots are index 3 `[B,320,32,32]`, index 6 `[B,640,16,16]`, index 9 `[B,1280,8,8]`, and the mid residual `[B,1280,8,8]`. The whole residual tuple has 12 down-skip slots including `conv_in`.
- The official `SD2Trainer` is not an Adapter MVP starting point: it creates fresh trainable LoRA parameters and also instantiates a trainable ConvNeXt discriminator plus VGG-LPIPS. Its Real-ESRGAN degradation recipe includes a second stage with `stage2_scale: 4`, which mismatches the same-resolution competition restoration regime.

### Runtime facts
- The project environment reports torch 2.11.0+cu128, CUDA build 12.8, Diffusers 0.32.2, Accelerate 1.4.0, PEFT 0.14.0, Transformers 4.49.0. The RTX 5080 Laptop has 15.920 GiB and supports bf16; the D: drive has 107.27 GiB free.
- Prior measured HYPIR inference at batch 1 / 512-pixel tile / bf16 peaked at 3.994 GiB allocated and 4.936 GiB reserved. This is inference-only evidence, not a training memory measurement.

## 2026-09-03 - E1 SwinIR reconnaissance and official verification
- Repository has no project-local SwinIR/Real-ESRGAN/ESRGAN/BSRGAN inference script or restoration checkpoint. The environment contains `basicsr==1.4.2` and `realesrgan==0.3.0`; `basicsr.archs.swinir_arch.SwinIR` imports successfully.
- Official SwinIR README identifies task 006 as JPEG compression artifact reduction and lists color `color_jpeg_car` JPEG qualities 10/20/30/40. It names `006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth` for the color JPEG40 checkpoint.
- Official `main_test_swinir.py` confirms for `color_jpeg_car`: `upscale=1`, `in_chans=3`, `img_size=126`, `window_size=7`, `img_range=255`, depths `[6,6,6,6,6,6]`, embed_dim `180`, num_heads `[6,6,6,6,6,6]`, `mlp_ratio=2`, `upsampler=''`, `resi_connection='1conv'`, and checkpoint key `params`.
- The official test code pads by reflected flips to a multiple of window size, crops output back to original dimensions, and requires tiled `tile % window_size == 0`; default `tile_overlap` is 32. A 512 tile is invalid for window 7, so the planned tile is 504 with overlap 32 unless smoke testing shows incompatibility.
- Five LQ inputs are JPEG RGB files at 4096x3072 (case1/3/4/5) and 3072x4096 (case2). JPEG storage is confirmed, but no claim is made that their effective degradation equals synthetic JPEG quality 40; E1 is a reasonable pretrained fidelity baseline only.

## 2026-09-03 - E1 execution completed

- Fixed direct-script project-root import and corrected SwinIR official range handling: input `[0,255]` uint8 -> `[0,1]` tensor; output `[0,1]` -> `[0,255]` before PNG.
- Contract tests: `6 passed` (`tests.test_e1_swinir_inference`).
- Ran exactly case1-case5 with tile 504, overlap 32, window 7; no OOM. Five RGB original-size PNGs are saved under `baseline/experiments/E1_fidelity/swinir_car_jpeg40/output/`.
- Fidelity average: PSNR `28.028629`, SSIM `0.780163`, LPIPS-Alex `0.216824`; HYPIR-50 average: `28.257533`, `0.776935`, `0.162039`; deltas: `-0.228905`, `+0.003227`, `+0.054786`.
- Model load `0.547554 s`; per-case forward times `156.498147/181.486418/186.059771/190.324722/197.396713 s`; peak allocated/reserved `3.335263/4.683594 GiB`; end-to-end command wall time approximately `1112.3 s`.
- Report written to `reports/E1_FIDELITY_BASELINE.md`; no E2 started.

## 2026-09-03 - E2 Global Alpha Blend completed

- Inputs HYPIR-50, E1 Fidelity, and GT passed strict RGB/uint8/range/resolution checks for case1-case5. No resize or model re-inference was performed.
- Generated exactly 30 E2 RGB PNGs: six global alphas (`0.0, 0.2, 0.4, 0.6, 0.8, 1.0`) times five cases. Endpoint pixel checks passed: alpha 0.0 equals Fidelity; alpha 1.0 equals HYPIR-50.
- Final metrics use the E0 `_metric` and batched `_lpips_scores` protocol. Average PSNR best: alpha 0.6 = `28.507181`; SSIM best: alpha 0.4 = `0.784890`; LPIPS best: alpha 1.0 = `0.162039`.
- Versus HYPIR-50, alpha 0.6 changes average PSNR/SSIM/LPIPS by `+0.249647/+0.006586/+0.015519`; no blend improves LPIPS. Versus Fidelity, alpha 0.2-0.8 improve all three average metrics.
- E2 report and complete metrics are saved at `reports/E2_GLOBAL_BLEND.md` and `baseline/experiments/E2_global_blend/metrics.csv`. No E3 started.

## 2026-09-03 - E3-A Texture-only Adaptive Alpha started
- User-authorized scope is exactly three offline methods: global alpha 0.6, LQ-only texture-driven alpha in [0.30, 0.70], and its reversed mapping. Existing HYPIR-50 and SwinIR images must be reused without re-inference.
- E2 already provides a tested shared metric protocol: native-resolution `skimage` PSNR (`data_range=255`) and SSIM, with LPIPS-Alex using the longest-side <=1024 preprocessing rule. E3-A should reuse this implementation rather than alter metrics.
- Workspace is not a Git worktree (`git diff` fails); source/input immutability will be checked from artifact paths, hashes/timestamps where relevant, and explicit no-inference code inspection.

## 2026-09-04 - E3-A Texture-only Adaptive Alpha completed
- Fixed method implementation: `global_alpha_0.6` uses `0.6 H + 0.4 F`; `adaptive_texture` uses `alpha=0.30+0.40T`; `adaptive_reverse` uses `alpha=0.70-0.40T`. Both adaptive maps share one LQ-only T and fixed Gaussian sigma=8; all alpha values are bounded [0.30,0.70].
- Average metrics: global alpha `28.507181/0.783521/0.177557`; adaptive texture `28.475310/0.785022/0.191381`; adaptive reverse `28.499589/0.783138/0.177883` (PSNR/SSIM/LPIPS-Alex).
- Adaptive texture fails the requested improvement test versus global alpha 0.6: PSNR decreases 0.031871 dB and LPIPS increases 0.013824, while SSIM increases 0.001501. The per-case pattern is mixed, with PSNR gains only on case1 (+0.006189) and case3 (+0.188234).
- Reverse mapping is a meaningful ablation spatially (mean alpha 0.659921 and correlation -1.0 versus forward mean 0.340079 and correlation +1.0), but its average metrics are slightly below global; it does not establish a better direction.
- Output contract, map dimensions, and E2 global-alpha pixel identity were verified. Visual inspection found coherent, nonblank texture maps and inverse alpha maps. Full unit suite passes 53 tests. Stop E3-A here; do not launch E3-B/E3-C or other experiments.
