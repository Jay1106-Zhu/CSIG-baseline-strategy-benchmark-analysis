# 进度日志

## 2026-08-31
- 阅读 `赛题.txt` 与 `数据集ossutil.md`。
- 清点数据：验证集 5 对、测试集 100 张，均为 RGB 4K，横竖比例约 71/29（测试集）。
- 目视核验验证样本，确认文字/书脊、鸟类、密集绿植、钟表等场景与赛题描述一致。
- 计算代理 PSNR 与 Laplacian 清晰度，确认 case4（绿植）退化最严重，case3（鸟类）较轻。

## 2026-09-01
- 检查项目规划文件与数据目录，确认测试集 100 张和验证集 5 对已在本地，无需再次下载。
- 检测到 RTX 5080 Laptop GPU（16 GB，计算能力 12.0，驱动 610.78），Conda 25.3.1；选择项目内 `.conda`、Python 3.11 和 CUDA 12.8 PyTorch wheel。
- 发现 `数据集ossutil.md` 暴露临时 AK/STS，已记录风险；不在安装流程中调用。
- 核心 restoration 依赖安装完成；补充包首次安装因 `basicsr` 的隔离构建环境找不到 torch 失败，准备改用非隔离构建并拆分安装。
- 完成 `basicsr/facexlib/gfpgan/realesrgan/open_clip_torch/pytorch-lightning` 等补充库安装，并加入 torchvision 兼容 shim。
- 生成 `requirements-cu128.txt`；全量导入检查通过，CUDA 可用且样本 RGB/分辨率读取验证通过。记录 OpenCV 中文路径需用 `imdecode` 的注意事项。
- 最终检查：环境位于 `.conda`（约 5.22 GB），CUDA 与 RTX 5080 计算能力 `(12, 0)` 正常；下载缓存位于 `.cache`（约 5.14 GB），因安全策略未自动删除，可在确认无需要后手动清理。
- 从 PyPI 官方索引补装 `tb-nightly==2.21.0a20251023` 后，`pip check` 已报告 `No broken requirements found`；重新生成并修正 `requirements-cu128.txt` 中的 CUDA wheel URL。
- 按请求将数据目录从 `赛题一\` 重命名为 `csig_dataset\`；已确认目标目录不存在且重命名成功。
- 开始细化当前 baseline 方案。发现同时解码全部 4K JPEG 的数据统计在 24 秒内超时，后续改为针对验证集和代表性测试图的轻量检查。
- 确认没有既有模型权重或推理代码；已根据同分辨率 4K、内容保真和 16 GB 显存约束，收敛到低噪声条件扩散修复加原图保真融合的 baseline 方向。额外并行指标调用在工具层失败且未返回细节，未重复执行；不影响该架构决策。
- 完成 baseline 规划：首选 SD 1.5 img2img + ControlNet Tile 的同分辨率重叠分块修复，以低噪声采样和原图融合控制内容漂移；确定了首轮消融、选择门槛及 SUPIR/DiffBIR/StableSR 的延后理由。
- 当前目录不是 Git 工作树，无法通过 `git diff` 展示文档变更；规划文件已直接写入并将通过读取校验。
- HYPIR 官方仓库已克隆并固定到 commit `b61d107c6cef38f01a93c7833558869731cfa8c1`；已阅读 README、`requirements.txt`、`test.py`、SD2 enhancer、tiled VAE 和 tiled window 代码。
- 确认真实验证图位于 `csig_dataset/验证集/`（`case1_lq.jpg` 至 `case5_lq.jpg`，各有对应 GT）；当前没有 HYPIR checkpoint。
- 发现当前环境缺少 `peft` 且 diffusers/transformers 版本高于官方 pin；下一步仅对齐官方 Python 依赖，保留 RTX 5080 所需的现有 torch 2.11.0+cu128/torchvision 0.26.0。
- 已按官方 pin 安装非 CUDA 依赖：accelerate 1.4.0、diffusers 0.32.2、transformers 4.49.0、peft 0.14.0 等；`pip check` 通过，导入 HYPIR SD2 和 BF16 CUDA 矩阵测试通过。
- 下载 `HYPIR/weights/HYPIR_sd2.pth` 完成，大小 1,038,090,392 字节，SHA-256 `D538A2CB925451FAB1F75ADFE715AC2B1C8BB12FA32A0851BA01BE2347B354D6`。
- Stable Diffusion 2.1 base 在 Hugging Face API 返回 401；`hf auth whoami` 显示未登录。已记录为当前唯一外部阻塞，不执行绕过 gating 的镜像或替代模型。
- 已建立两套最小单图目录：比赛测试集 `baseline/evaluation_input/case1.jpg`（4096x3072 RGB）/ `baseline/evaluation_output/`，以及带 GT 的验证集 `baseline/input/case1_lq.jpg`（4096x3072 RGB）/ `baseline/output/`。HYPIR `test.py` 运行测试图后将生成 `baseline/evaluation_output/result/case1.png`。
- 官方关键文件已通过 `py_compile`；未修改 HYPIR 源码。等待 HF 授权后继续 base 下载、单图推理和输出质量验证。
- 用户提供临时代理端口 7890；确认 `127.0.0.1:7890` 监听且经代理访问 Hugging Face 可达，但未登录仍返回 401。下载阶段将使用临时代理变量，不持久化系统代理。
- 核对用户提供的 `https://huggingface.co/sd-research/stable-diffusion-2-1-base`：公开、非 gated，含完整 Diffusers 目录，可作为当前运行所需的 SD2.1 base；与 HYPIR README 指定的 stabilityai repo id 不同，运行时将通过 LoRA state_dict key 断言验证兼容性。
- 用户完成 HF 登录后，Token `whoami` 验证为 `Jayniko`，但 stabilityai repo API 仍返回 404；未绕过权限，改用结构匹配的公开 `sd-research` 镜像并记录来源差异。
- 通过临时代理下载 base 12 个必需文件完成；首次下载因误启用缺失的 `hf_transfer` 失败，关闭可选加速后成功。
- 官方 HYPIR `test.py` evaluation 单图成功：case1 4096x3072 -> PNG 同尺寸；耗时 58.13 s，CUDA 峰值 allocated 3.994 GiB/reserved 4.936 GiB，无 OOM。
- 官方 HYPIR `test.py` validation 单图成功：case1_lq 4096x3072 -> PNG 同尺寸；耗时 63.15 s，峰值同上。PSNR 29.3353 dB、SSIM 0.84663、LPIPS-Alex 0.45266；原始 LQ 对 GT 为 PSNR 32.0288 dB、SSIM 0.94845，默认生成强度损害文字保真。
- 输出健康检查通过（RGB、uint8、finite、0--255）；目视确认无黑图/花屏，但 evaluation 招牌文字存在字符重绘风险。HYPIR baseline 跑通，下一步只应做验证集保真评估，不直接扩展 100 张或改网络。
- 新增独立图像对比工具 `baseline/compare_images.py`，不导入、不修改 HYPIR；默认匹配 `baseline/evaluation_input/` 与 `baseline/evaluation_output/result/` 同 stem 图片并生成二联图。
- 对验证集增加可选 `--ground-truth-dir`：`case1_lq` 输入/输出会按公共 case key 匹配 `case1_gt`，生成 `Input/LQ | HYPIR Output | Ground Truth` 三联图；后续 `case1`--`case100` 可批量复用同一命令。
- 对比图仅对展示副本统一缩放（默认最大面板 2048x2048），三张图使用同一 scale，不裁剪、不改变 HYPIR 输出文件；顶部记录角色、文件名和原始分辨率。
- 已生成 `baseline/comparison/case1_compare.png`（evaluation 二联图）和 `baseline/comparison/case1_lq_compare.png`（验证集三联图）；三联图画布 6240x1638，三个面板均为 2048x1536，对应原图 4096x3072。
- 新增 `tests/test_compare_images.py`，覆盖扩展名不同的 stem 匹配、缺失输入跳过、统一缩放/完整内容、GT 三联图和 LQ/GT case 匹配；共 5 项测试通过，脚本 `py_compile` 通过。
- 重复运行对比工具后 `baseline/evaluation_output/result/case1.png` SHA-256 未变化；`git -C HYPIR diff --name-only` 为空，HYPIR 源码和推理逻辑未改动。
- 按用户请求将验证集 `case1_lq`--`case5_lq` 全部复制到 `baseline/input/`，使用官方 `HYPIR/test.py` 完成 5 张 baseline 推理；命令总耗时约 348.9 秒，模型加载 11.59 秒，无 OOM/异常退出。
- 输出已生成于 `baseline/output/result/`：`case1_lq.png`、`case2_lq.png`、`case3_lq.png`、`case4_lq.png`、`case5_lq.png`；均为 RGB、uint8、finite，且与输入保持原分辨率：case1/3/4/5 为 4096x3072，case2 为 3072x4096。
- 已为 5 个验证 case 生成三联对比图 `baseline/comparison/case*_lq_compare.png`，面板顺序为 Input/LQ、HYPIR Output、Ground Truth。
- 本次验证集指标（LQ → HYPIR）：case1 PSNR 32.0288→29.3353 / SSIM 0.94845→0.84663；case2 28.1894→24.2140 / 0.83144→0.73105；case3 35.6221→28.8654 / 0.93496→0.81210；case4 18.0562→16.3524 / 0.30932→0.25760；case5 26.2756→23.8758 / 0.86432→0.77725。默认 HYPIR baseline 在 5 组上均低于原始 LQ，当前仅作跑通和风险侦察，不作为提交参数结论。
- 新增独立指标脚本 `baseline/evaluate_metrics.py`，仅使用 Pillow、NumPy、scikit-image、LPIPS-Alex 和 PyTorch，不导入 HYPIR；计算前严格检查 LQ/GT/Output 的 RGB 三通道和完全相同尺寸，尺寸不一致直接报错，不静默 resize。
- 指标脚本按 `case1_lq`/`case1_gt`/`case1_lq.png` 的公共 case key 自动匹配，输出每 case 的 PSNR、SSIM、LPIPS-Alex、三项 delta、Improved/Degraded 结论及 Average 行到 `baseline/evaluation_metrics.csv`。
- 本次 5 张验证图评估完成（CUDA，约 127.4 秒）：平均 LQ→HYPIR 为 PSNR 28.0344→24.5286（Δ-3.5058）、SSIM 0.77770→0.68493（Δ-0.09277）、LPIPS 0.35653→0.46100（Δ+0.10447）；PSNR/SSIM/LPIPS 平均均为 Degraded。

## model_t 单变量 sweep（2026-09-01）
- 固定当前 `sd-research/stable-diffusion-2-1-base`、`coeff_t=200`、seed 231、LoRA rank 256、patch/stride 512/256、upscale 1、empty captioner，其余参数与 baseline 一致。
- 完成 `model_t=200/150/100/75/50` 五组，每组均处理 evaluation case1--case5，输出、三联对比图、独立指标 CSV、参数记录和总耗时均保存于 `baseline/experiments/model_t_<value>/`。
- 平均指标随 `model_t` 降低而恶化；本次五个候选中 `model_t=200` 的平均 PSNR/SSIM/LPIPS 均最好。未观察到文字或钟表在低 `model_t` 下更稳定；低值组视觉上更易出现纹理/边缘重绘风险。
- 汇总 CSV：`baseline/experiments/model_t_sweep_summary.csv`；实验报告：`baseline/experiments/model_t_sweep_report.md`。
- 原始 `baseline/output/`、`baseline/comparison/`、`baseline/evaluation_metrics.csv` 未覆盖；HYPIR tracked source 无 diff。
- 用户复核并经原图检查确认：case3 鸟的边缘轮廓与滩涂/水面纹理变化不自然；case4 baseline 出现 LQ/GT 中不存在的鹦鹉/蜥蜴样对象、疑似动物头部和脸样结构，已记录为对象级 hallucination，而非普通叶片锐化。

## coeff_t 单变量 sweep（2026-09-01）
- 核对源码：`HYPIR/test.py` 的 `--coeff_t` 传入 `SD2Enhancer`；`HYPIR/HYPIR/enhancer/sd2.py` 在 `scheduler.step(eps, self.coeff_t, z_in)` 中使用它，`model_t` 独立作为 UNet timestep。
- 新增独立编排脚本 `baseline/experiments/run_coeff_t_sweep.ps1`，固定 model_t=200、seed=231、空 prompt、LoRA rank/modules、patch/stride=512/256、upscale=1 等条件，候选 `coeff_t=200/150/100/75/50`。
- 五组推理全部成功，耗时分别为 316.423/350.165/369.286/355.261/342.774 秒；每组均生成 5 张 RGB 同尺寸输出、5 张 LQ/Output/GT 三联图、独立 metrics CSV、metadata 和 inference.log。
- 使用一次 LPIPS-Alex 模型完成五组评测；新增 `build_coeff_t_artifacts.py` 与 `build_coeff_t_summary.py`，汇总写入 `baseline/experiments/coeff_t_sweep_summary.csv`。
- 平均指标随 coeff_t 降低而改善：200 为 PSNR 24.528587 / SSIM 0.684926 / LPIPS 0.461005；50 为 28.257533 / 0.776935 / 0.338965。`coeff_t=50` 相对 LQ 的 ΔPSNR +0.223113、ΔSSIM -0.000764、ΔLPIPS -0.017567。
- 逐图检查 baseline 与 coeff_t=50 的五组三联图：文字带状伪影、书脊细字和钟表过度锐化减弱；鸟类水面高频重绘减弱；绿植的对象样高频结构不再明显但输出更平滑。缩略图不足以确认逐字符/逐数字/指针端点一致性，报告不作确定语义断言。
- 报告 `baseline/experiments/coeff_t_sweep_report.md` 已完成；按用户要求本轮结束后暂停，不开始下一轮实验。

## Phase 2 confidence_fusion_v1（2026-09-01）
- 已完整阅读 `PROJECT_HANDOFF.md`、`progress.md`、`findings.md`、`task_plan.md`、`赛题.txt` 及 baseline/experiments 中两份 sweep 汇总/报告。
- 核验 `coeff_t_200/experiment_metadata.md` 与 `coeff_t_50/experiment_metadata.md`：两组均为 `model_t=200`、seed=231、当前 `sd-research/stable-diffusion-2-1-base` 镜像、patch/stride=512/256、upscale=1、空 prompt，五张 PNG 均存在且尺寸与 LQ 匹配；决定直接复用。
- `git -C HYPIR status --short --untracked-files=all` 仅显示模型/权重未跟踪文件，没有 tracked source 修改。
- 已在 `task_plan.md` 增加 Phase 2 计划；后续所有产物限定在 `baseline/experiments/confidence_fusion_v1/`。

- 新增独立脚本 `run_confidence_fusion.py`、`evaluate_confidence_fusion.py`、`region_change_analysis.py`；五张图生成 confidence map/masks、三种 fusion、七面板 comparison 与 error-map。
- 首次评测尝试因脚本中文字路径/ Pillow 读取处理错误失败；修复为显式 RGB 转换并动态定位 GT 目录。第二次因 120 秒工具时限超时；第三次使用长时限成功完成，未重复推理。
- 整图指标写入 `baseline/experiments/confidence_fusion_v1/evaluation_metrics.csv`，区域 GT 误差写入 `region_metrics.csv`，相对 LQ 的区域改写写入 `region_changes_vs_lq.csv`。
- 平均结果：LQ 28.0344/0.777699/0.356532；HYPIR-200 24.5286/0.684926/0.461005；HYPIR-50 28.2575/0.776935/0.338965；fixed 50/50 26.8586/0.747600/0.365744；confidence 200/50 27.2912/0.766543/0.332414；confidence 200/LQ 28.0487/0.776696/0.317280（PSNR/SSIM/LPIPS）。
- 平均相对 LQ 改写量：HYPIR-200 high/low L1=12.333/3.491，confidence 200/50=6.949/1.378，confidence 200/LQ=4.588/0.448；说明空间加权确实降低 low-confidence 改写，但不证明真实细节恢复。
- 已写 `experiment_metadata.md`、`README.md`、`inference.log`、`confidence_fusion_v1_report.md`；报告结论为 Weak success，Q10=WEAK EVIDENCE。
- 最终 sanity check 通过：15 个 fusion PNG 与 5 个 confidence map 均 RGB/uint8/finite/尺寸匹配，10 个 mask 为 L 模式且尺寸匹配；3 个独立脚本 `py_compile` 通过。
- `git -C HYPIR diff --name-only` 为空；status 仍只有既有未跟踪 base model/LoRA 文件，完整记录见 `confidence_fusion_v1/hypir_git_status.txt`。既有 baseline CSV/输出 SHA 已复核并未改写。

## 2026-09-02 - Phase 4 structure-anchored local restoration v1
- Read the pasted Structure-Anchored Local Restoration v1 specification and re-read the Phase 3 diagnosis and Phase 2 confidence-fusion evidence.
- Added test-first coverage in `tests/test_structure_local_restoration.py`; the new tests initially failed with `ModuleNotFoundError`, then passed after implementation. Full suite now has 17 tests passing.
- Added independent `baseline/experiments/structure_local_restoration.py`. It computes quarter-resolution Sobel edge, local-variance texture, and inverse-Laplacian blur proxies from LQ only, then applies three fixed formulas with weights clipped to [0, 0.60]. GT is only used after fusion for evaluation.
- Reused the five existing coeff_t=200 and coeff_t=50 outputs; no new diffusion inference and no HYPIR source edits. Added read-only comparison to existing `confidence_200_LQ` when present.
- Generated `baseline/experiments/structure_local_restoration_v1/`: 30 fusion PNGs (3 strategies x 2 sources x 5 cases), 30 weight maps (color + grayscale), 15 comparison panels, metrics, local analysis, metadata, README, and report.
- Metrics use PSNR/SSIM at native dimensions and LPIPS-Alex after a uniform max-side 1024 resize for every method. Average PSNR: LQ 28.0344, HYPIR-50 28.2575, HYPIR-200 24.5286, confidence_200_LQ 28.0487, texture_selective_h200 28.4803 dB.
- LQ-quantile local evidence: strong-edge H200 error/change 20.9474/19.9579 L1; structure_guard_h200 15.4210/3.3265; texture_selective_h200 15.2215/4.5728. Text/clock semantics are not claimed from these image-space regions.
- Health audit passed: all fusion RGB PNGs and grayscale weight maps match LQ dimensions; 50 per-case metric rows + 10 Average rows have LPIPS; HYPIR tracked source remains unchanged. Runtime was about 933 seconds, dominated by high-resolution metric computation.

## 2026-09-02 - Phase 5 texture-region HYPIR weight sweep
- Added `tests/test_texture_weight_sweep.py` before implementation; the initial import failed as expected, then all three tests passed after adding the sweep module.
- Added independent `baseline/experiments/texture_weight_sweep.py`. It reuses existing HYPIR-200 PNGs and v1 LQ features; candidates `0.25/0.40/0.55/0.70/0.85/1.00` replace only the fixed `texture >= P80 & edge < P80` region. Strong-edge and all outside-mask weights are copied exactly from `texture_selective` v1.
- First LPIPS run hit the 20-minute tool limit after writing an incomplete `texture_weight_sweep/` directory. The explicit child process was stopped; the incomplete directory was preserved as a run trace. The implementation was optimized to one batched LPIPS call per case and rerun in `texture_weight_sweep_v2/`.
- Completed v2 artifacts: 35 fusion PNGs, 70 weight maps, 5 seven-panel comparisons, `metrics.csv`, `local_analysis.csv`, `experiment_metadata.md`, and `report.md`. A report-only bug involving the baseline empty weight was fixed and the report rebuilt from existing CSVs.
- Average full-image metrics (PSNR/SSIM/LPIPS): v1 baseline `28.480280/0.781421/0.164549`; candidates: `0.25 -> 28.479684/0.781403/0.164545`, `0.40 -> 28.476294/0.781214/0.164317`, `0.55 -> 28.468252/0.780793/0.164004`, `0.70 -> 28.456087/0.780213/0.163652`, `0.85 -> 28.439112/0.779462/0.163265`, `1.00 -> 28.418456/0.778650/0.162872`.
- Average regional GT error (v1 -> candidate 0.25/0.40/0.55/0.70/0.85/1.00): `strong_edge 15.2215 -> 15.2215` for every candidate; `textured_non_edge 8.1972 -> 8.2142/8.2505/8.3922/8.6282/8.9448/9.3239`; `blurred_texture 29.3456 -> 29.2994/29.2481/29.7000/30.6161/32.0222/33.9444`.
- Case3 bird: texture/non-edge GT error rises from `2.8215` baseline to `2.8735/3.1457/3.5038/3.9217/4.3795/4.8569`; full PSNR peaks at baseline `35.6301`, candidate 0.25 is `35.6275` and higher weights decline monotonically.
- Case4 dense foliage: textured-non-edge error is essentially unchanged/slightly worse (`23.9475` baseline; `23.9469/24.0023/24.1689/24.4373/24.8000/25.2500`); blurred-texture error improves only at 0.25/0.40 (`29.2994/29.2481` vs `29.3456`) and worsens from 0.55 onward. Full PSNR declines from `18.0199` baseline to `18.0198/18.0176/18.0135/18.0077/18.0001/17.9907`.
- Case1/case2/case5 do not show a consistent benefit from increased texture allowance; full-image PSNR best weights are case1 `0.25`, case2 `0.25`, case3 `0.25`, case4 `0.25`, case5 `0.55`.
- Real-case invariance audit checked 30 candidate maps with no mismatch on strong-edge or outside-mask pixels. All 35 fusion outputs are RGB, finite, 0--255, and dimension-matched. Full test suite: 20 tests passed. HYPIR source was not imported or modified.
## 2026-09-01 - Phase 3 structure diagnosis started
- Read the user-provided diagnostic specification from the pasted attachment.
- Confirmed `coeff_t_200` and `coeff_t_50` each contain five native-resolution outputs, metadata, metrics, and comparisons.
- Confirmed validation inputs/GT are RGB and dimension-matched; NumPy/SciPy/scikit-image/Pillow are available in `.conda`.
- No new diffusion inference will be run; all new artifacts are scoped to `baseline/experiments/structure_diagnosis/`.

## 2026-09-01 - Phase 3 structure diagnosis completed
- Added `baseline/experiments/structure_diagnosis/analyze_structure.py` and `tests/test_structure_diagnosis.py` using test-first development; full suite: 13 tests passed.
- Reused the existing native-resolution coeff_t=200/50 outputs and analyzed 3,565 patches (256x256, stride 128) across case1-case5.
- Wrote patch metrics/summary/quantiles, LQ-only feature correlations, per-case maps, top/bottom 10% patch CSVs, coordinate-audited crop sheets, metadata, and report under the diagnosis directory.
- Final evidence: H200 15 Type-A useful patches, 865 Type-B/Type-D harmful-or-mismatch candidates, 10 Type-C unchanged-but-needed; H50 mean generated-change L1 2.7131 vs H200 7.9598.
- Visual crop review covered text (case1/2), bird (case3), ambiguous foliage (case4), and clock geometry (case5) without OCR or semantic claims.
- Health checks passed: finite CSV values, RGB/nonempty case PNGs, py_compile, existing test suite, and empty tracked HYPIR diff. No existing baseline output/comparison/metrics files were overwritten.
# 2026-09-02 — Structure-Anchored Residual Fusion v2

- Restored the existing plan and identified HYPIR-200/HYPIR-50 and texture-selective source directories.
- Added RED tests in `tests/test_structure_anchored_residual_fusion_v2.py`; initial run failed as expected because the production module did not exist.
- Implemented `baseline/experiments/structure_anchored_residual_fusion_v2.py`: fixed LQ Sobel/Canny edge bands, residual novelty suppression, gradient alignment allowance, exact `F = LQ + gate * (H200 - LQ)` fusion, regional metrics, LPIPS, panels, metadata, and summary.
- GREEN: v2 tests pass; full suite passes (24 tests). A synthetic `inf-inf` metric warning was removed with explicit metric-delta handling.
- Next: run the real five-case offline experiment and inspect metrics/artifacts.
- Real run completed with LPIPS: 5 cases, 25 per-case method rows plus 5 Average rows in `baseline/experiments/structure_anchored_residual_fusion_v2/metrics.csv`.
- Average PSNR: LQ 28.034420, H50 28.257533, H200 24.528587, current texture-selective H200 28.480280, v2 28.158209. Average SSIM: 0.777699, 0.776935, 0.684926, 0.781421, 0.779548 respectively. Average LPIPS-Alex: 0.204315, 0.162039, 0.159666, 0.164546, 0.195091.
- Regional evidence: v2 strong-edge mean change_L1=0.481077 versus H50=8.794542 and texture-selective H200=5.551991, but v2 textured_non_edge error_to_GT_L1=10.156728 versus 10.027113/9.952752 and blurred_texture=17.307310 versus 17.337979/17.397288.
- Gate maps, 5 comparison panels, and v2 outputs are complete; fusion/gate PNGs are RGB, finite, and original case dimensions. Old source output timestamps remain unchanged. Visual checks covered case1 text, case2 spine, case3 bird, case4 foliage, and case5 clock.
- Decision: stop this v2 direction under the fixed rule because v2 is below both H50 and current texture-selective H200 in average PSNR; no further sweep or training is justified.

# 2026-09-02 — Adaptive H50/H200 Fusion v1

- Added RED tests in `tests/test_adaptive_fusion_v1.py`; the initial import failed because the new module did not exist, then all four tests passed after implementation.
- Implemented `baseline/experiments/adaptive_fusion_v1/experiment.py` and package exports. Alpha is a fixed LQ-only continuous map using Sobel edge, local variance texture, inverse-Laplacian blur proxy, and a top-20-percent strong-edge penalty, clipped to `[0, 0.30]`.
- Reused existing LQ, HYPIR-50, HYPIR-200, and `texture_selective_h200` images. No HYPIR/SD inference, training, LoRA, or parameter sweep was run.
- Completed five-case outputs with LPIPS-Alex using the prior max-side 1024 protocol. Average metrics: LQ `28.034420/0.777699/0.204318`; H50 `28.257533/0.776935/0.162039`; H200 `24.528587/0.684926/0.159664`; texture-selective `28.480280/0.781421/0.164548`; adaptive `28.243879/0.776406/0.157813` (PSNR/SSIM/LPIPS).
- Per-case PSNR versus texture-selective: case1 `-0.011137`, case2 `-0.262750`, case3 `-0.998636`, case4 `+0.050895`, case5 `+0.039621` dB. Adaptive helps foliage/clock slightly but harms text/spine/bird cases.
- Regional means are recorded in `region_metrics.csv`; strong-edge alpha is zero by construction, textured non-edge mean alpha is about `0.1092`, and blurred-texture mean alpha is about `0.1333` where present.
- Output health passed for 20 PNGs (RGB/finite/range), visual checks passed for alpha and case1 six-panel layout, and the full 28-test suite passed.
- Decision: adaptive v1 does not clearly exceed the supplied current best on PSNR/SSIM, so stop this adaptive fusion direction and do not sweep further.

## 2026-09-02 - Phase 8 Scenario Routing v1

- Read existing `structure_local_restoration_v1/metrics.csv` and `local_analysis.csv`; reused LQ, coeff_t=50, coeff_t=200, and texture-selective H200 PNGs without diffusion inference.
- Added test-first `tests/test_scenario_routing_v1.py`; initial import failed as expected, then the four contract tests passed. Full suite: 32 tests passed.
- Added independent `baseline/experiments/scenario_routing_v1.py` with fixed LQ-only descriptors (edge density, local variance, blur proxy, high-frequency energy, gradient mean/std) and globally fixed scenario mapping. No GT, case name, semantic label, classifier, training, or sweep enters routing.
- Generated `baseline/experiments/scenario_routing_v1/`: `per_case_method_matrix.csv` (20 case/method rows with global metrics, overall change_L1, and three region error/change pairs), `routing_decision.csv` (5 decisions), `metrics.csv` (25 per-case rows + 5 Average rows), five routing visualization panels, and `summary.md`.
- Actual decisions: case1/case3/case4 -> HYPIR-200; case2/case5 -> LQ. Average routing metrics: PSNR 25.803636, SSIM 0.722419, LPIPS-Alex 0.169598 versus texture-selective H200 28.480280, 0.781421, 0.164546.
- Per-case matrix confirms changing optima: texture-selective H200 is composite best for case1/case2/case3; HYPIR-50 is composite and PSNR/SSIM best for case4/case5; raw HYPIR-200 has no PSNR/SSIM win and only LPIPS advantage on case4.
- Regional means: HYPIR-200 error is highest on strong_edge (20.9474) and blurred_texture (33.9444); texture-selective H200 is lowest on textured_non_edge (8.1972). This supports strategy differences but not a reliable input selector.
- Visual and health audit passed: five RGB nonempty panels, dimensions valid; source experiment directories were read-only. Decision is **B**: routing has no evidence of value; stop routing and do not train/tune further.

## 2026-09-02 - LoRA / Adapter training feasibility audit

- Recovered the completed experiment evidence, including the current texture-selective H200 reference and the failed image-space control-conditioned v1; this audit does not download data, train, or modify HYPIR/experiment code.
- Read the local HYPIR inference/training/degradation paths, SD2.1 U-Net config, and official LoRA checkpoint structure. Verified the installation/runtime versions, bf16 support, 15.920 GiB GPU capacity, and free disk space.
- Used a meta-device U-Net forward to obtain exact additional-residual shapes without loading model weights on GPU. The next action is to write the requested feasibility audit with an evidence-bounded GO/NO-GO recommendation and hard stop gates.
- One combined Python environment probe importing torch, diffusers, accelerate, peft, and transformers exceeded the terminal time limit before producing output. It was replaced by separate successful torch and pip metadata probes; the failed combined probe was not retried.
- Wrote and verified `training_feasibility_audit.md` (17,176 bytes). It recommends a time-boxed GO for a 0.94M spatial residual Adapter, 30k-patch maximum MVP, and explicit Day 1/2/5 NO-GO gates. No training, data download, HYPIR change, or experiment-code change occurred.

## 2026-09-03 - E1 official checkpoint verification started
- Completed local reconnaissance. No project-local restoration checkpoint or inference script was found; BasicSR's installed SwinIR architecture is usable.
- Verified from official SwinIR README and `main_test_swinir.py` that `006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth` is the color JPEG artifact-reduction, JPEG quality 40, upscale=1 model with SwinIR-M configuration and `params` state-dict key.
- Confirmed official tiled inference requires tile size divisible by window size 7; 512 is invalid. Planned tile is 504 with overlap 32, subject to smoke-test validation.
- Next action: download the single official checkpoint, hash it, record metadata, and smoke-test compatibility before any validation case is processed.

## 2026-09-03 - E1 official checkpoint and baseline completed

- Official checkpoint downloaded and verified: `006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth`, 102,873,665 bytes, SHA-256 `265c18d8809aaca0cd97a6283bee0ed1883ab88395e456381264cac2bb7b5867`; strict 544-tensor load passed.
- Corrected E1 wrapper for direct execution and official SwinIR `[0,1]` preprocessing; six E1 contract tests pass.
- Ran exactly five validation cases with fixed tile 504 / overlap 32 / window 7. All outputs are RGB PNGs at original dimensions and are saved in the independent E1 output directory.
- E1 average metrics: Fidelity PSNR `28.028629`, SSIM `0.780163`, LPIPS-Alex `0.216824`; HYPIR-50 PSNR `28.257533`, SSIM `0.776935`, LPIPS-Alex `0.162039`.
- Runtime/resource records: model load `0.547554 s`; average SwinIR forward `182.353154 s/case`; end-to-end wall time `~1112.3 s`; peak allocated/reserved `3.335263/4.683594 GiB`.
- Wrote `reports/E1_FIDELITY_BASELINE.md`. E1 is complete and awaits ChatGPT's E2 decision; no E2 work was performed.

## 2026-09-03 - E2 Global Alpha Blend completed

- Added `baseline/experiments/e2_global_blend.py` and test-first `tests/test_e2_global_blend.py`. The first import test failed as expected; five final E2 contract tests pass.
- Confirmed all HYPIR-50/Fidelity/GT triplets are RGB uint8, within range, and dimension-matched. Ran only offline pixel blending for the six authorized alphas, creating 30 PNGs in the independent E2 directory.
- Initial output generation plus evaluation took `706.019 s`. Recomputed metrics only from those saved PNGs with E0-style per-case batched LPIPS in `577.980 s`; no HYPIR/SwinIR model was rerun.
- Output health and endpoint identity checks passed. Average best alpha: PSNR `0.6` (`28.507181`), SSIM `0.4` (`0.784890`), LPIPS `1.0` (`0.162039`).
- Wrote `reports/E2_GLOBAL_BLEND.md`; no E3 work was performed.

## 2026-09-03 - E3-A Texture-only Adaptive Alpha started
- Restored planning context and prior E0/E1/E2 artifacts. Confirmed all five expected HYPIR-50, SwinIR, LQ, and GT input files exist.
- Confirmed the workspace has no Git metadata; documented this single failed diagnostic and will not repeat it.
- Next: create RED contract tests for exactly the specified LQ-only alpha-map experiment.
- RED completed: `tests.test_e3_texture_adaptive` failed as expected with a missing E3-A module.
- First GREEN run exposed a report-only Python syntax error before experiment logic could execute; root cause is a nested f-string expression with an escaped quote. The next edit isolates correlation formatting in a helper.
- GREEN completed: all four E3-A contract tests pass. The module uses only LQ Sobel magnitude, local variance, and Gaussian blur residual features with fixed 1st/99th percentile normalization, equal weights, and sigma=8 smoothing. It reuses E2's metric function unchanged.
- Post-run consistency audit found the E3 average rows used full-precision case values rather than E2's six-decimal row convention. A RED regression test now fails on the missing `_average_for_csv` helper; the next edit will add that shared convention without rerunning inference.
- Added `_average_for_csv` with a RED/GREEN regression cycle and rebuilt summaries from saved per-case rows. A read-only CUDA LPIPS reproducibility probe timed out at 120 seconds; it was not repeated. The pixel-identical global case3 row is aligned to the established E2 six-decimal reference (`0.074909`), with derived deltas updated only in the saved summaries.
- The first final audit attempt failed because `case*_texture.png` also matched alpha-map filenames. The audit is being rerun with explicit filename lists; this was a diagnostic-script error only.

## 2026-09-04 - E3-A Texture-only Adaptive Alpha completed
- Added `baseline/experiments/e3_texture_adaptive.py` and `tests/test_e3_texture_adaptive.py` using test-first development. Texture uses LQ-only Sobel magnitude, 9x9 local variance, and absolute Gaussian blur residual; each uses fixed 1st/99th percentile normalization, equal weights, then fixed sigma=8 smoothing.
- Ran exactly one offline E3-A pass for case1-case5 in 359.3 s using existing HYPIR-50 and SwinIR outputs. No HYPIR/SwinIR inference, training, detectors, output-derived features, GT-driven alpha, or parameter sweep occurred.
- Wrote 15 RGB uint8 output PNGs across exactly `global_alpha_0.6`, `adaptive_texture`, and `adaptive_reverse`; wrote 15 native-resolution RGB texture/alpha maps, `metrics.csv`, and `reports/E3_TEXTURE_ADAPTIVE.md`.
- Average metrics (PSNR/SSIM/LPIPS-Alex): global `28.507181/0.783521/0.177557`; adaptive texture `28.475310/0.785022/0.191381`; adaptive reverse `28.499589/0.783138/0.177883`.
- `adaptive_texture` does not beat global alpha 0.6 on the composite metrics: PSNR -0.031871 dB and LPIPS +0.013824 despite SSIM +0.001501. It improves PSNR/SSIM only on cases1 and 3; cases2/4/5 lose PSNR.
- `adaptive_reverse` behaves differently in spatial allocation but remains close to global and is slightly worse on all three average metrics (PSNR -0.007592, SSIM -0.000383, LPIPS +0.000326).
- Visual map review and final health audit passed; global output is pixel-identical to E2 alpha 0.6. Full existing unit suite: 53 tests passed. E3-A is complete; no further experiment is authorized.

## 2026-09-04 - Team handoff packaging started
- Current workspace is approximately 16.5 GB. The dominant removable components are `.conda` (~5.99 GB), HYPIR model/LoRA weights (~6.20 GB), generated PNG experiment artifacts (~5.10 GB), and the 100-image test set (~216 MB).
- Planned package keeps source code, tests, reports, metrics, reproducibility metadata, five validation LQ/GT pairs, and downscaled representative comparison previews. It excludes local environments, caches, model checkpoints, HYPIR Git metadata, the full test set, and bulk intermediate images.

## 2026-09-04 - Team handoff packaging completed
- Built `team_handoff/` with 257 files: source and tests, reports and research notes, non-image metrics/metadata, five validation LQ/GT pairs, and six downscaled comparison previews.
- Handoff tree size is 22,275,790 bytes (21.24 MiB); outer archive `CSIG_team_handoff_2026-09-04.zip` is 19,162,580 bytes before the final documentation sync.
- ZIP entry audit passed: no `.conda`, `.cache`, model/weight directories, HYPIR `.git`, full test set, or bulk PNG outputs are present. Original artifacts were not deleted or modified.

## 2026-09-09 — HYPIR-200 error decomposition E0–E4
- Offline diagnosis in `baseline/experiments/error_decomposition_v1/`. No HYPIR source edits.
- E1: case4 high-demand 8 patches A=7/B=1/C=0; fish head at mid04. Case3 is C-dominant (water grain), not the same disease.
- E2: case4 1/16 still the same tree; 1/8 missing yellow flowers; local 256 already wrong leaf type.
- E3: global LQ–H200 blend peaks at α=0.2, PSNR 28.45 vs texture_selective 28.48. Stop global fusion as the final method.
- E4: seeds 17/89/401 vs 231; pairwise PSNR 39.5–39.9 dB; same serrated leaves, catkins, and fish. Deterministic mapping bias.
- Replaced DAS-V1 with `CURRENT_PLAN.md`. Next: Exp-F1 residual-confidence fusion only.

## 2026-09-10 — Non-overlap 256 tile metrics
- `compute_patch_metrics.py`: 192 tiles/case, 960 total. Native PSNR/SSIM; LPIPS-Alex on 256 crops (not 1024 protocol).
- Block-mean ΔPSNR H200 vs LQ: case1 −4.66, case2 −3.83, case3 −8.93, case4 −1.64, case5 −2.52. Case1/3 block means worse than full-image because flat tiles get dirty.
- Case4 LPIPS better on all three gradient strata; ΔPSNR almost uniform. Case3 low/mid water tiles are the worst PSNR hits.
- CSVs: `patch_metrics_full.csv`, `patch_metrics_summary.csv`, `e1_patch_metrics.csv`. Synced into `report.md` and `CURRENT_PLAN.md`.

## 2026-09-10 — HYPIR fusion v1
- Independent offline module `baseline/experiments/hypir_fusion_v1/`. 10 unit tests pass. HYPIR tracked source unchanged.
- Reused `coeff_t_50` / `coeff_t_200` PNGs. Scheme A: LQ + α·mask·(H200−LQ) on Y, LQ chroma. Scheme B uses H50 as base.
- Manual scene α: text 0.25, book 0.30, bird 0.12, plant 0.08, clock 0.30. Structure mask = 1/4 Sobel cosine × magnitude similarity.
- Average PSNR/SSIM/LPIPS_1024: LQ 28.034/0.778/0.204; H50 28.258/0.777/0.162; H200 24.529/0.685/0.160; fusion_A **28.460/0.782/0.189**; fusion_B 28.256/0.779/0.156; texture_selective 28.480/0.781/0.165.
- Visual: Fusion does not detect/delete the fish head; plant α=0.08 attenuates the H200 residual so the eye/contour collapse back to a pink blur. Sobel mask suppresses new strong edges (serrated leaves) but not smooth-region semantics. Case3 water grain fades for the same reason (bird α=0.12). Text/clock structure held.
- Did not beat the texture_selective +0.05 dB gate. Report: `baseline/experiments/hypir_fusion_v1/results/fusion_v1/report.md`.

## 2026-09-11 — HYPIR fusion v2 residual confidence (F1)
- Do not overwrite fusion_v1. New package: `baseline/experiments/hypir_fusion_v2/`.
- Keep scene alpha, YCbCr (Cb/Cr locked to LQ), H50/H200 bases, metrics CSV, visualization.
- Mask groups: A = original fusion_v1 (`M_struct`); B = `M_struct` (same formula as A); C = `M_struct * conf`; D = `M_struct * conf^2`; extra gamma=0.5 as requested.
- `conf = 1 - percentile_norm(GaussianBlur(mean_c |H200-LQ|, σ=8), 1–99)`.
- Scope: validation case1–case5 only. No HYPIR inference, training, LoRA, classifier, or depth model.
- 8 unit tests + 10 v1 tests pass. fusion_A PSNR 28.460407 matches v1 exactly. HYPIR git: only untracked `models/` and `weights/`.
- Average PSNR: A/B fusion_A 28.460; C fusion_A_conf 28.198; D fusion_A_conf2 28.109; best F1 is fusion_B_conf2 28.368. Anchor texture_selective 28.480. Did not beat 28.48.
- Visual: fish-head conf map goes dark inside the head, but A/C/D outputs stay the same pink blur (α=0.08). Water conf≈0.91 so F1 does not target ripples. Text/clock not redrawn; C/D softer than A.
- Report: `baseline/experiments/hypir_fusion_v2/results/fusion_v2/report.md`. Stop stacking output-space masks.

## 2026-09-11 — HYPIR process-level control audit
- Do not stack masks. Do not retrain. Do not re-run completed coeff_t/model_t inference.
- First: read `HYPIR/HYPIR/enhancer/sd2.py` `forward_generator` and `DDPMScheduler.step`.
- Code fact: inference is a single UNet epsilon pass + `pred_original_sample`. No sampling loop, no CFG, no noise injection.
- 8 unit tests pass. Offline eval reused coeff_t_50/75/100/150/200 PNGs. `ran_inference=False`.
- Average PSNR: t50 28.258, t75 27.958, t100 27.505, t150 26.213, t200 24.529. Anchor 28.480. fusion_v1_A 28.460. None beat the anchor.
- Visual: fish-head develops with coeff_t (eye at 75, full head at 200). Wrong catkin/burr at yellow-flower crop. Water grain increases with t.
- Decision: stop process control; LoRA/Adapter is the next stage if authorized. Report written.

## 2026-09-11 — LoRA feasibility audit
- No training, no download, no HYPIR edits.
- Inventory: CSIG val 5 paired 4K, test 100 LQ-only, no parquet/LSDIR/DF2K on disk.
- Loader is GT → RealESRGAN synthetic LQ (`stage2_scale: 4`). Official trainer gaussian-inits LoRA, does not load `HYPIR_sd2.pth`.
- Decision: **NO-GO LoRA**. Keep `texture_selective_h200`.
- Write-up: `baseline/experiments/hypir_lora_feasibility_audit/`.

## 2026-09-11 — fusion_v3 multi-band
- Laplacian L0 high / L1+L2 mid / G3 low. Y only, LQ chroma. 6 unit tests pass. No HYPIR edits.
- Average PSNR: B 26.604, C 26.579, D01 26.493, D02 26.389 vs texture 28.480.
- Case4 fish: high band contains eye and scales. Fusion still a fish head.
- **NO-GO multi-band.** Freeze texture_selective_h200. Stop output-space work.
