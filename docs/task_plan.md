# HYPIR 官方 Baseline 跑通计划

## 目标
在当前 Windows + RTX 5080 Laptop + 项目 `.conda` 环境中，完整运行 HYPIR 官方 inference 流程，并对一张真实 evaluation/validation 图片生成同分辨率增强结果；不修改模型结构、不训练、不引入自定义控制模块。

## 阶段
- [x] 阶段 1：盘点工作区、Conda/Python/PyTorch/CUDA/GPU 与现有数据
- [x] 阶段 2：获取 HYPIR 官方仓库并阅读 README、requirements、inference、configs、checkpoint 与 tiled/patch 代码
- [x] 阶段 3：按官方 requirements 安装缺失依赖，记录 Windows/RTX 5080 兼容性问题
- [x] 阶段 4：下载并校验 pretrained checkpoint/base model（LoRA 已完成；官方 repo 仍返回 404，使用结构匹配的公开 SD2.1 Diffusers 镜像）
- [x] 阶段 5：选取一张本地 evaluation/validation 图片，建立最小 input/output 目录
- [x] 阶段 6：运行官方 inference；使用官方 patch/tile 参数保持原分辨率
- [x] 阶段 7：校验输出文件、尺寸、颜色、异常值、耗时与显存；有 GT 时计算 PSNR/SSIM/LPIPS
- [x] 阶段 8：整理 HYPIR 完整数据流、生成强度/内容保持/幻觉风险代码位置与后续研究方向
- [x] 阶段 9：新增独立图像对比工具；支持 evaluation 二联图、验证集 LQ/Output/GT 三联图和 case1--case100 批量匹配
- [x] 阶段 10：使用官方 HYPIR baseline 完成验证集 case1--case5 全部推理，并生成五组三联对比图
- [x] 阶段 11：新增独立 evaluation 指标脚本，完成 5 组 PSNR/SSIM/LPIPS-Alex、delta、结论和 CSV 汇总

## 阶段 12：coeff_t 单变量 sweep
- [x] 核对 `coeff_t` 参数入口、传递路径与候选范围
- [x] 在独立目录运行 coeff_t=200/150/100/75/50，固定 model_t=200 及其余变量
- [x] 为每组生成五张输出、三联图、metrics CSV、参数记录和耗时
- [x] 汇总逐 case/Average 指标，逐张检查三联图并形成报告

## 下一步
完成 coeff_t sweep 结论后暂停，不直接开始下一轮实验；根据指标与视觉证据提出下一步变量建议。

## Phase 2：Confidence-guided HYPIR fusion prototype
- [x] 核对并复用 coeff_t=200/50 的可验证输出，建立独立产物目录
- [x] 从 LQ 计算并可视化平滑 structural reliability confidence map 与 high/low masks
- [x] 生成 fixed 50/50、confidence 200/50、confidence 200/LQ 融合输出
- [x] 生成七面板 comparison、误差证据与输出健康检查
- [x] 计算整图 PSNR/SSIM/LPIPS、delta 与 high/low 区域 L1/L2/gradient 分析
- [x] 写 metadata、README、最终报告，回答 Q1-Q10 并检查 HYPIR git status

## Next Step
完成最终 sanity checks，记录 HYPIR git status 与 Phase 2 结果，然后向用户汇报 Weak success 结论。

## Phase 3: Structure-Anchored HYPIR local diagnosis
- [x] Verify and reuse the completed coeff_t=200/50 outputs without new diffusion inference
- [x] Add failing tests for native-resolution patch metrics, LQ-only features, quantile classifications, and artifact generation
- [x] Implement an independent diagnosis script under `baseline/experiments/structure_diagnosis/`
- [x] Run case1-case5 analysis with 256 patch / 128 stride, save maps, crops, CSVs, metadata, and report
- [x] Audit outputs, source immutability, and answer Q1-Q10 from GT-aligned evidence

## Next Step
Report the completed diagnosis artifacts and the evidence-based next-step recommendation.

## Phase 4: Structure-Anchored Local Restoration v1
- [x] Read the user specification and reuse existing HYPIR-50/HYPIR-200 outputs without diffusion inference
- [x] Add failing tests for LQ-only feature maps, bounded edge protection, convex fusion, and artifact contracts
- [x] Implement three fixed interpretable strategies: structure_guard, texture_selective, blurred_texture
- [x] Run all three strategies with HYPIR-50 and HYPIR-200 on validation case1-case5
- [x] Write weight maps, 30 fusion images, comparison panels, PSNR/SSIM/LPIPS CSV, local change/error CSV, metadata, README, and report
- [x] Audit dimensions, RGB/L mode, finite outputs, metric completeness, tests, and HYPIR source immutability

## Next Step
Use the v1 report as the gate for a small held-out/seed local-control check; do not start a broad sweep or LoRA unless edge protection and texture gains reproduce.

## Phase 5: Texture-region HYPIR weight sweep
- [x] Add failing tests for the six fixed texture weights and strong-edge invariance
- [x] Implement an independent offline sweep that reuses HYPIR-200 outputs and the v1 LQ feature maps
- [x] Run weights 0.25/0.40/0.55/0.70/0.85/1.00 on validation case1-case5 with full and regional metrics
- [x] Generate per-case outputs, weight maps, comparison panels, metadata, CSVs, and report
- [x] Audit output health, exact strong-edge preservation, HYPIR source immutability, and determine whether GT error falls as texture generation rises

## Next Step
Decision complete: stop the texture-weight direction; do not launch another broad search or modify HYPIR.

## Phase 6: Structure-Anchored Residual Fusion v2
- [x] Add failing tests for LQ-only edge bands, residual edge novelty/orientation gating, bounded fusion, and independent artifact contracts
- [x] Implement one fixed, simple v2 gate under `baseline/experiments/structure_anchored_residual_fusion_v2/` with no diffusion inference or GT-driven weights
- [x] Run case1-case5 using existing LQ/HYPIR-50/HYPIR-200/GT and write metrics, regional change/error CSVs, visual panels, metadata, and summary
- [x] Verify output health, source immutability, exact baseline coverage, and compare v2 against H50 and `texture_selective_h200`
- [x] Stop the direction if v2 does not clearly improve over H50 and the current texture-selective baseline; record the evidence-based conclusion

## Next Step
No further v2 work; report the completed artifacts and the evidence-based stop conclusion.

## Phase 7: Adaptive H50/H200 Fusion v1
- [x] Add failing tests for LQ-only alpha features, bounded strong-edge protection, convex H50/H200 fusion, and artifact contracts
- [x] Implement one fixed alpha map and independent offline experiment under `baseline/experiments/adaptive_fusion_v1/`
- [x] Run case1-case5 using existing LQ/HYPIR-50/HYPIR-200/texture-selective outputs; do not run diffusion, training, or sweeps
- [x] Write metrics.csv, region_metrics.csv, alpha maps, six-panel comparisons, and summary.md with the explicit stop decision
- [x] Audit output health, metric comparability, source immutability, and whether adaptive fusion exceeds `texture_selective_h200`

## Next Step
Adaptive v1 did not clearly exceed the current texture-selective baseline; stop adaptive fusion and do not launch another sweep.

## Phase 9: LoRA / Adapter training feasibility audit
- [x] Recover the completed baseline, fusion, routing, and control-conditioned evidence
- [x] Inspect the local HYPIR SD2.1 loading path, LoRA target layers, inference flow, and feasible adapter injection sites
- [x] Inspect the installed CUDA/PyTorch runtime and bound a 16 GB training configuration from measured inference use plus architecture assumptions
- [x] Design and compare minimal synthetic-data, adapter, objective, and validation protocols without downloading data or training
- [x] Write `training_feasibility_audit.md` with a time-boxed GO / NO-GO decision; do not modify HYPIR or experiment code

## Next Step
Training feasibility audit complete. Do not start implementation until the user authorizes the time-boxed MVP.

## Phase 8: Scenario Routing v1
- [x] Inventory existing per-case outputs and metric sources for LQ/H50/H200/texture_selective_h200
- [x] Build complete per-case x method matrix with global and regional metrics; identify per-case winners
- [x] Implement an LQ-only, feature-driven routing rule without GT or case-specific hardcoding
- [x] Run routing v1 offline, save routing_decision.csv, metrics.csv, summary.md, and routing_visualization/
- [x] Audit validation-overfitting risk and issue explicit A/B decision: proceed to architecture design or stop routing

## Next Step
Scenario Routing v1 failed the average PSNR/SSIM/LPIPS gate; stop routing and report conclusion B. Do not train a classifier, add diffusion inference, or tune rules on the five validation cases.

## Phase 10: HYPIR fusion v2 residual confidence mask (F1)
- [x] Add failing tests for residual confidence, M_struct * conf^gamma, and independent fusion_v2 artifacts
- [x] Implement `baseline/experiments/hypir_fusion_v2/` without modifying HYPIR or overwriting fusion_v1
- [x] Reuse existing LQ/H50/H200 PNGs; no diffusion, training, LoRA, classifier, or new depth model
- [x] Run only validation case1-case5 with mask groups A/B/C/D plus gamma=0.5
- [x] Write results/fusion_v2/ (final/struct/conf masks, residual heatmap, fusion, crops, metrics.csv)
- [x] Inspect case4 fish-head, case3 water, case1/5 text+clock; write report.md vs texture_selective 28.48

## Next Step
F1 failed the 28.48 gate. Report the fusion_v2 artifacts and stop stacking output-space masks.

## Phase 11: HYPIR process-level control audit (no new fusion)
- [x] Read HYPIR enhancer/test/trainer/scheduler; map every inference knob from code
- [x] Select at most 2 real generation-freedom parameters; do not invent new ones
- [x] Do not re-run completed coeff_t/model_t inference; reuse existing 5-case PNGs
- [x] Independent package `baseline/experiments/hypir_process_control_v1/`
- [x] Re-score vs texture_selective 28.48 with LPIPS_1024 and unified diagnostic crops
- [x] Answer the 5 process-control questions with a single yes/no decision

## Next Step
Process control is at its inference ceiling. Do not sweep coeff_t again. Next stage, if authorized, is LoRA/Adapter on the mapping — not this round.

## Phase 12: LoRA feasibility audit (no training)
- [x] Inventory every local image set; do not guess counts
- [x] Trace HYPIR dataset loader: GT-only vs paired LQ/GT
- [x] Trace SD2Trainer LoRA inject, freeze, loss, timesteps
- [x] Compare official Real-ESRGAN degradation to CSIG 4K same-res
- [x] Write GO/NO-GO without training or downloads

## Next Step
NO-GO LoRA. Keep texture_selective_h200. Do not train, do not download LSDIR, do not finetune the 5 val pairs.

## Phase 13: fusion_v3 multi-band (last output-space test)
- [x] Reuse fusion_v1 YCbCr; Laplacian pyramid only (not FFT/DCT/HYPIR 2-band wavelet)
- [x] Independent `hypir_fusion_v3/`; do not overwrite v1/v2/texture_selective
- [x] Variants B/C/D with α_mid in {0.1, 0.2}; visualize LQ/H200 low-mid-high on case4
- [x] 5-case PSNR/SSIM/LPIPS vs 28.48; same diagnostic crops
- [x] GO or NO-GO multi-band; if NO-GO freeze the anchor

## Next Step
NO-GO multi-band. Freeze texture_selective_h200. Stop output-space tuning. Prepare test-set inference engineering if authorized.

## 对比工具用法

### Evaluation 二联图
```powershell
& .\.conda\python.exe baseline\compare_images.py
```

匹配 `baseline/evaluation_input/` 和 `baseline/evaluation_output/result/`，输出到 `baseline/comparison/`。

### Validation 三联图
```powershell
& .\.conda\python.exe baseline\compare_images.py `
  --input-dir baseline\input `
  --output-dir baseline\output\result `
  --ground-truth-dir csig_dataset\验证集 `
  --comparison-dir baseline\comparison
```

输出面板为 `Input/LQ | HYPIR Output | Ground Truth`。统一缩放只作用于 combine 展示图，不改变原始输入、HYPIR 输出或 GT。

## Evaluation 指标用法

```powershell
& .\.conda\python.exe baseline\evaluate_metrics.py `
  --lq-dir baseline\input `
  --gt-dir csig_dataset\验证集 `
  --output-dir baseline\output\result `
  --csv baseline\evaluation_metrics.csv `
  --device cuda
```

脚本输出每 case 和 `Average` 汇总，并在 CSV 中记录 PSNR、SSIM、LPIPS-Alex、delta 与 Improved/Degraded 结论。计算前会严格检查 RGB 三通道和尺寸一致性；不一致时直接报错。

## 错误记录
| 现象 | 处理 |
|---|---|
| `skimage` 未安装，无法直接调用 SSIM | 规划中改用可安装的 `torchmetrics`/`opencv` 或统一在评测环境复现官方指标；当前仅使用 MSE/PSNR 与清晰度做代理分析 |
| `数据集ossutil.md` 含明文 AK/STS | 已确认数据已在本地解压；不再执行该命令，凭据应立即作废或轮换 |
| `basicsr` 隔离构建报 `ModuleNotFoundError: torch` | 固定 `basicsr==1.4.2` 并使用 `--no-build-isolation`，再单独安装其余 wheel 包 |
| `basicsr`/`gfpgan` 声明依赖 `tb-nightly`，镜像索引未同步 | 从 PyPI 安装 `tb-nightly==2.21.0a20251023` 后，`pip check` 已通过 |
| Windows 下 `cv2.imread` 无法读取中文路径 | 推理读取改用 `cv2.imdecode(np.fromfile(path, dtype=np.uint8), ...)` 或 Pillow |
| 主机策略拒绝删除安装缓存 | 保留 `.cache` 中下载分段与 wheel，环境运行不依赖它们；确认后可由用户手动清理 |
| 批量解码全部 110 张 4K JPEG 的统计命令超过 24 秒超时 | 改为只检查验证对及少量代表测试图；不作为模型选型阻塞项 |
| 追加的并行指标收集调用在工具层失败，未返回子命令细节 | 不重复该组合调用；保留已有代理指标，后续验证时以独立脚本逐项记录结果 |
| `git diff` 无法运行 | 当前目录没有 Git 工作树；改用直接读取文件确认规划内容 |
| 并行 `conda run` 争用临时激活文件 | 验证调用同时启动导致 temp 文件冲突 | 改用项目环境的绝对路径 `D:\\MyProjects\\CSIG\\.conda\\python.exe` 串行执行 |
| `huggingface-cli whoami` 首次报 `UnicodeEncodeError` | Windows GBK 控制台无法编码弃用警告 | 使用 `hf.exe auth whoami`，确认当前未登录 |
| `snapshot_download(stabilityai/stable-diffusion-2-1-base)` 返回 `401 Unauthorized` | Stable Diffusion 2.1 base 是 gated 模型且本机无 token | 等待用户在 Hugging Face 接受许可并执行 `hf auth login`；不绕过 gating |
| 直接执行 `texture_weight_sweep.py` 找不到 `baseline` 包 | Windows 脚本目录不含项目根路径 | 增加模块/直接脚本双导入分支，并通过编译与测试 |
| LPIPS sweep 命令达到 20 分钟工具上限 | 每个方法重复编码 LQ/GT，子进程仍后台运行 | 停止明确 PID，改为每 case 一次 batch LPIPS；在独立 v2 目录完成 |
| 报告生成时空 weight 无法转浮点 | 汇总把基线空 weight 混入候选排序 | 只对 `texture_weight_*` 候选计算区域最佳权重，并从已写 CSV 重建报告 |

## 环境决策
- GPU：NVIDIA GeForce RTX 5080 Laptop GPU，计算能力 12.0，显存约 16 GB，驱动 610.78。
- Python：使用 Conda Python 3.11，环境路径固定为项目根目录 `.conda`。
- PyTorch：优先 CUDA 12.8 wheel，以覆盖 RTX 5080 的 Blackwell 架构；安装后用 `torch.cuda` 做运行时验证。

## Phase 10: E1 independent SwinIR fidelity baseline
- [x] Verify the official SwinIR color JPEG artifact-reduction task, configuration, and checkpoint structure from first-party sources
- [x] Inspect the five LQ files' actual encoding/degradation without assuming JPEG quality 40 matches them
- [x] Download exactly one official color JPEG40 SwinIR-M checkpoint and record source, size, SHA-256, and metadata
- [x] Add test-first inference/output contracts and a minimal tiled inference wrapper without modifying HYPIR/E0
- [x] Smoke-test checkpoint compatibility and tile/padding/output dimensions before validation inference
- [x] Run exactly five validation cases, save raw RGB PNGs, compute E0-compatible PSNR/SSIM/LPIPS, and record runtime/peak memory
- [x] Write `reports/E1_FIDELITY_BASELINE.md` with per-case comparison against HYPIR-50 and no autonomous E2 decision

## Next Step
E1 complete; hand the report to ChatGPT for the E2 decision. Do not start E2 autonomously.

## Phase 11: E2 Global Alpha Blend
- [x] Verify existing E0 HYPIR-50, E1 Fidelity, and GT images are RGB uint8 with exactly matching resolutions
- [x] Add test-first offline global blend implementation using only the six authorized alpha values
- [x] Generate and preserve 30 RGB PNGs in six `outputs/alpha_*` directories without rerunning either model
- [x] Compute E0-compatible PSNR, SSIM, and batched LPIPS-Alex plus deltas versus HYPIR-50 and Fidelity
- [x] Validate endpoint equivalence, output health, and write `reports/E2_GLOBAL_BLEND.md`

## Next Step
E2 complete; hand the report to ChatGPT for the E3 decision. Do not start E3 autonomously.

## Phase 12: E3-A Texture-only Adaptive Alpha
- [x] Add failing tests for LQ-only texture-map construction, fixed texture/reverse alpha mappings, convex RGB fusion, and artifact contracts.
- [x] Implement exactly the three authorized offline methods using existing HYPIR-50, SwinIR, LQ, and GT assets; do not run model inference or parameter sweeps.
- [x] Generate native-resolution RGB PNG outputs and LQ-only texture/alpha maps for case1-case5.
- [x] Compute E0/E1/E2-compatible PSNR, SSIM, LPIPS-Alex, alpha statistics, texture statistics/correlations, and per-case deltas.
- [x] Inspect maps and output health, run the existing unit suite, and write the E3-A report with the required conclusion only.

## Next Step
E3-A complete; stop here and report the fixed-scope result. Do not start E3-B/E3-C or any further experiment.

## Phase 13: Team handoff package
- [x] Inventory project sizes and identify reproducible source versus local environment/cache/model artifacts
- [x] Assemble a minimal handoff tree with documentation, source, tests, metrics, validation samples, and representative previews
- [x] Add a handoff README documenting included/excluded content, setup, reproduction commands, and exact size accounting
- [x] Create and verify the compressed archive without modifying the original experiment artifacts

## Next Step
Handoff archive verified; deliver `CSIG_team_handoff_2026-09-04.zip` and its size/hash summary.

## Phase 14: HYPIR-200 error decomposition (E0–E4)
- [x] Freeze FR metrics (PSNR/SSIM native, LPIPS max-side 1024)
- [x] Stratified 256 patch labels on case4 (24) and case3 (12); high-demand A=7/8
- [x] Multi-scale 1×…1/16 PSNR + visual panels
- [x] LQ–H200 blend curve vs H50 (global α fusion already done; stop as final method)
- [x] case4 multi-seed 17/89/401; deterministic mapping, fish/wrong leaves locked
- [x] Full non-overlap 256 grid PSNR/SSIM/LPIPS (960 tiles)
- [x] Write `error_decomposition_v1/report.md`; replace DAS-V1 with `CURRENT_PLAN.md`

## Phase 15: HYPIR output-space fusion v1
- [x] Independent module under `baseline/experiments/hypir_fusion_v1/` (no HYPIR source edits, no LoRA/Adapter/new models)
- [x] Scheme A (`base=LQ`) and scheme B (`base=H50`); Y-channel residual; LQ chroma
- [x] Sobel structure mask at 1/4 + scene-specific manual alpha
- [x] Unit tests and validation case1-case5 with PSNR/SSIM/LPIPS_1024
- [x] Write fusion/mask/heatmap/crops and `results/fusion_v1/report.md`

## Next Step
fusion_A 均 PSNR 28.46，未过门槛。Fusion 是降权压回模糊，不是识别删除鱼头。V1.1：`M_final = structure_mask × residual_penalty`（不要单独用大残差，会杀文字）。不要开 LoRA / 100 张。

## Errors Encountered
| Error | Attempt | Resolution |
|---|---:|---|
| `git diff` unavailable because `D:\\MyProjects\\CSIG` is not a Git worktree | 1 | Preserve the existing direct-artifact inspection workflow; do not retry Git status/diff commands. |
| E3-A test import stopped with `SyntaxError: f-string expression part cannot include a backslash` | 1 | Report formatter used an escaped nested f-string expression; replace it with a helper that formats correlation values. |
| E3-A visual inspection rejected `detail=low` | 1 | Local image viewer accepts only default high-resized or original detail; retry once with supported default detail. |
| E3-A averaging regression test could not import `_average_for_csv` | 1 | The helper did not yet exist; add it and use it for both CSV and report averages. |
| E3-A averaging regression test compared binary floats with exact equality | 1 | The helper returned the correct value with normal binary representation; use a precision-aware numeric assertion. |
| Read-only CUDA LPIPS batch-size probe exceeded the 120-second command limit | 1 | Do not repeat the expensive probe; retain the established E2 six-decimal reference for the pixel-identical global case row. |
| Final artifact audit used an overlapping `case*_texture.png` glob and failed its map-count assertion | 1 | Replace broad globs with explicit expected map filenames; no artifact was changed. |
