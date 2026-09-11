# FINAL STRATEGY AUDIT（跑图前，2026-09-11）

本轮 **没有** 改代码、没有跑 100 张 test、没有新实验。

`FINAL RECOMMENDATION` 在文末。

---

## Step 1 — 代码与记录是否一致

未改任何源码。对照现有实现与已记录实验：

| 组件 | 真实入口 | 与记录是否一致 |
|---|---|---|
| H50 | `HYPIR/test.py` + `coeff_t=50`，`model_t=200`，`upscale=1`，`captioner=empty`。产物 `baseline/experiments/coeff_t_50/output/result/` | 一致。均 PSNR 28.257533 |
| H200 | 同上，`coeff_t=200`。产物 `coeff_t_200/output/result/` | 一致。24.528587 |
| texture_selective_h200 | `baseline/experiments/structure_local_restoration.py`：LQ-only 权重 `0.05 + 0.46*texture*(0.35+0.65*(1-edge))`，clip 到 `[0, 0.60]`，**RGB** 混合 `w*H200+(1-w)*LQ`。产物 `structure_local_restoration_v1/fusion/texture_selective/h200/` | 一致。28.480280 / 0.781421 / 0.16455 |
| fusion_v1_A | `hypir_fusion_v1`：Y 通道 `LQ + α_scene·M_struct·(H200−LQ)`，Cb/Cr 锁 LQ。`CASE_TO_SCENE` **写死验证集 case1–5** | 指标一致 28.460407。**不能按测试集文件名复用该表** |
| 验证推理 | `run_coeff_t_sweep.ps1` 显式 `--upscale 1` | 一致 |
| 官方/默认推理 | `HYPIR/test.py` **`--upscale` 默认 4**；`HYPIR/README.md` 示例也是 4 | **与赛题“分辨率不变”不一致**。跑 test 必须显式 `upscale=1` |
| 测试侦察 | `baseline/evaluation_input/case1.jpg` 来自测试集 case1，已跑过 HYPIR | 仅 1/100，无 GT |

不自行重构。上述 upscale 默认值只作为跑图时的配置风险记录。

---

## Step 2 — 五个候选

数字一律来自 fusion_v1 口径（PSNR / SSIM / LPIPS-Alex max-side 1024），与锚同一套：

| 方法 | 均 PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| LQ | 28.034 | 0.778 | 0.204 |
| H50 | 28.258 | 0.777 | 0.162 |
| H200 | 24.529 | 0.685 | 0.160 |
| fusion_v1_A | 28.460 | 0.782 | 0.189 |
| **texture_selective_h200** | **28.480** | **0.781** | **0.165** |

### A. LQ

- 优势：无 hallucination；case3 PSNR 最高之一（35.62，仅 fusion_A 略高）。
- 风险：case2/case5 明显低于 H50/texture（钟表 −1.1 dB vs H50）。
- 适合：极保守保真、鸟类水面。
- 不适合：书脊、钟表（少改等于放弃已验证的锐度收益）。
- 不能作为默认提交。

### B. H50

- 优势：全图第二稳；case4 PSNR 最高（18.085）；case5 最高（27.395）；鱼头几乎未画实。
- 风险：case2 低于 texture（28.98 vs 29.24）；case3 低于 LQ（34.69 vs 35.62）。
- 适合：绿植、钟表（有验证数字）。
- 不适合：作为唯一默认——平均比锚低 0.22 dB。
- 不是“正确恢复”，只是少改。

### C. H200

- 优势：case4 LPIPS 最好（0.377）；看起来最锐。
- 风险：均 PSNR 24.53；确定性错叶/鱼头；case3 −6.76 dB。
- 适合：无。
- 不适合：任何最终输出。禁止作为默认。

### D. texture_selective_h200（冻结锚）

- 优势：五张平均最好；不依赖类别名；强边降权（文字/钟表保护）；权重上限 0.60，不会变成裸 H200。
- 风险：case4 PSNR 18.020 **低于 LQ 18.056**；高方差非边缘（正是绿植）仍放进 H200，鱼头可残留（fusion_v3 crop：texture 仍是粉色鱼形）。
- 适合：无标签的 100 张默认；文字/书脊/鸟。
- 不适合：若已知是密集绿植，H50 更安全（但测试集不知）。

### E. fusion_v1_A

- 优势：case1/case3 PSNR 略高于 texture（32.174 / 35.743）；植物 α=0.08 把鱼头压成模糊粉团。
- 风险：均 PSNR 低锚 0.020；LPIPS 0.189 差于锚 0.165；**场景 α 绑死验证 case id**；未知 case 回退 `default_alpha=0.15`，对测试 case1.jpg（≠验证文字）会用错 α。
- 适合：仅当 **已知** 场景且能手工选 α。
- 不适合：无标注的 100 张。

---

## Step 3 — Category routing（分析用，不是可执行策略）

验证集逐 case 最优（PSNR，fusion_v1 表）：

| category | val | PSNR 最优 | 证据强度 | 若有标签会选 | 可执行？ |
|---|---|---|---|---|---|
| text | case1 | fusion_A 32.174（texture 32.154） | 中 | texture 或 fusion_A | 测试无标签 |
| bird/feather | case3 | fusion_A 35.743（texture 35.630，LQ 35.622） | 中：必须非常保守 | fusion_A / texture | 测试无标签 |
| clock | case5 | **H50 27.395**（texture 27.356） | 中 | H50 | 测试无标签 |
| people/face | **验证集没有** | — | **LOW CONFIDENCE** | 默认 texture（强边保护），不要猜 | 无证据 |
| dense plants | case4 | **H50 18.085**（fusion_A 18.064，LQ 18.056，texture 18.020） | 高（幻觉） | H50 | 测试无标签 |

**不能**把这张表直接打到测试集 `case1.jpg`–`case100.jpg` 上：测试 `case1` 不是验证文字图。

已有反例：`scenario_routing_v1` 用 LQ 启发式把 case1/3/4 判成 H200，均 PSNR **25.80**，比锚低 2.68 dB。结论当时就是停 routing、不训分类器。

因此：**可执行的 category routing 不存在。** 分析表只说明“假如有人标了类会怎么选”。

---

## Step 4 — Dense plants

1. **为什么不能直接用 H200？** case4 PSNR 16.35 vs LQ 18.06（−1.70）。E4：seed 231/17/89/401 鱼头/错叶不变。fusion_v3：眼睛和鳞片在高频。coeff_t 只是把同一 prior 画浅或画实。
2. **为什么 H50 也不是正确恢复？** PSNR 仅 +0.028 vs LQ。视觉上仍是糊的粉团，不是 GT 荚果/复叶。它赢在 **少把错误中频写进去**。
3. **texture_selective 比 H50 多什么风险？** 公式在高 texture、低 edge 处提高 H200 权重。绿植正是这类区域。case4 PSNR 18.020 < H50 18.085，且 < LQ。fusion_v3 `crops/case4_fish.png` 上 texture 仍是鱼形，H50/LQ 更接近 blob。
4. **fusion_v1_A 是否值得最终提交？** 植物上鱼头更糊（α=0.08），PSNR 18.064 略好于 texture，但平均仍低于锚，且 **测试集无法使用验证 case→scene 表**。不值得作为 100 张方案。
5. **降鱼头 hallucination，数字支持谁？**  
   - 最安全：H50（或 LQ）  
   - 次之：fusion_v1_A（粉团，不是删除）  
   - 再次：texture_selective（鱼形仍在）  
   - 最差：H200  
   在 **不能识别绿植** 的前提下，不能把 100 张都改成 H50（会丢掉 case2 +1.05 和整体 0.22 dB）。

---

## Step 5 — 统一策略 vs 分类策略

**方案 U：** 100 张全部 `texture_selective_h200`。

**方案 C：** 按类切换。Oracle 混用（fusion_A, texture, fusion_A, H50, H50）大约均 PSNR 28.53，刚到旧门槛。这是 **验证集标签泄漏的上界**，不是测试集可得的方法。

| | 方案 U | 方案 C |
|---|---|---|
| 平均（五张） | 28.480 已测 | 理论 ~28.53，**未在无标签条件下得到** |
| 测试集可执行 | 是：LQ 特征，无类名 | 否：无标注；LQ routing 已失败 |
| 绿植幻觉 | 中等（比 H200 好，比 H50 差） | 若识别对了则更好；识别错成 H200 则灾难 |
| 稳定性 | 单一确定流水线 | 依赖不可靠的类 |

**选 U。**

类别识别失败时的 fallback（若有人以后硬做 routing）：**永远回退 `texture_selective_h200`**，禁止随机，禁止回退 H200。

---

## Step 6 — 测试集有没有类别信息

`csig_dataset/测试集/`：恰好 100 个 `case1.jpg` … `case100.jpg`。无 csv/json/txt/xml 标注。无子目录。无 prompt。71 张 4096×3072、29 张 3072×4096，皆 RGB。

**不能安全按 category routing。** 禁止为此训分类器或上 CLIP/DINO。

---

## Step 7 — 跑图前决策表

| 优先级 | 方法 | 使用场景 | 风险 | 是否建议最终跑 |
|---:|---|---|---|---|
| 1 | texture_selective_h200 | **全部 100 张 test** | case4 类图可能略低于 LQ；鱼头可残留但远弱于裸 H200 | **YES** |
| 2 | H50 | 仅当 **已知** 密集绿植/钟表（本测试集无标签） | 书脊/平均分会掉 | **NO**（无标签时） |
| 3 | fusion_v1_A | 仅验证集场景表；测试文件名会错配 α | 错配 α；平均低于锚 | **NO** |
| — | H200 | 无 | 确定性幻觉 | **NO** |
| — | LQ | 无 | 放弃钟表/书脊收益 | **NO** |

### FINAL RECOMMENDATION

* **默认方法：** `texture_selective_h200`（先 `HYPIR/test.py` `model_t=200 coeff_t=200 upscale=1 captioner=empty seed=231`，再跑现有 `structure_local_restoration.py` 的 texture_selective 权重，输出 JPG 名为 `case{k}.jpg`）。
* **category routing：** 不启用。
* **fallback：** `texture_selective_h200`（确定、非随机）。
* **是否需要修改代码：** 本轮 **否**。真正跑 100 张前需要一层 **工程封装**（强制 `upscale=1`、PNG→JPG、目录命名），那是提交工程，不是新研究。`DEFERRED / NOT FOR THIS SUBMISSION ROUND`。
* **是否需要新实验：** **否。**
* **是否可以开始跑 100 张 test：** **本轮否**（任务禁止）。**策略上已冻结，可以进入提交工程。** 不要再做 fusion/mask/coeff/LoRA。

---

## 原则核对

1. 已有实验 > 猜测：锚 28.480 未被任何允许候选超过。
2. 幻觉风险 > 锐度：禁止 H200 默认。
3. 稳定性 > 单张：不启用已失败的 routing。
4. 简单可靠 > 新模块：复用现成 texture_selective。
5. 不破坏锚：不换 fusion_v1_A（−0.02 dB 且不可移植）。
6. 停止 inference tuning。
