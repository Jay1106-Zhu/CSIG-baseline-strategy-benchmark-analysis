# Degradation gap：官方合成 LQ vs CSIG

## 官方训练退化（代码路径）

`RealESRGANBatchTransform.__call__`（`HYPIR/HYPIR/dataset/batch_transform.py:158`）+ yaml：

1. 可选 USM 锐化 **GT**（`use_sharpener: true`）
2. 第一段：blur（kernel 7–21，sigma 0.2–3）→ 随机 resize（prob up/down/keep，范围 0.15–1.5）→ Gaussian 或 Poisson → JPEG quality 30–95
3. 第二段：`stage2_scale: 4` 把空间尺寸除以 4 → 再 blur/resize/noise/JPEG 30–95 → sinc
4. `resize_back: true`：bicubic 拉回 512

这是经典 Real-ESRGAN **4× SR 合成**。LQ 在过程中经过低分辨率，再被拉回。退化每次随机，batch 内用 queue_size=256 打散。

## CSIG 实际输入（磁盘）

- LQ 与 GT **同分辨率**（4K），不是 4× 下采样任务
- 验证 5 对 filename 配对，mean |LQ−GT|：case3 2.20（轻），case1 2.90，case5 7.22，case2 6.07，case4 **24.06**（绿植差最大）
- 仓库 **没有** CSIG 退化配方、没有 kernel、没有 JPEG quality 标签
- 已有实验：H200 相对 LQ 是确定性映射偏差（E4 多种子几乎相同），不是随机压缩噪声

## 逐项对照

| 现象 | 官方合成 | CSIG 验证观察 | 匹配？ |
|---|---|---|---|
| blur | 双段各向异性/sinc，sigma 到 3 | 有失锐，但是同分辨率 | 部分 |
| noise | Gaussian/Poisson 经常开 | 不是主因 | 弱 |
| JPEG | 双段，quality 可到 30 | 可能有压缩，幅度未知 | 未知 |
| color shift | 无专门项；USM 改 GT | case 有色差但结构对齐 | 弱 |
| 4× 下采样 | **强制 stage2_scale=4** | **无。输入已是 4K** | **否** |
| ringing/aliasing | resize+sinc 会造 | 可能有，但不是 4× SR 伪影 | 弱 |
| 中频植物语义 | 合成 LQ 仍是同一张植物的模糊版 | H200 把粉团画成鱼头、复叶画成锯齿单叶 | **否** |

## 五类内容

| 类 | 验证集 | 测试集标签 | 官方 LSDIR（本机） |
|---|---|---|---|
| 文字 | case1 | 仓库无标签 | 未下载，无法确定 |
| 书脊/文字 | case2 | 无 | 无法确定 |
| 鸟/羽毛 | case3 | 无 | 无法确定 |
| 钟表 | case5 | 无 | 无法确定 |
| 人物/人脸 | **验证集没有** | 赛题声称测试含小人脸；无文件可数 | 无法确定 |
| 密集绿植 | case4 | 无 | 无法确定 |

即使下载 LSDIR，它也是「自然图 GT + 随机 4× 退化」，**不会**提供「CSIG 那种模糊粉团应对应荚果」的监督。那是内容 prior 问题，不是再加一层 JPEG。

## 结论

官方退化 **reasonably matched？否。**  
主要缺口：4× SR 假设，以及完全没有针对错误植物语义的配对。用这条 pipeline 训 LoRA，期望的是「解模糊/解压缩」，得到的仍可能是更强的错误中频植物。
