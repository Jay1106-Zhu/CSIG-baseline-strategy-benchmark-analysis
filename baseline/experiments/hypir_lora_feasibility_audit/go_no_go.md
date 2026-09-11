# NO-GO LoRA

本轮未训练、未下载、未改 HYPIR、未跑扩散。判断只基于当前仓库代码和磁盘文件。

## 最关键证据

1. **仓库里没有可训练的 LQ→GT 集。** 唯一 pixel-aligned 配对是 CSIG 验证集 5 对。测试集 100 张没有 GT。没有 LSDIR / DF2K / DIV2K / parquet 文件列表。
2. **这 5 对是唯一带 GT 的真实评测，不能拿来训 LoRA。** 训完再在这 5 张上刷 28.48 没有外推意义。
3. **官方 loader 不是读 (LQ, GT)。** `RealESRGANDataset` 只加载 GT（HQ），`RealESRGANBatchTransform` 在线合成 LQ。`configs/sd2_train.yaml` 的 `file_list` 仍是 `TODO`。
4. **官方合成退化与 CSIG 同分辨率 4K 修复不匹配。** `stage2_scale: 4`、resize 下限 0.15、JPEG 到 30、先 USM 锐化 GT。赛题输入与 GT 同尺寸，不是 4× SR。
5. **当前失败模式不是通用 blur/JPEG。** Case4 是确定性错误中频植物 prior（锯齿单叶、鱼头）。用 Real-ESRGAN 合成 LQ 再拟合 GT，学不到「CSIG 模糊粉团 → 荚果/复叶」。
6. **仓库没有五类标签文件。** 验证集可人工看成文字/书脊/鸟/绿植/钟表；无人脸。测试 100 张赛题声称含小人脸，但目录里没有任何类别标注。
7. **官方 LoRA 是 rank 256、257 层对、259.47M 参数。** 5 张图无法支撑。原论文训练 batch 1024；本地 yaml 是 30k step + GAN + VGG-LPIPS + ConvNeXt D。
8. **`SD2Trainer.init_generator` 不加载 `HYPIR_sd2.pth`。** 它在 SD2.1 UNet 上 `add_adapter` + `init_lora_weights="gaussian"`，是从头训 LoRA，不是对现有 HYPIR mapping 做 adaptation。
9. **代码结构可以复用，数据条件不成立。** 有 `train.py`、target modules 明确、VAE/text encoder 已冻结。缺的是配对数据和匹配的退化，不是缺脚本。
10. **inference-level 已经证明拧 `coeff_t` 只是少画同一 prior。** 在没有 CSIG 匹配数据时，再训一遍通用 Real-ESRGAN LoRA 不会改掉鱼头。

## 最现实的替代路线

保持提交锚 **`texture_selective_h200`（均 PSNR 28.480）**。不要训 LoRA，不要再扫 mask / coeff_t。

不要用 5 对验证集 finetune。不要为了「有 checkpoint」去下载 LSDIR。

若以后有外部、许可清晰、且与 CSIG 同分辨率退化接近的大量配对数据，才重新评估 mapping adaptation。那是新数据条件，不是当前仓库条件。
