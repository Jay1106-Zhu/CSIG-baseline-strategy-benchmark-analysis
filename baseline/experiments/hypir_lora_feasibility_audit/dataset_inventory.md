# dataset inventory

计数来自 2026-09-11 对磁盘的实际遍历，不是 README。

没有找到任何 `.parquet` 训练列表。`sd2_train.yaml` 的 `file_list` / `image_path_prefix` / `image_path_key` / `prompt_key` 全部是 `TODO`。

| 数据集 | LQ数量 | GT数量 | 是否paired | 分辨率 | 是否有五类标签 | 是否适合LoRA |
|---|---:|---:|---|---|---|---|
| CSIG 验证集 `csig_dataset/验证集/` | 5 | 5 | 是：`caseN_lq.jpg` ↔ `caseN_gt.jpg`，同尺寸 RGB | case1/3/4/5：4096×3072；case2：3072×4096 | 仓库无标签文件。人工：文字/书脊/鸟/绿植/钟表。无人脸 | 否。仅 5 对，且是唯一真实评测 |
| CSIG 测试集 `csig_dataset/测试集/` | 100 | 0 | 否，无 GT | 71 张 4096×3072，29 张 3072×4096，皆 RGB | 赛题文本声称五类；目录无 csv/json/txt 标注 | 否。无 GT 不能训 |
| `baseline/input/` | 5 | 0 | LQ 副本，GT 在验证集 | 同上 | 无 | 否 |
| `team_handoff/sample_data/validation/` | 5 | 5 | 验证集副本 | 同上 | 无 | 否 |
| HYPIR `examples/lq/` | 6 | 0 | 否 | 256×256 或 225×225；3 张 RGBA | 无 | 否。演示 LQ，无 GT |
| HYPIR `assets/gallery_sd2/` | 0 | 0 | 展示输出，不是训练对 | 约 550–1584 × 826–832 | 无 | 否 |
| HYPIR 官方 LSDIR_512 | 无法从当前仓库确定 | 无法从当前仓库确定 | README 示例路径 `/opt/data/common/data260t/LSDIR_512` 在本机不存在 | 无法从当前仓库确定 | 无法从当前仓库确定 | 否。未下载 |
| DF2K / DIV2K / Flickr2K | 0 | 0 | 仅 `third_party/DiffIR` 文档提到下载 URL | 无法从当前仓库确定 | 无 | 否。本地无图像 |
| ImageNet 词表 | 0 | 0 | `.conda` 里 timm 的 synset 文本 | 不适用 | 不适用 | 否 |
| 官方 train file_list | 0 | 0 | yaml 为 TODO | 不适用 | 无 | 否 |

## 验证集配对抽查（实际打开文件）

| case | LQ size | GT size | 同 shape | mean abs(LQ−GT) |
|---|---|---|---|---:|
| 1 | 4096×3072 RGB | 4096×3072 RGB | 是 | 2.900 |
| 2 | 3072×4096 RGB | 3072×4096 RGB | 是 | 6.072 |
| 3 | 4096×3072 RGB | 4096×3072 RGB | 是 | 2.195 |
| 4 | 4096×3072 RGB | 4096×3072 RGB | 是 | 24.061 |
| 5 | 4096×3072 RGB | 4096×3072 RGB | 是 | 7.215 |

filename mapping 可靠。像素对齐（同分辨率）。不是「同一场景不同随机文件名」。仓库里没有退化参数记录，无法从文件复现 CSIG 官方退化。

## 明确不存在的东西

- 训练用 LQ 目录
- 训练用 GT 目录
- 类别标注
- LSDIR parquet
- 已缓存的合成 LQ 训练集
