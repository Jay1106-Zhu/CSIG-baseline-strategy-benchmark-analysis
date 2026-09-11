# HYPIR LoRA feasibility audit

**NO-GO LoRA**

只做代码审计 + 数据审计。未训练、未下载、未改 HYPIR、未跑扩散、未改 fusion/anchor。

## 文件

| 文件 | 内容 |
|---|---|
| `go_no_go.md` | 结论与 10 条证据 |
| `dataset_inventory.md` | 本地所有图像集计数 |
| `training_pipeline.md` | train.py / loss / freeze / LoRA init |
| `lora_target_analysis.md` | 官方 target modules |
| `degradation_gap.md` | Real-ESRGAN 4× vs CSIG 4K |

## 一句话

当前仓库只有 5 对真实 LQ–GT，官方训练脚本读的是 GT + 在线 4× 合成退化，且不加载已有 `HYPIR_sd2.pth`。不具备对 CSIG mapping 做 LoRA adaptation 的数据条件。
