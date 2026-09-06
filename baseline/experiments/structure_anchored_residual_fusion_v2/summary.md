# Structure-Anchored Residual Fusion v2

结论：**停止该方向**。这是一个只使用已有图像的离线验证；没有训练、LoRA 更新或新增 HYPIR/SD 推理。

## 方法

`R200 = HYPIR-200 - LQ`，最终输出为 `F = LQ + gate * R200`。gate 只由 LQ 的 Sobel 梯度、Canny 邻域、局部纹理/模糊代理，以及 H200 residual 的边缘新颖性和梯度方向一致性构成。GT 从未进入 gate 或任何阈值。

固定门控：strong_edge=0.020；weak_edge=0.100；existing-structure-nearby=0.065；其他 non-edge=0.035。新边缘 novelty 会乘以 `(1 - 0.85 * novelty)`，方向一致最多增加 0.035，strong_edge 最终仍封顶 0.020。

## 输入与产物

- LQ: `D:\MyProjects\CSIG\baseline\input`；GT（仅评价）: `D:\MyProjects\CSIG\csig_dataset\验证集`
- HYPIR-200: `D:\MyProjects\CSIG\baseline\experiments\coeff_t_200\output\result`；HYPIR-50: `D:\MyProjects\CSIG\baseline\experiments\coeff_t_50\output\result`；current texture_selective_h200: `D:\MyProjects\CSIG\baseline\experiments\texture_weight_sweep_v2\fusion\texture_selective`
- v2 只写入本目录的 `fusion/v2/`；旧实验目录不修改。

## 全图指标平均

| Method | PSNR | SSIM | LPIPS-Alex |
|---|---:|---:|---:|
| LQ | 28.034420 | 0.777699 | 0.204315 |
| HYPIR-50 | 28.257533 | 0.776935 | 0.162039 |
| HYPIR-200 | 24.528587 | 0.684926 | 0.159666 |
| texture_selective_h200 | 28.480280 | 0.781421 | 0.164546 |
| Structure-Anchored Residual Fusion v2 | 28.158209 | 0.779548 | 0.195091 |

## Per-case PSNR (SSIM/LPIPS per case are in metrics.csv)

| Case | LQ | H50 | H200 | texture_selective_h200 | v2 |
|---|---:|---:|---:|---:|---:|
| case1 | 32.0288 | 32.1433 | 29.3353 | 32.1545 | 32.0784 |
| case2 | 28.1894 | 28.9763 | 24.2140 | 29.2414 | 28.4733 |
| case3 | 35.6221 | 34.6887 | 28.8654 | 35.6301 | 35.7040 |
| case4 | 18.0562 | 18.0846 | 16.3524 | 18.0199 | 18.0631 |
| case5 | 26.2756 | 27.3948 | 23.8758 | 27.3556 | 26.4722 |

## 分区域证据

`region_metrics.csv` 同时记录 change_L1 与 error_to_GT_L1。change 变小只代表更少改写，error 变小才是对 GT 的恢复证据。

### strong_edge

| Method | change_L1 | error_to_GT_L1 |
|---|---:|---:|
| LQ | 0.000000 | 18.227353 |
| HYPIR-50 | 8.794542 | 16.883241 |
| HYPIR-200 | 24.102404 | 24.158529 |
| texture_selective_h200 | 5.551991 | 16.912734 |
| Structure-Anchored Residual Fusion v2 | 0.481077 | 18.047649 |

### textured_non_edge

| Method | change_L1 | error_to_GT_L1 |
|---|---:|---:|
| LQ | 0.000000 | 10.369658 |
| HYPIR-50 | 3.235962 | 10.027113 |
| HYPIR-200 | 8.228671 | 11.541013 |
| texture_selective_h200 | 1.974473 | 9.952752 |
| Structure-Anchored Residual Fusion v2 | 0.779281 | 10.156728 |

### blurred_texture

| Method | change_L1 | error_to_GT_L1 |
|---|---:|---:|
| LQ | 0.000000 | 17.303354 |
| HYPIR-50 | 2.125488 | 17.337979 |
| HYPIR-200 | 9.609250 | 19.864564 |
| texture_selective_h200 | 2.234327 | 17.397288 |
| Structure-Anchored Residual Fusion v2 | 0.965017 | 17.307310 |

## 重点 case 检查

- case1 text：strong_edge gate 用于降低 H200 对文字笔画的改写；需以 panel 和 error_to_GT_L1 判定，不把锐度变化当作 OCR 正确。
- case2 book spine：同样检查竖向书脊结构是否少改写；没有使用文字检测或 GT mask。
- case3 bird：检查羽毛/轮廓与背景交界，方向一致只允许小幅残差。
- case4 foliage：检查 blurred_texture 是否获得有限恢复，同时防止 H200 新边缘扩散。
- case5 clock：检查圆环、刻度和指针的边缘保护；没有宣称数字或几何身份恢复。

## 决策规则与限制

本次不做参数 sweep。只有当 v2 平均 PSNR 比 H50 和 current texture_selective_h200 都至少高 0.05 dB，且平均 SSIM 不低于两者最优值 0.001 以上，才视为值得继续；否则停止这个方向。验证集只有 case1-case5，结论不代表未见测试集。

详见 `metrics.csv`、`region_metrics.csv`、`gate_maps/` 和每个 case 的 `comparison/case*.png`。
