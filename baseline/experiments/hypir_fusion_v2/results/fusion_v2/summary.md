# HYPIR fusion v2 summary

Offline residual-confidence fusion (F1). H200 is a detail candidate;
LQ/H50 are structure anchors; `conf = 1 - normalize(|H200-LQ|)` down-weights
large residuals that the Sobel structure mask still allows.

- date: 2026-09-11
- LPIPS: Alex max-side 1024, computed=True
- scheme A: `Y = Y_LQ + α·M_final·(Y_H200 - Y_LQ)`
- scheme B: `Y = Y_H50 + α·M_final·(Y_H200 - Y_H50)`
- chroma: LQ Cb/Cr for both schemes
- A/B mask: `M_final = M_struct` (original fusion_v1; A and B share this formula)
- C mask: `M_final = M_struct * conf`
- D mask: `M_final = M_struct * conf^2`
- extra: `M_final = M_struct * conf^0.5`

## Average metrics

| method | group | formula | PSNR | SSIM | LPIPS_1024 |
|---|---|---|---:|---:|---:|
| LQ | ref | unchanged LQ | 28.034420 | 0.777699 | 0.204318 |
| HYPIR-50 | ref | coeff_t=50 | 28.257533 | 0.776935 | 0.162039 |
| HYPIR-200 | ref | coeff_t=200 | 24.528587 | 0.684926 | 0.159664 |
| fusion_A | A/B | M_struct, base=LQ | 28.460407 | 0.781670 | 0.188585 |
| fusion_B | A/B | M_struct, base=H50 | 28.256170 | 0.778593 | 0.156365 |
| fusion_A_conf05 | gamma=0.5 | M_struct * conf^0.5, base=LQ | 28.281209 | 0.780505 | 0.195124 |
| fusion_B_conf05 | gamma=0.5 | M_struct * conf^0.5, base=H50 | 28.350071 | 0.779430 | 0.157114 |
| fusion_A_conf | C | M_struct * conf, base=LQ | 28.197553 | 0.779771 | 0.198141 |
| fusion_B_conf | C | M_struct * conf, base=H50 | 28.365727 | 0.779709 | 0.157888 |
| fusion_A_conf2 | D | M_struct * conf^2, base=LQ | 28.109006 | 0.778809 | 0.201287 |
| fusion_B_conf2 | D | M_struct * conf^2, base=H50 | 28.368122 | 0.779899 | 0.158938 |
| texture_selective_h200 | anchor | current HYPIR-family anchor | 28.480280 | 0.781421 | 0.164549 |

## Per-case (scheme A / LQ base)

| case | scene | alpha | mean_struct | mean_conf | mean_final_conf | mean_final_conf2 | fusion_A | A_conf | A_conf2 | texture |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case1 | text | 0.25 | 0.2765 | 0.8710 | 0.2175 | 0.1893 | 32.1744 | 32.1094 | 32.0644 | 32.1545 |
| case2 | book | 0.30 | 0.5920 | 0.8259 | 0.4981 | 0.4449 | 29.2045 | 28.6043 | 28.4009 | 29.2414 |
| case3 | bird | 0.12 | 0.6311 | 0.8387 | 0.5425 | 0.4879 | 35.7429 | 35.6385 | 35.6040 | 35.6301 |
| case4 | plant | 0.08 | 0.4804 | 0.7280 | 0.3641 | 0.3005 | 18.0636 | 18.0596 | 18.0579 | 18.0199 |
| case5 | clock | 0.30 | 0.5561 | 0.8105 | 0.4531 | 0.3973 | 27.1167 | 26.5759 | 26.4178 | 27.3556 |

## Diagnostic crops

| case | crop | method | PSNR | SSIM |
|---|---|---|---:|---:|
| case3 | low02_water | LQ | 46.6415 | 0.9887 |
| case3 | low02_water | H200 | 34.4847 | 0.8452 |
| case3 | low02_water | A_struct | 46.0769 | 0.9878 |
| case3 | low02_water | A_conf05 | 46.1346 | 0.9880 |
| case3 | low02_water | A_conf | 46.1852 | 0.9881 |
| case3 | low02_water | A_conf2 | 46.2789 | 0.9883 |
| case3 | low02_water | B_struct | 42.2635 | 0.9767 |
| case3 | low02_water | B_conf | 42.3260 | 0.9769 |
| case3 | mid02_mud | LQ | 45.4147 | 0.9848 |
| case3 | mid02_mud | H200 | 31.5072 | 0.7897 |
| case3 | mid02_mud | A_struct | 44.6544 | 0.9830 |
| case3 | mid02_mud | A_conf05 | 44.7824 | 0.9832 |
| case3 | mid02_mud | A_conf | 44.9200 | 0.9835 |
| case3 | mid02_mud | A_conf2 | 45.1100 | 0.9841 |
| case3 | mid02_mud | B_struct | 39.4815 | 0.9671 |
| case3 | mid02_mud | B_conf | 39.6740 | 0.9680 |
| case4 | mid04_fish | LQ | 17.3633 | 0.2721 |
| case4 | mid04_fish | H200 | 15.7665 | 0.1927 |
| case4 | mid04_fish | A_struct | 17.3751 | 0.2725 |
| case4 | mid04_fish | A_conf05 | 17.3725 | 0.2726 |
| case4 | mid04_fish | A_conf | 17.3679 | 0.2723 |
| case4 | mid04_fish | A_conf2 | 17.3634 | 0.2720 |
| case4 | mid04_fish | B_struct | 17.3808 | 0.2668 |
| case4 | mid04_fish | B_conf | 17.3850 | 0.2672 |
| case4 | high02_leaf | LQ | 17.3139 | 0.2637 |
| case4 | high02_leaf | H200 | 16.1898 | 0.2096 |
| case4 | high02_leaf | A_struct | 17.3284 | 0.2645 |
| case4 | high02_leaf | A_conf05 | 17.3233 | 0.2643 |
| case4 | high02_leaf | A_conf | 17.3180 | 0.2640 |
| case4 | high02_leaf | A_conf2 | 17.3156 | 0.2638 |
| case4 | high02_leaf | B_struct | 17.3767 | 0.2673 |
| case4 | high02_leaf | B_conf | 17.3766 | 0.2671 |

## Visual checklist

- case4 `crops/case4_mid04_fish.png`: does residual conf further flatten the fish-head vs fusion_v1?
- case4 `crops/case4_high02_leaf.png`: real foliage vs invented serrated edges
- case3 `crops/case3_low02_water.png`: invented water grain vs fusion_v1
- case1/2/5 comparison panels: text, spine, clock hands must not be redrawn

## How to read the masks

- `masks/*_struct*`: Sobel agreement. Bright = H200 contour agrees with LQ.
- `masks/*_conf*`: residual confidence. Dark = |H200-LQ| is large.
- `masks/*_final_*`: M_struct * conf^gamma actually applied.
- `heatmaps/*_residual*`: |H200-LQ| before/after percentile norm.

## Not in this experiment

HYPIR re-inference, diffusion parameter changes, LoRA, Adapter,
classifier, depth model, training, 100-image test set.
