# HYPIR process control v1 summary

Offline. Single-step HYPIR. Selected control: `coeff_t` (already inferred).
No HYPIR re-inference, no fusion overwrite, no LoRA/Adapter/ControlNet.

- date: 2026-09-11
- LPIPS: Alex max-side 1024, computed=True
- selected: coeff_t
- levels: 50, 75, 100, 150, 200
- eps scales: 50=0.2256, 75=0.2866, 100=0.3439, 150=0.4561, 200=0.5717

## Average metrics

| method | PSNR | SSIM | LPIPS_1024 | vs texture 28.48 |
|---|---:|---:|---:|---|
| LQ | 28.034420 | 0.777699 | 0.204318 | ref |
| coeff_t_50 | 28.257533 | 0.776935 | 0.162039 | no |
| coeff_t_75 | 27.957945 | 0.771081 | 0.155251 | no |
| coeff_t_100 | 27.504505 | 0.762503 | 0.150622 | no |
| coeff_t_150 | 26.213322 | 0.734088 | 0.148753 | no |
| coeff_t_200 | 24.528587 | 0.684926 | 0.159664 | no |
| texture_selective_h200 | 28.480280 | 0.781421 | 0.164549 | anchor |
| fusion_v1_A | 28.460407 | 0.781670 | 0.188585 | no |

## Per-case PSNR

| case | LQ | t=50 | t=75 | t=100 | t=150 | t=200 | texture | fusion_A |
|---|---|---|---|---|---|---|---|---|
| case1 | 32.0288 | 32.1433 | 32.0009 | 31.7572 | 30.8843 | 29.3353 | 32.1545 | 32.1744 |
| case2 | 28.1894 | 28.9763 | 28.5931 | 27.9294 | 26.1587 | 24.2140 | 29.2414 | 29.2045 |
| case3 | 35.6221 | 34.6887 | 34.1660 | 33.4723 | 31.5233 | 28.8654 | 35.6301 | 35.7429 |
| case4 | 18.0562 | 18.0846 | 18.0416 | 17.9393 | 17.3899 | 16.3524 | 18.0199 | 18.0636 |
| case5 | 26.2756 | 27.3948 | 26.9880 | 26.4243 | 25.1104 | 23.8758 | 27.3556 | 27.1167 |

## Diagnostic crops (PSNR vs GT)

| case | crop | method | PSNR | SSIM |
|---|---|---|---:|---:|
| case1 | text | LQ | 25.8140 | 0.7957 |
| case1 | text | t=50 | 25.5237 | 0.8041 |
| case1 | text | t=75 | 25.1708 | 0.7934 |
| case1 | text | t=100 | 24.6798 | 0.7772 |
| case1 | text | t=150 | 23.2990 | 0.7296 |
| case1 | text | t=200 | 21.3725 | 0.6699 |
| case1 | text | texture | 25.5772 | 0.7876 |
| case1 | text | fusion_A | 25.8036 | 0.7966 |
| case2 | spine | LQ | 27.2461 | 0.8279 |
| case2 | spine | t=50 | 28.3806 | 0.8467 |
| case2 | spine | t=75 | 27.6518 | 0.8412 |
| case2 | spine | t=100 | 26.6592 | 0.8314 |
| case2 | spine | t=150 | 24.2749 | 0.8017 |
| case2 | spine | t=200 | 22.1558 | 0.7654 |
| case2 | spine | texture | 28.7619 | 0.8544 |
| case2 | spine | fusion_A | 28.6061 | 0.8506 |
| case3 | bird | LQ | 34.9938 | 0.8649 |
| case3 | bird | t=50 | 34.4053 | 0.8522 |
| case3 | bird | t=75 | 34.2499 | 0.8494 |
| case3 | bird | t=100 | 34.0078 | 0.8457 |
| case3 | bird | t=150 | 33.0007 | 0.8306 |
| case3 | bird | t=200 | 30.8277 | 0.7912 |
| case3 | bird | texture | 35.0280 | 0.8656 |
| case3 | bird | fusion_A | 35.0601 | 0.8657 |
| case3 | water | LQ | 46.6415 | 0.9887 |
| case3 | water | t=50 | 42.1484 | 0.9745 |
| case3 | water | t=75 | 41.2441 | 0.9678 |
| case3 | water | t=100 | 40.2281 | 0.9580 |
| case3 | water | t=150 | 37.7768 | 0.9226 |
| case3 | water | t=200 | 34.4847 | 0.8452 |
| case3 | water | texture | 45.9460 | 0.9869 |
| case3 | water | fusion_A | 46.0769 | 0.9878 |
| case3 | mud | LQ | 45.4147 | 0.9848 |
| case3 | mud | t=50 | 39.6949 | 0.9662 |
| case3 | mud | t=75 | 38.7093 | 0.9577 |
| case3 | mud | t=100 | 37.5531 | 0.9453 |
| case3 | mud | t=150 | 34.8331 | 0.8989 |
| case3 | mud | t=200 | 31.5072 | 0.7897 |
| case3 | mud | texture | 43.5954 | 0.9800 |
| case3 | mud | fusion_A | 44.6544 | 0.9830 |
| case4 | fish | LQ | 17.3633 | 0.2721 |
| case4 | fish | t=50 | 17.3852 | 0.2671 |
| case4 | fish | t=75 | 17.3280 | 0.2604 |
| case4 | fish | t=100 | 17.2120 | 0.2510 |
| case4 | fish | t=150 | 16.7032 | 0.2254 |
| case4 | fish | t=200 | 15.7665 | 0.1927 |
| case4 | fish | texture | 17.3317 | 0.2583 |
| case4 | fish | fusion_A | 17.3751 | 0.2725 |
| case4 | leaf | LQ | 17.3139 | 0.2637 |
| case4 | leaf | t=50 | 17.3877 | 0.2674 |
| case4 | leaf | t=75 | 17.3638 | 0.2647 |
| case4 | leaf | t=100 | 17.3070 | 0.2590 |
| case4 | leaf | t=150 | 16.9814 | 0.2370 |
| case4 | leaf | t=200 | 16.1898 | 0.2096 |
| case4 | leaf | texture | 17.3479 | 0.2626 |
| case4 | leaf | fusion_A | 17.3284 | 0.2645 |
| case4 | yellow_flower | LQ | 16.0335 | 0.2022 |
| case4 | yellow_flower | t=50 | 16.0705 | 0.2057 |
| case4 | yellow_flower | t=75 | 16.0361 | 0.2056 |
| case4 | yellow_flower | t=100 | 15.9310 | 0.2033 |
| case4 | yellow_flower | t=150 | 15.4440 | 0.1964 |
| case4 | yellow_flower | t=200 | 14.7592 | 0.1855 |
| case4 | yellow_flower | texture | 16.0309 | 0.2054 |
| case4 | yellow_flower | fusion_A | 16.0441 | 0.2034 |
| case5 | clock | LQ | 32.0487 | 0.9467 |
| case5 | clock | t=50 | 33.6413 | 0.9523 |
| case5 | clock | t=75 | 33.5603 | 0.9486 |
| case5 | clock | t=100 | 33.3463 | 0.9430 |
| case5 | clock | t=150 | 32.4898 | 0.9162 |
| case5 | clock | t=200 | 31.2887 | 0.8435 |
| case5 | clock | texture | 32.7372 | 0.9506 |
| case5 | clock | fusion_A | 32.6163 | 0.9476 |

## Code facts

- Inference is one UNet call. Not a diffusion sampler.
- `coeff_t` scales x0 conversion. `model_t` is only the UNet timestep embedding.
- No CFG, no LoRA scale, no noise injection, no start/end schedule.
