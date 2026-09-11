# NO-GO multi-band

# HYPIR fusion v3 summary

- date: 2026-09-11
- pyramid levels: 3
- high: L0 = G0 - expand(G1); ~1-2 px (finest Laplacian)
- mid: expand(L1)+expand(L2); ~4-8 px after two/three 2x blurs
- low: expand(G3); content coarser than ~1/8 resolution
- LPIPS: Alex max-side 1024, computed=True

## Average metrics

| method | PSNR | SSIM | LPIPS_1024 | vs 28.48 |
|---|---:|---:|---:|---|
| LQ | 28.034420 | 0.777699 | 0.204318 | ref |
| HYPIR-50 | 28.257533 | 0.776935 | 0.162039 | ref |
| HYPIR-200 | 24.528587 | 0.684926 | 0.159664 | ref |
| texture_selective_h200 | 28.480280 | 0.781421 | 0.164549 | anchor |
| fusion_v1_A | 28.460407 | 0.781670 | 0.188585 | ref |
| multi_B | 26.604373 | 0.706967 | 0.170597 | no |
| multi_C | 26.579072 | 0.709848 | 0.151134 | no |
| multi_D01 | 26.492912 | 0.708780 | 0.148841 | no |
| multi_D02 | 26.388679 | 0.707223 | 0.147331 | no |

## Per-case PSNR

| case | LQ | HYPIR-200 | multi_B | multi_C | multi_D01 | multi_D02 | texture_selective_h200 |
|---|---|---|---|---|---|---|---|
| case1 | 32.0288 | 29.3353 | 30.4779 | 30.4854 | 30.4362 | 30.3773 | 32.1545 |
| case2 | 28.1894 | 24.2140 | 26.7241 | 26.6833 | 26.5809 | 26.4498 | 29.2414 |
| case3 | 35.6221 | 28.8654 | 33.2277 | 32.9279 | 32.7624 | 32.5650 | 35.6301 |
| case4 | 18.0562 | 16.3524 | 17.3558 | 17.3055 | 17.2548 | 17.1965 | 18.0199 |
| case5 | 26.2756 | 23.8758 | 25.2364 | 25.4933 | 25.4303 | 25.3548 | 27.3556 |

## Crops

| case | crop | method | PSNR | SSIM |
|---|---|---|---:|---:|
| case1 | text | LQ | 25.8140 | 0.7957 |
| case1 | text | H50 | 25.5237 | 0.8041 |
| case1 | text | H200 | 21.3725 | 0.6699 |
| case1 | text | B | 23.1017 | 0.6725 |
| case1 | text | C | 22.9404 | 0.6869 |
| case1 | text | D01 | 22.8479 | 0.6862 |
| case1 | text | D02 | 22.7488 | 0.6853 |
| case1 | text | texture | 25.5772 | 0.7876 |
| case1 | text | fusion_A | 25.8036 | 0.7966 |
| case2 | spine | LQ | 27.2461 | 0.8279 |
| case2 | spine | H50 | 28.3806 | 0.8467 |
| case2 | spine | H200 | 22.1558 | 0.7654 |
| case2 | spine | B | 25.6939 | 0.7802 |
| case2 | spine | C | 25.4489 | 0.7839 |
| case2 | spine | D01 | 25.2663 | 0.7830 |
| case2 | spine | D02 | 25.0572 | 0.7817 |
| case2 | spine | texture | 28.7619 | 0.8544 |
| case2 | spine | fusion_A | 28.6061 | 0.8506 |
| case3 | bird | LQ | 34.9938 | 0.8649 |
| case3 | bird | H50 | 34.4053 | 0.8522 |
| case3 | bird | H200 | 30.8277 | 0.7912 |
| case3 | bird | B | 34.0933 | 0.8286 |
| case3 | bird | C | 33.9544 | 0.8277 |
| case3 | bird | D01 | 33.8609 | 0.8267 |
| case3 | bird | D02 | 33.7443 | 0.8254 |
| case3 | bird | texture | 35.0280 | 0.8656 |
| case3 | bird | fusion_A | 35.0601 | 0.8657 |
| case3 | water | LQ | 46.6415 | 0.9887 |
| case3 | water | H50 | 42.1484 | 0.9745 |
| case3 | water | H200 | 34.4847 | 0.8452 |
| case3 | water | B | 37.6978 | 0.8741 |
| case3 | water | C | 37.5338 | 0.8722 |
| case3 | water | D01 | 37.4385 | 0.8710 |
| case3 | water | D02 | 37.2946 | 0.8688 |
| case3 | water | texture | 45.9460 | 0.9869 |
| case3 | water | fusion_A | 46.0769 | 0.9878 |
| case3 | mud | LQ | 45.4147 | 0.9848 |
| case3 | mud | H50 | 39.6949 | 0.9662 |
| case3 | mud | H200 | 31.5072 | 0.7897 |
| case3 | mud | B | 37.5498 | 0.8608 |
| case3 | mud | C | 37.1022 | 0.8556 |
| case3 | mud | D01 | 36.8689 | 0.8519 |
| case3 | mud | D02 | 36.6066 | 0.8475 |
| case3 | mud | texture | 43.5954 | 0.9800 |
| case3 | mud | fusion_A | 44.6544 | 0.9830 |
| case4 | fish | LQ | 17.3633 | 0.2721 |
| case4 | fish | H50 | 17.3852 | 0.2671 |
| case4 | fish | H200 | 15.7665 | 0.1927 |
| case4 | fish | B | 16.6786 | 0.2082 |
| case4 | fish | C | 16.6458 | 0.2086 |
| case4 | fish | D01 | 16.6056 | 0.2077 |
| case4 | fish | D02 | 16.5580 | 0.2062 |
| case4 | fish | texture | 17.3317 | 0.2583 |
| case4 | fish | fusion_A | 17.3751 | 0.2725 |
| case4 | leaf | LQ | 17.3139 | 0.2637 |
| case4 | leaf | H50 | 17.3877 | 0.2674 |
| case4 | leaf | H200 | 16.1898 | 0.2096 |
| case4 | leaf | B | 16.8447 | 0.2181 |
| case4 | leaf | C | 16.8348 | 0.2206 |
| case4 | leaf | D01 | 16.8065 | 0.2200 |
| case4 | leaf | D02 | 16.7724 | 0.2193 |
| case4 | leaf | texture | 17.3479 | 0.2626 |
| case4 | leaf | fusion_A | 17.3284 | 0.2645 |
| case4 | yellow_flower | LQ | 16.0335 | 0.2022 |
| case4 | yellow_flower | H50 | 16.0705 | 0.2057 |
| case4 | yellow_flower | H200 | 14.7592 | 0.1855 |
| case4 | yellow_flower | B | 15.2128 | 0.1805 |
| case4 | yellow_flower | C | 15.2006 | 0.1840 |
| case4 | yellow_flower | D01 | 15.1803 | 0.1845 |
| case4 | yellow_flower | D02 | 15.1563 | 0.1848 |
| case4 | yellow_flower | texture | 16.0309 | 0.2054 |
| case4 | yellow_flower | fusion_A | 16.0441 | 0.2034 |
| case5 | clock | LQ | 26.3877 | 0.8466 |
| case5 | clock | H50 | 27.9907 | 0.8819 |
| case5 | clock | H200 | 25.0302 | 0.7880 |
| case5 | clock | B | 25.3820 | 0.7744 |
| case5 | clock | C | 25.7457 | 0.7997 |
| case5 | clock | D01 | 25.7109 | 0.8001 |
| case5 | clock | D02 | 25.6706 | 0.7998 |
| case5 | clock | texture | 27.3114 | 0.8632 |
| case5 | clock | fusion_A | 27.1063 | 0.8573 |
