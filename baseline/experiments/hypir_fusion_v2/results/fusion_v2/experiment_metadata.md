# HYPIR fusion v2 metadata

- date: 2026-09-11
- lq_dir: `D:\MyProjects\CSIG\baseline\input`
- h50_dir: `D:\MyProjects\CSIG\baseline\experiments\coeff_t_50\output\result`
- h200_dir: `D:\MyProjects\CSIG\baseline\experiments\coeff_t_200\output\result`
- gt_dir: `D:\MyProjects\CSIG\csig_dataset\验证集` (metrics only; never used in the mask)
- cases: case1, case2, case3, case4, case5
- schemes: A, B
- scene_alpha: {'text': 0.25, 'book': 0.3, 'bird': 0.12, 'plant': 0.08, 'clock': 0.3}
- alpha_overrides: {'text': 0.25, 'book': 0.3, 'bird': 0.12, 'plant': 0.08, 'clock': 0.3}
- struct_blur_sigma: 1.5
- residual_blur_sigma: 8.0
- residual_percentiles: 1.0-99.0
- formula: `out_Y = base_Y + alpha * M_final * (H200_Y - base_Y)`
- M_struct: v1 1/4 Sobel cosine * magnitude similarity
- conf: `1 - percentile_norm(GaussianBlur(mean_c |H200-LQ|, σ), 1–99)`
- groups A/B: `M_final = M_struct` (A is original fusion_v1; B is the same formula)
- group C: `M_final = M_struct * conf`
- group D: `M_final = M_struct * conf^2`
- extra gamma=0.5: `M_final = M_struct * conf^0.5`
- chroma: LQ YCrCb Cr/Cb
- inference: none; existing HYPIR PNGs only
- lpips: Alex max-side 1024, enabled=True
- does not write hypir_fusion_v1/results/

## Scene notes

- text: 中文文字; alpha=0.25
- book: 书脊文字; alpha=0.30
- bird: 鸟/水面; recommended 0.10-0.15, default 0.12
- plant: 密集绿植; recommended 0.05-0.10, default 0.08
- clock: 钟表; alpha=0.30

## MASK_GROUPS

- A: key=struct gamma=None formula=`M_struct`
- B: key=struct gamma=None formula=`M_struct`
- C: key=conf gamma=1.0 formula=`M_struct * conf`
- D: key=conf2 gamma=2.0 formula=`M_struct * conf^2`
