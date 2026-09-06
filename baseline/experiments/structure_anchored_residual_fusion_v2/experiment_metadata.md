# Structure-Anchored Residual Fusion v2 metadata

- date: 2026-09-02
- status: completed
- scope: validation case1-case5; offline fusion only
- lq_dir: `D:\MyProjects\CSIG\baseline\input`
- gt_dir: `D:\MyProjects\CSIG\csig_dataset\验证集` (evaluation only)
- h200_dir: `D:\MyProjects\CSIG\baseline\experiments\coeff_t_200\output\result`
- h50_dir: `D:\MyProjects\CSIG\baseline\experiments\coeff_t_50\output\result`
- texture_selective_dir: `D:\MyProjects\CSIG\baseline\experiments\texture_weight_sweep_v2\fusion\texture_selective`
- output_dir: `D:\MyProjects\CSIG\baseline\experiments\structure_anchored_residual_fusion_v2`
- formula: `F = LQ + gate * (HYPIR-200 - LQ)`
- feature_source: LQ Sobel gradient, Canny edge proximity, local variance, inverse Laplacian blur proxy
- residual checks: H200 residual edge novelty and LQ/H200 gradient direction alignment
- fixed gates: strong=0.020, weak=0.100, near=0.065, non-edge=0.035; max=0.160
- GT usage: post-fusion PSNR/SSIM/LPIPS and regional error only; never used for masks or weights
- inference: no model or LoRA training; no new HYPIR/SD inference
