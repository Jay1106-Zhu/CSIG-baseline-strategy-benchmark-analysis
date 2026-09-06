# Adaptive H50/H200 Fusion v1 Metadata

- date: 2026-09-02
- scope: validation case1-case5; offline fusion only
- lq_dir: `D:\MyProjects\CSIG\baseline\input`
- gt_dir: `D:\MyProjects\CSIG\csig_dataset\验证集` (post-fusion metrics only)
- h50_dir: `D:\MyProjects\CSIG\baseline\experiments\coeff_t_50\output\result`
- h200_dir: `D:\MyProjects\CSIG\baseline\experiments\coeff_t_200\output\result`
- texture_selective_dir: `D:\MyProjects\CSIG\baseline\experiments\texture_weight_sweep_v2\fusion\texture_selective`
- formula: `F=(1-alpha)*H50+alpha*H200`
- alpha_formula: `clip(0.035 + 0.11*texture + 0.045*edge*(1-strong_edge) + 0.035*blur*texture*(1-strong_edge) - 0.18*strong_edge, 0, 0.30)`
- alpha_source: LQ-only Sobel gradient, local variance, blur proxy, and top-20-percent strong-edge mask
- gt_usage: evaluation only; never used in alpha or thresholds
- inference: no diffusion inference, training, or LoRA
- lpips: Alex with max-side 1024px resize, matching prior local-restoration metrics
