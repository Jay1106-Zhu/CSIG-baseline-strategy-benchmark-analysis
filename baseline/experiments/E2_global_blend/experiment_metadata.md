# E2 Global Alpha Blend metadata

- cases: case1, case2, case3, case4, case5
- alphas: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
- hypir_dir: `D:\MyProjects\CSIG\baseline\experiments\coeff_t_50\output\result`
- fidelity_dir: `D:\MyProjects\CSIG\baseline\experiments\E1_fidelity\swinir_car_jpeg40\output`
- gt_dir: `D:\MyProjects\CSIG\csig_dataset\验证集`
- output_root: `D:\MyProjects\CSIG\baseline\experiments\E2_global_blend`
- device: `cuda`
- total_wall_time_seconds: 577.980
- formula: I = alpha * HYPIR-50 + (1-alpha) * Fidelity
- processing: clip to [0,255], round, cast uint8, save RGB PNG
- reuse_outputs: True
- inference: offline pixel blend only; no model inference, training, TTA, ensemble, or parameter changes
- metrics: E0-compatible PSNR, SSIM, LPIPS-Alex (max-side 1024)
