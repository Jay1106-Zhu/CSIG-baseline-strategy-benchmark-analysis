# Structure-Anchored Local Restoration v1 Metadata

- date: 2026-09-02
- status: completed
- scope: validation case1-case5; offline fusion; no new diffusion inference
- input_lq: `baseline\input`
- input_gt: `csig_dataset\验证集` (post-inference evaluation only)
- input_hypir_200: `baseline\experiments\coeff_t_200\output\result`
- input_hypir_50: `baseline\experiments\coeff_t_50\output\result`
- output_dir: `baseline\experiments\structure_local_restoration_v1`
- strategies: `structure_guard, texture_selective, blurred_texture`
- weight_range: `[0.0, 0.60]`; all constants are fixed across cases
- feature_source: LQ only (quarter-resolution Sobel edge, local standard deviation texture, inverse Laplacian blur proxy)
- fusion_formula: `I_out(x)=w(x)*I_HYPIR(x)+(1-w(x))*I_LQ(x)`
- HYPIR sources: existing coeff_t=200 and coeff_t=50 PNGs; no repeated diffusion inference
- LPIPS: Alex network, evaluated after uniform max-side resize to 1024px for memory-bounded comparison
- local_regions: LQ-only top-20-percent strong edge, blurred texture, textured non-edge, and remainder

## Strategies

1. `structure_guard`: `0.04 + (1-edge)*(0.28 + 0.08*texture)`. Strong edges receive the least diffusion.
2. `texture_selective`: `0.05 + 0.46*texture*(0.35 + 0.65*(1-edge))`. Texture gets more diffusion only away from strong contours.
3. `blurred_texture`: `0.05 + (0.40*blur*texture + 0.08*texture)*(1-edge)`. Blurred textured areas may receive more diffusion; flat areas and edges remain conservative.

GT is never passed to feature or weight computation. It is used only to measure errors after all fusion images are written.
