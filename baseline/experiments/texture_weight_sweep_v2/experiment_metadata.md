# Texture-region HYPIR weight sweep metadata

- date: 2026-09-02
- status: completed
- scope: validation case1-case5; offline fusion only
- input_lq: `D:\MyProjects\CSIG\baseline\input`
- input_gt: `D:\MyProjects\CSIG\csig_dataset\验证集` (post-inference evaluation only)
- input_hypir_200: `D:\MyProjects\CSIG\baseline\experiments\coeff_t_200\output\result`
- output_dir: `D:\MyProjects\CSIG\baseline\experiments\texture_weight_sweep_v2`
- candidates: `0.25, 0.40, 0.55, 0.70, 0.85, 1.00`
- baseline: `texture_selective_h200` from Structure-Anchored Local Restoration v1
- feature_source: LQ only; same quarter-resolution edge/texture/blur features as v1
- texture_region: `texture >= P80` and `edge < P80`; this is the v1 `textured_non_edge` mask and includes its `blurred_texture` subset
- fusion: `I_out=w*I_HYPIR-200+(1-w)*I_LQ`
- invariant: outside the texture region, including all `strong_edge` pixels, the v1 `texture_selective` weight map is copied exactly
- hypir_inference: none; existing HYPIR-200 PNGs are reused
- LPIPS: Alex network after uniform max-side resize to 1024px

The candidate value is a direct blend weight in the texture region. It is not a
new HYPIR parameter and does not alter edge protection, feature thresholds,
training, architecture, or the HYPIR source tree.
