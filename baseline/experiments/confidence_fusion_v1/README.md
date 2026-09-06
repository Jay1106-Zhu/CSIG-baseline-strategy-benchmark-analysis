# confidence_fusion_v1

Offline prototype for spatially adaptive blending of already generated HYPIR outputs. It uses only validation cases 1-5 and does not modify HYPIR, train LoRA, download models, or run new diffusion inference.

## Contents

- `confidence_maps/`: viridis visualizations of the LQ-derived structural reliability heuristic.
- `masks/`: per-case median-threshold high/low confidence masks.
- `fusion/fixed_50_50/`: fixed blend of HYPIR-200 and HYPIR-50.
- `fusion/confidence_200_50/`: confidence-weighted HYPIR-200/HYPIR-50 blend.
- `fusion/confidence_200_lq/`: confidence-weighted HYPIR-200/LQ conservative negative control.
- `comparison/`: seven-panel LQ, both source outputs, all fusion methods, and GT; error-map panels are also included.
- `evaluation_metrics.csv`: per-case and Average PSNR, SSIM, LPIPS-Alex, and deltas versus LQ.
- `region_metrics.csv`: high/low confidence L1, L2 RMSE, equivalent PSNR, and gradient difference to GT.
- `region_changes_vs_lq.csv`: per-region pixel/gradient change from LQ.
- `experiment_metadata.md`: provenance, algorithm, equations, and sanity checks.
- `confidence_fusion_v1_report.md`: evidence-based interpretation and Q1-Q10 answers.

LPIPS is lower-is-better. The confidence map is a structural reliability heuristic only; it must not be interpreted as semantic or ground-truth confidence.
