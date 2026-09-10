# HYPIR fusion v1 metadata

- date: 2026-09-10
- lq_dir: `D:\MyProjects\CSIG\baseline\input`
- h50_dir: `D:\MyProjects\CSIG\baseline\experiments\coeff_t_50\output\result`
- h200_dir: `D:\MyProjects\CSIG\baseline\experiments\coeff_t_200\output\result`
- gt_dir: `D:\MyProjects\CSIG\csig_dataset\验证集` (metrics only; never used in the mask)
- cases: case1, case2, case3, case4, case5
- schemes: A, B
- scene_alpha: {'text': 0.25, 'book': 0.3, 'bird': 0.12, 'plant': 0.08, 'clock': 0.3}
- alpha_overrides: {'text': 0.25, 'book': 0.3, 'bird': 0.12, 'plant': 0.08, 'clock': 0.3}
- blur_sigma: 1.5
- formula: `out_Y = base_Y + alpha * mask * (H200_Y - base_Y)`
- mask: 1/4 Sobel cosine * magnitude similarity, Gaussian blur, upsample
- chroma: LQ YCrCb Cr/Cb
- inference: none; existing HYPIR PNGs only
- lpips: Alex max-side 1024, enabled=True

## Scene notes

- text: 中文文字; alpha=0.25
- book: 书脊文字; alpha=0.30
- bird: 鸟/水面; recommended 0.10-0.15, default 0.12
- plant: 密集绿植; recommended 0.05-0.10, default 0.08
- clock: 钟表; alpha=0.30
