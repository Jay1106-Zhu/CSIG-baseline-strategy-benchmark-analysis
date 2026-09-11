# HYPIR fusion v3 metadata

- date: 2026-09-11
- lq_dir: `D:\MyProjects\CSIG\baseline\input`
- h50_dir: `D:\MyProjects\CSIG\baseline\experiments\coeff_t_50\output\result`
- h200_dir: `D:\MyProjects\CSIG\baseline\experiments\coeff_t_200\output\result`
- cases: case1, case2, case3, case4, case5
- pyramid_levels: 3
- high: L0 = G0 - expand(G1); ~1-2 px (finest Laplacian)
- mid: expand(L1)+expand(L2); ~4-8 px after two/three 2x blurs
- low: expand(G3); content coarser than ~1/8 resolution
- chroma: LQ YCrCb Cr/Cb
- inference: none
- does not write fusion_v1 / fusion_v2 / texture_selective
