# E1 SwinIR checkpoint metadata

- model: SwinIR-M color JPEG compression artifact reduction
- official task: `color_jpeg_car`
- official JPEG quality: 40
- official checkpoint filename: `006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth`
- official source repository: `https://github.com/JingyunLiang/SwinIR`
- official README source: `https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/README.md`
- official inference source: `https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/main_test_swinir.py`
- official release URL: `https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth`
- local path: `baseline/experiments/E1_fidelity/swinir_car_jpeg40/checkpoint/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth`
- file size: `102873665` bytes
- SHA-256: `265c18d8809aaca0cd97a6283bee0ed1883ab88395e456381264cac2bb7b5867`
- top-level checkpoint keys: `params`
- state-dict key used for loading: `params`
- state-dict tensor count: 544
- state-dict dtype: float32
- strict compatibility check: passed; all 544 keys matched

## Official model configuration

- `upscale=1`
- `in_chans=3`
- `img_size=126`
- `window_size=7`
- `img_range=255.0`
- `depths=[6,6,6,6,6,6]`
- `embed_dim=180`
- `num_heads=[6,6,6,6,6,6]`
- `mlp_ratio=2`
- `upsampler=''`
- `resi_connection='1conv'`

## Download and verification

The checkpoint was downloaded through the official GitHub release URL using the browser-access workflow on 2026-09-03. It was copied into the project-owned E1 checkpoint directory before verification. No other model checkpoint was downloaded.
