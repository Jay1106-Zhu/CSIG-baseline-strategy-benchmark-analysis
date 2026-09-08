# DiffIR Motion Deblurring Baseline Report

## 1 Experiment Goal

This is the narrowed B2 baseline bake-off requested for the CSIG project. The
run tests the official pretrained DiffIR Motion Deblurring model and compares
it with the existing Identity and HYPIR-50 references on the five validation
cases. It is a baseline characterization, not a training or optimization
experiment.

Only the following models were run:

| Model | Role |
| --- | --- |
| Identity | decoded LQ input floor/reference |
| HYPIR-50 | existing diffusion reference (`model_t=200`, `coeff_t=50`) |
| DiffIR-DiffIRS2 | motion-deblurring diffusion baseline (local runner) |
| DiffIROfficial | same DiffIRS2, loaded from official test yaml |

EVSSM and DiffBIR were intentionally not run after the scope was narrowed to
DiffIR Motion Deblurring. Therefore this report does not claim a winner among
all four models named in the original bake-off prompt.

## 2 Environment

- Python 3.11.16
- PyTorch 2.11.0+cu128, CUDA 12.8
- NVIDIA GeForce RTX 5080 Laptop GPU (15.92 GiB reported device memory)
- Five RGB validation LQ images, evaluated at their native dimensions
- Repository branch: `repo-hygiene-and-official-diffir`
- Published base snapshot: `50ae047`

The repository contains source, result tables, visual evidence, and manifests;
large model checkpoints remain outside the repository in the local model cache.

## 3 Models and Checkpoints

### DiffIR

- Official repository: `https://github.com/Zj-BinXia/DiffIR`
- Source commit: `293f86cdf313914ea0ffb2457ed032e4f1bd9dd2`
- Model: `DiffIRS2`
- Checkpoint: `Deblurring-DiffIRS2.pth`, loaded from `params_ema`
- SHA-256:
  `679fe3aae09a49a31517d1f04faed93dc85456050cde0df416a7ee4fe4297031`
- The official S1 checkpoint was also cached for provenance, but S2 is the
  checkpoint used for the reported inference.

### HYPIR and Identity

HYPIR-50 reuses the existing native-resolution outputs in
`baseline/experiments/coeff_t_50/output/result`. Identity is the decoded LQ
image itself. No HYPIR source, checkpoint, or prior experiment was modified.

## 4 Inference Settings

DiffIR was called with LQ only. Neither runner loads or passes GT to the
model. Both decode RGB, preserve native dimensions, and use DiffIRS2 with four
timesteps and seed `0` (`manual_seed: 0` in the official test yaml).

The original bakeoff runner hard-codes the network kwargs. The official-yaml
runner (`runners/run_diffir_official.py`) reads
`options/test_DiffIRS2_csig.yml`, which is a copy of official
`test_DiffIRS2.yml` with CSIG paths and the local `Deblurring-DiffIRS2.pth`.
`scale: 4` is CPEN PixelUnshuffle, not output upscaling. `window_size: 8`
only pads to a multiple of 8; official `DiffIR/test.py` has no 4K tile path.

Full-frame inference OOMed on every 4K case. Both runners then used Hann
`tile=512` / `overlap=128`. The five `outputs/diffir_official` PNGs are
SHA-256 identical to `outputs/diffir`. Quality metrics are therefore the
same; only wall-clock times differ. See
`logs/diffir_official_fullframe_oom.log`.

The benchmark reads GT only after all DiffIR outputs are present. PSNR and SSIM
use native resolution. LPIPS-Alex and DISTS use the same deterministic
aspect-preserving resize with maximum side 1024 for every model. MUSIQ, MANIQA,
and CLIPIQA are blank because `pyiqa` is not installed in this environment;
they were not approximated with a different implementation.

## 5 Quantitative Results

Average results from `results_average.csv`:

| Model | PSNR (dB) | SSIM | LPIPS-Alex | DISTS |
| --- | ---: | ---: | ---: | ---: |
| Identity | 28.0344 | 0.7777 | 0.2313 | 0.1798 |
| HYPIR-50 | **28.2575** | 0.7769 | **0.1808** | **0.1608** |
| DiffIR-DiffIRS2 | 27.7922 | 0.7753 | 0.2047 | 0.1790 |
| DiffIROfficial | 27.7922 | 0.7753 | 0.2047 | 0.1790 |

HYPIR-50 has the strongest average PSNR, LPIPS, and DISTS in this five-case
comparison. Identity has the highest average SSIM by a small margin. DiffIR is
between Identity and HYPIR on LPIPS and DISTS, but below both on average PSNR
and SSIM. These averages are descriptive only; no single weighted score or
champion claim is used.

## 6 Per-case Analysis

The per-case table is the authoritative source in `results_per_case.csv`.

| Case | Subject | PSNR leader | DiffIR observation |
| --- | --- | --- | --- |
| case1 | Chinese text | HYPIR-50 (32.1433) | 31.4169, slightly below Identity; LPIPS close to HYPIR |
| case2 | book/small text | HYPIR-50 (28.9763) | 28.4034, between HYPIR and Identity on PSNR |
| case3 | bird detail | Identity (35.6221) | 35.2261, second on PSNR and above HYPIR |
| case4 | dense foliage | HYPIR-50 (18.0846) | 17.9822, slightly below Identity; all methods are difficult here |
| case5 | clock geometry | HYPIR-50 (27.3948) | 25.9324, below Identity and HYPIR |

The case-level pattern is mixed rather than a universal DiffIR gain. DiffIR
does not exceed HYPIR-50 on any average metric, and it does not lead PSNR on
any case. Its strongest relative case is case3, where it remains behind
Identity but exceeds HYPIR-50 in PSNR.

## 7 Visual and Hallucination Analysis

`visuals/full_comparison.jpg` and `visuals/detail_crops.jpg` use fixed crop
coordinates shared by Identity, HYPIR-50, and DiffIR. The contact sheets include
LQ, GT, Identity, HYPIR, and DiffIR in that order. The crops cover Chinese
text, small print, bird detail, dense foliage, and clock structure.

The images support the quantitative conclusion: DiffIR is a valid restoration
attempt, but this run does not provide evidence that it reconstructs text or
clock geometry more faithfully than HYPIR-50. The five cases are too small to
make a robust hallucination-rate estimate, so visual observations are kept as
qualitative evidence rather than converted into a score.

## 8 4K Runtime and VRAM

The original DiffIR runner took 47.39 to 65.36 seconds per case (mean 55.05).
The official-yaml runner took 39.78 to 62.49 seconds (mean 50.11). All five
cases required the same OOM retry and `512/128` tiled execution. The manifest's
`peak_vram_gb` is the CUDA allocator high-water mark after the failed full-frame
probe (about 20.5 to 20.64 GB), not physical resident VRAM; the GPU reports
15.92 GiB device memory. This distinction is recorded so the allocator value is
not misread as available hardware capacity.

Identity is a zero-runtime reference. HYPIR runtime/VRAM was not rerun in this
phase, so those fields remain unset rather than being inferred from old logs.

## 9 Pareto Analysis

Under this fixed protocol, HYPIR-50 dominates the tested learned baselines on
average PSNR, LPIPS, and DISTS, while Identity has the best average SSIM. DiffIR
offers an independent official motion-deblurring diffusion path and exact
same-resolution outputs, but it is not on the leading quantitative Pareto
frontier for this five-case sample. Engineering-wise, DiffIR is reproducible
with deterministic tiling but is comparatively expensive at roughly 45 seconds
per 4K image.

## 10 Backbone Recommendation

For the requested comparison, retain HYPIR-50 as the current diffusion
reference and keep DiffIR as a documented alternative motion-deblurring
baseline. The present data do not justify selecting DiffIR as the primary
backbone or declaring it superior to HYPIR. Identity remains useful as a
fidelity floor and as a sanity check for any later enhancement method.

This recommendation is limited to the tested models, checkpoint, and five
validation cases. EVSSM and DiffBIR were not run and are intentionally outside
the claim.

## 11 Next Architecture Recommendation

The next step should be determined from the broader project objective and a
larger, held-out evaluation set. Within this run, the evidence supports:

1. Preserve HYPIR-50 and Identity as fixed references.
2. Use DiffIR only as an independently reproducible motion-deblurring baseline
   or ablation target.
3. Do not introduce blending, routing, sharpening, or GT-driven parameter
   choices based on these five images.
4. If DiffIR is revisited, first investigate degradation/checkpoint mismatch
   and memory-efficient inference on held-out data; do not infer that its lower
   score is caused by a single tunable strength parameter.

## Reproduction and Artifact Index

- Original runner: `baseline_bakeoff/runners/run_diffir.py`
- Official yaml runner: `baseline_bakeoff/runners/run_diffir_official.py`
- Official yaml: `baseline_bakeoff/options/test_DiffIRS2_csig.yml`
- Official outputs: `baseline_bakeoff/outputs/diffir_official/`
- Official inference manifest: `baseline_bakeoff/logs/diffir_official_inference_manifest.json`
- Full-frame OOM note: `baseline_bakeoff/logs/diffir_official_fullframe_oom.log`
- Benchmark: `baseline_bakeoff/benchmark.py`
- Run manifest: `baseline_bakeoff/run_manifest.json`
- DiffIR inference manifest: `baseline_bakeoff/logs/diffir_inference_manifest.json`
- Per-case metrics: `baseline_bakeoff/results_per_case.csv`
- Average metrics: `baseline_bakeoff/results_average.csv`
- Full comparison: `baseline_bakeoff/visuals/full_comparison.jpg`
- Detail crops: `baseline_bakeoff/visuals/detail_crops.jpg`
