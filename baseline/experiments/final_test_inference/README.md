# Final test inference wrapper

Frozen method: `texture_selective_h200`. Formula is imported from `structure_local_restoration.py` and asserted unchanged.

This round: **dry-run case1 only**. Do not start 100 images until the operator confirms.

```powershell
# inventory
& .\.conda\python.exe baseline\experiments\final_test_inference\run_final_test.py --root . --mode inventory

# dry-run (test case1 only, not validation case1)
& .\.conda\python.exe baseline\experiments\final_test_inference\run_final_test.py --root . --mode dry-run

# 100 images — DO NOT RUN until dry-run is signed off
& .\.conda\python.exe baseline\experiments\final_test_inference\run_final_test.py --root . --mode full --confirm-full
```

HYPIR is invoked with explicit `--upscale 1 --model_t 200 --coeff_t 200 --seed 231 --captioner empty`.

- dry-run output: `baseline/experiments/final_test_inference/dry_run/output/case1.jpg`
- official output (after confirm): `baseline/experiments/final_test_inference/final_submission/output_dir/case{k}.jpg`
