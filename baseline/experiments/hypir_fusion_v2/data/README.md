# Optional local inputs

Put images here if you want the experiment to read local folders instead of the project defaults:

```text
LQ/     case1.png ...   (or case1_lq.jpg)
H50/    matching HYPIR-50 outputs
H200/   matching HYPIR-200 outputs
```

If these directories are empty, `experiment.py` falls back to:

- `baseline/input/`
- `baseline/experiments/coeff_t_50/output/result/`
- `baseline/experiments/coeff_t_200/output/result/`

Do not copy the 4K validation set here unless you are running a detached copy of the experiment.
