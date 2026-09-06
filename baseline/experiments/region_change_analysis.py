"""Measure how much each method changes LQ within confidence regions."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "baseline" / "experiments" / "confidence_fusion_v1"
CASES = [f"case{i}" for i in range(1, 6)]
METHODS = ["LQ", "HYPIR-200", "HYPIR-50", "fixed_50_50", "confidence_200_50", "confidence_200_LQ"]

def rgb(path):
    with Image.open(path) as im: return np.asarray(im.convert("RGB"), dtype=np.float32).copy()
def grad(a):
    g = a.mean(2); gx = np.gradient(g, axis=1); gy = np.gradient(g, axis=0); return np.sqrt(gx*gx + gy*gy)
def path(case, method):
    if method == "LQ": return ROOT / "baseline/input" / f"{case}_lq.jpg"
    if method == "HYPIR-200": return ROOT / "baseline/experiments/coeff_t_200/output/result" / f"{case}_lq.png"
    if method == "HYPIR-50": return ROOT / "baseline/experiments/coeff_t_50/output/result" / f"{case}_lq.png"
    sub = {"fixed_50_50":"fixed_50_50", "confidence_200_50":"confidence_200_50", "confidence_200_LQ":"confidence_200_lq"}[method]
    return OUT / "fusion" / sub / f"{case}.png"

rows=[]
for case in CASES:
    lq = rgb(path(case, "LQ")); glq = grad(lq)
    hi = rgb(OUT / "confidence_maps" / f"{case}.png")[:,:,0]  # only for shape
    high = np.asarray(Image.open(OUT / "masks" / f"{case}_high.png"), dtype=np.uint8) > 127
    for method in METHODS:
        arr = rgb(path(case, method)); d = arr-lq; gd = grad(arr)-glq
        for region, mask in (("high_confidence", high), ("low_confidence", ~high)):
            pix, gp = d[mask], gd[mask]
            rows.append({"case":case,"method":method,"region":region,"pixels":int(mask.sum()),"change_L1":f"{np.mean(np.abs(pix)):.6f}","change_RMSE":f"{np.sqrt(np.mean(pix*pix)):.6f}","gradient_change":f"{np.mean(np.abs(gp)):.6f}"})
fields=["case","method","region","pixels","change_L1","change_RMSE","gradient_change"]
with (OUT/"region_changes_vs_lq.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    for m in METHODS:
        for region in ("high_confidence","low_confidence"):
            s=[r for r in rows if r["method"]==m and r["region"]==region]
            w.writerow({"case":"Average","method":m,"region":region,"pixels":int(np.mean([int(r["pixels"]) for r in s])),**{k:f"{np.mean([float(r[k]) for r in s]):.6f}" for k in fields[4:]}})
print(OUT/"region_changes_vs_lq.csv")
