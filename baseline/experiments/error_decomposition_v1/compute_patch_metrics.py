"""Per-tile PSNR / SSIM / LPIPS for non-overlapping 256 patches. No HYPIR inference."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms.functional import pil_to_tensor

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CASES = tuple(f"case{i}" for i in range(1, 6))
TILE = 256
PATHS = {
    "lq": ROOT / "baseline" / "input",
    "gt": ROOT / "csig_dataset" / "验证集",
    "h200": ROOT / "baseline" / "experiments" / "coeff_t_200" / "output" / "result",
    "h50": ROOT / "baseline" / "experiments" / "coeff_t_50" / "output" / "result",
}


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(gt, pred, data_range=255))


def ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(structural_similarity(gt, pred, channel_axis=2, data_range=255))


class LpipsAlex:
    def __init__(self) -> None:
        import lpips

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net="alex").to(self.device).eval()

    def scores(self, preds: list[np.ndarray], gt: np.ndarray) -> list[float]:
        def tensor(array: np.ndarray) -> torch.Tensor:
            image = Image.fromarray(array)
            return pil_to_tensor(image).unsqueeze(0).to(device=self.device, dtype=torch.float32).div(127.5).sub(1.0)

        with torch.inference_mode():
            prediction = torch.cat([tensor(p) for p in preds], dim=0)
            target = tensor(gt).expand(len(preds), -1, -1, -1)
            values = self.model(prediction, target).reshape(-1).detach().cpu().tolist()
        return [float(v) for v in values]


def tiles(shape: tuple[int, int, int]) -> list[tuple[int, int, int, int]]:
    h, w = shape[:2]
    boxes = []
    for y0 in range(0, h - TILE + 1, TILE):
        for x0 in range(0, w - TILE + 1, TILE):
            boxes.append((x0, y0, x0 + TILE, y0 + TILE))
    return boxes


def load_case(case: str) -> dict[str, np.ndarray]:
    return {
        "lq": load_rgb(PATHS["lq"] / f"{case}_lq.jpg"),
        "gt": load_rgb(PATHS["gt"] / f"{case}_gt.jpg"),
        "h50": load_rgb(PATHS["h50"] / f"{case}_lq.png"),
        "h200": load_rgb(PATHS["h200"] / f"{case}_lq.png"),
    }


def main() -> None:
    lpips_eval = LpipsAlex()
    all_rows: list[dict[str, object]] = []
    for case in CASES:
        images = load_case(case)
        boxes = tiles(images["gt"].shape)
        grads = []
        for x0, y0, x1, y1 in boxes:
            lq = images["lq"][y0:y1, x0:x1].astype(np.float32)
            gt = images["gt"][y0:y1, x0:x1].astype(np.float32)
            gx = np.abs(np.diff(lq.mean(axis=2), axis=1, prepend=lq.mean(axis=2)[:, :1]))
            gy = np.abs(np.diff(lq.mean(axis=2), axis=0, prepend=lq.mean(axis=2)[:1, :]))
            g_lq = gx + gy
            gx = np.abs(np.diff(gt.mean(axis=2), axis=1, prepend=gt.mean(axis=2)[:, :1]))
            gy = np.abs(np.diff(gt.mean(axis=2), axis=0, prepend=gt.mean(axis=2)[:1, :]))
            g_gt = gx + gy
            grads.append(float(np.mean(np.abs(g_gt - g_lq))))
        q1, q2 = np.quantile(np.asarray(grads), [1.0 / 3.0, 2.0 / 3.0])
        print(f"{case}: {len(boxes)} tiles", flush=True)
        for (x0, y0, x1, y1), grad in zip(boxes, grads):
            gt = images["gt"][y0:y1, x0:x1]
            lq = images["lq"][y0:y1, x0:x1]
            h50 = images["h50"][y0:y1, x0:x1]
            h200 = images["h200"][y0:y1, x0:x1]
            lp = lpips_eval.scores([lq, h50, h200], gt)
            if grad <= q1:
                stratum = "low"
            elif grad <= q2:
                stratum = "mid"
            else:
                stratum = "high"
            all_rows.append(
                {
                    "case": case,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "stratum": stratum,
                    "lq_gt_gradient_difference": f"{grad:.6f}",
                    "psnr_lq": f"{psnr(gt, lq):.6f}",
                    "psnr_h50": f"{psnr(gt, h50):.6f}",
                    "psnr_h200": f"{psnr(gt, h200):.6f}",
                    "ssim_lq": f"{ssim(gt, lq):.6f}",
                    "ssim_h50": f"{ssim(gt, h50):.6f}",
                    "ssim_h200": f"{ssim(gt, h200):.6f}",
                    "lpips_lq": f"{lp[0]:.6f}",
                    "lpips_h50": f"{lp[1]:.6f}",
                    "lpips_h200": f"{lp[2]:.6f}",
                }
            )
    path = OUT / "patch_metrics_full.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_rows = []
    keys = [
        "psnr_lq",
        "psnr_h50",
        "psnr_h200",
        "ssim_lq",
        "ssim_h50",
        "ssim_h200",
        "lpips_lq",
        "lpips_h50",
        "lpips_h200",
    ]
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in all_rows:
        groups.setdefault((row["case"], row["stratum"]), []).append(row)
        groups.setdefault((row["case"], "all"), []).append(row)
        groups.setdefault(("Average", "all"), []).append(row)
    for (case, stratum), rows in sorted(groups.items()):
        rec = {"case": case, "stratum": stratum, "n": len(rows)}
        for key in keys:
            rec[key] = f"{np.mean([float(r[key]) for r in rows]):.6f}"
        rec["dpsnr_h50"] = f"{np.mean([float(r['psnr_h50']) - float(r['psnr_lq']) for r in rows]):.6f}"
        rec["dpsnr_h200"] = f"{np.mean([float(r['psnr_h200']) - float(r['psnr_lq']) for r in rows]):.6f}"
        rec["dssim_h50"] = f"{np.mean([float(r['ssim_h50']) - float(r['ssim_lq']) for r in rows]):.6f}"
        rec["dssim_h200"] = f"{np.mean([float(r['ssim_h200']) - float(r['ssim_lq']) for r in rows]):.6f}"
        rec["dlpips_h50"] = f"{np.mean([float(r['lpips_h50']) - float(r['lpips_lq']) for r in rows]):.6f}"
        rec["dlpips_h200"] = f"{np.mean([float(r['lpips_h200']) - float(r['lpips_lq']) for r in rows]):.6f}"
        summary_rows.append(rec)
    summary_path = OUT / "patch_metrics_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    labels_path = OUT / "e1_labels.csv"
    if labels_path.is_file():
        labels = list(csv.DictReader(labels_path.open(encoding="utf-8")))
        index = {(r["case"], int(r["x0"]), int(r["y0"])): r for r in all_rows}
        e1_rows = []
        for lab in labels:
            key = (lab["case"], int(lab["x0"]), int(lab["y0"]))
            metrics = index[key]
            e1_rows.append(
                {
                    **{k: lab[k] for k in ("case", "patch_id", "stratum", "x0", "y0", "x1", "y1", "error_type", "structure_preserved", "new_object", "plant_morphology", "confidence", "notes")},
                    **{k: metrics[k] for k in keys},
                    "dpsnr_h50": f"{float(metrics['psnr_h50']) - float(metrics['psnr_lq']):.6f}",
                    "dpsnr_h200": f"{float(metrics['psnr_h200']) - float(metrics['psnr_lq']):.6f}",
                    "dssim_h50": f"{float(metrics['ssim_h50']) - float(metrics['ssim_lq']):.6f}",
                    "dssim_h200": f"{float(metrics['ssim_h200']) - float(metrics['ssim_lq']):.6f}",
                    "dlpips_h50": f"{float(metrics['lpips_h50']) - float(metrics['lpips_lq']):.6f}",
                    "dlpips_h200": f"{float(metrics['lpips_h200']) - float(metrics['lpips_lq']):.6f}",
                }
            )
        e1_path = OUT / "e1_patch_metrics.csv"
        with e1_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(e1_rows[0].keys()))
            writer.writeheader()
            writer.writerows(e1_rows)
        print("wrote", e1_path)
    print("wrote", path)
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
