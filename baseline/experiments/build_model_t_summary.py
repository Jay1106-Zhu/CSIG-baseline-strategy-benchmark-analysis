"""Build a long-format summary from the completed model_t experiments."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_TS = (200, 150, 100, 75, 50)
FIELDS = [
    "model_t",
    "Case",
    "LQ PSNR (dB)",
    "Output PSNR (dB)",
    "Delta PSNR (dB)",
    "PSNR Conclusion",
    "LQ SSIM",
    "Output SSIM",
    "Delta SSIM",
    "SSIM Conclusion",
    "LQ LPIPS-Alex",
    "Output LPIPS-Alex",
    "Delta LPIPS",
    "LPIPS Conclusion",
    "total_elapsed_seconds",
    "model_load_seconds",
    "Output Path",
]


def _metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("-") and ":" in line:
            key, value = line[1:].split(":", 1)
            result[key.strip()] = value.strip()
    return result


def build_summary(output_path: Path = ROOT / "model_t_sweep_summary.csv") -> Path:
    rows: list[dict[str, str]] = []
    for model_t in MODEL_TS:
        exp_dir = ROOT / f"model_t_{model_t}"
        metrics_path = exp_dir / "evaluation_metrics.csv"
        metadata = _metadata(exp_dir / "experiment_metadata.md")
        with metrics_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                row["model_t"] = str(model_t)
                row["total_elapsed_seconds"] = metadata.get("total_elapsed_seconds", "") if row["Case"] == "Average" else ""
                row["model_load_seconds"] = metadata.get("model_load_seconds", "") if row["Case"] == "Average" else ""
                rows.append({field: row.get(field, "") for field in FIELDS})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


if __name__ == "__main__":
    print(build_summary())
