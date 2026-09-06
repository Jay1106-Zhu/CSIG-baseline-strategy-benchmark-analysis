"""Evaluate coeff_t sweep outputs and create per-experiment triptychs."""

from __future__ import annotations

import sys
from pathlib import Path

import lpips
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline import compare_images, evaluate_metrics  # noqa: E402


COEFF_TS = (200, 150, 100, 75, 50)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for coeff_t in COEFF_TS:
        exp_dir = PROJECT_ROOT / "baseline" / "experiments" / f"coeff_t_{coeff_t}"
        output_dir = exp_dir / "output" / "result"
        csv_path = exp_dir / "evaluation_metrics.csv"
        cases = evaluate_metrics.match_evaluation_cases(
            PROJECT_ROOT / "baseline" / "input",
            PROJECT_ROOT / "csig_dataset" / "验证集",
            output_dir,
        )
        rows = []
        for case in cases:
            lq, gt, output = evaluate_metrics.validate_triplet(case.lq_path, case.gt_path, case.output_path)
            rows.append(evaluate_metrics._metric_row(case, lq, gt, output, model, device))
        average = evaluate_metrics._average(rows, csv_path)
        evaluate_metrics.write_csv(rows, average, csv_path)
        comparison_dir = exp_dir / "comparison"
        result = compare_images.process_comparisons(
            PROJECT_ROOT / "baseline" / "input",
            output_dir,
            comparison_dir,
            ground_truth_dir=PROJECT_ROOT / "csig_dataset" / "验证集",
        )
        if result.generated != 5 or result.skipped_invalid or result.skipped_missing_ground_truth:
            raise RuntimeError(
                f"coeff_t={coeff_t}: expected 5 comparisons, got {result.generated}; "
                f"invalid={result.skipped_invalid}, missing_gt={result.skipped_missing_ground_truth}"
            )
        print(
            f"coeff_t={coeff_t}: PSNR={average.output_psnr:.6f}, "
            f"SSIM={average.output_ssim:.6f}, LPIPS={average.output_lpips:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
