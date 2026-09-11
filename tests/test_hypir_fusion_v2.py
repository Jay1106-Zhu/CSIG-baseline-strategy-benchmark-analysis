import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from baseline.experiments.hypir_fusion_v1.fusion import compute_structure_mask, rgb_to_y
from baseline.experiments.hypir_fusion_v2 import (
    MASK_GROUPS,
    combine_masks,
    compute_residual_confidence,
)
from baseline.experiments.hypir_fusion_v2.experiment import run_experiment


class ResidualConfidenceTests(unittest.TestCase):
    def test_confidence_is_low_where_h200_differs_from_lq(self):
        lq = np.full((96, 96, 3), 40, dtype=np.uint8)
        h200 = lq.copy()
        h200[20:70, 20:70] = 200
        conf, diagnostics = compute_residual_confidence(lq, h200, blur_sigma=0.0, return_diagnostics=True)
        self.assertEqual(conf.shape, (96, 96))
        self.assertTrue(np.isfinite(conf).all())
        self.assertGreaterEqual(float(conf.min()), 0.0)
        self.assertLessEqual(float(conf.max()), 1.0)
        interior = float(conf[30:60, 30:60].mean())
        background = float(conf[2:10, 2:10].mean())
        self.assertLess(interior, 0.35)
        self.assertGreater(background, 0.65)
        self.assertGreater(background - interior, 0.40)
        residual = diagnostics["residual"]
        self.assertGreater(float(residual[30:60, 30:60].mean()), float(residual[2:10, 2:10].mean()))

    def test_identical_images_keep_high_confidence(self):
        image = np.full((48, 64, 3), 90, dtype=np.uint8)
        conf = compute_residual_confidence(image, image.copy(), blur_sigma=0.0)
        self.assertGreater(float(conf.mean()), 0.99)

    def test_normalize_uses_percentile_and_stays_in_unit_interval(self):
        lq = np.zeros((32, 32, 3), dtype=np.uint8)
        h200 = lq.copy()
        h200[0, 0] = 255
        h200[16:20, 16:20] = 80
        conf, diagnostics = compute_residual_confidence(
            lq, h200, blur_sigma=0.0, low_percentile=1.0, high_percentile=99.0, return_diagnostics=True
        )
        self.assertTrue(np.all((conf >= 0.0) & (conf <= 1.0)))
        self.assertTrue(np.all((diagnostics["residual_norm"] >= 0.0) & (diagnostics["residual_norm"] <= 1.0)))
        self.assertTrue(np.allclose(conf, 1.0 - diagnostics["residual_norm"], atol=1e-5))


class CombineMaskTests(unittest.TestCase):
    def test_group_a_and_b_are_structure_only(self):
        self.assertEqual(MASK_GROUPS["A"]["gamma"], None)
        self.assertEqual(MASK_GROUPS["B"]["gamma"], None)
        self.assertEqual(MASK_GROUPS["A"]["key"], MASK_GROUPS["B"]["key"])
        struct = np.full((8, 8), 0.80, dtype=np.float32)
        conf = np.full((8, 8), 0.25, dtype=np.float32)
        combined = combine_masks(struct, conf, gamma=MASK_GROUPS["A"]["gamma"])
        self.assertTrue(np.allclose(combined, struct))

    def test_gamma_one_multiplies_structure_by_confidence(self):
        struct = np.full((8, 8), 0.80, dtype=np.float32)
        conf = np.full((8, 8), 0.50, dtype=np.float32)
        combined = combine_masks(struct, conf, gamma=1.0)
        self.assertTrue(np.allclose(combined, 0.40, atol=1e-5))

    def test_higher_gamma_penalizes_low_confidence_more(self):
        struct = np.ones((4, 4), dtype=np.float32)
        conf = np.full((4, 4), 0.40, dtype=np.float32)
        mild = combine_masks(struct, conf, gamma=0.5)
        mid = combine_masks(struct, conf, gamma=1.0)
        hard = combine_masks(struct, conf, gamma=2.0)
        self.assertGreater(float(mild.mean()), float(mid.mean()))
        self.assertGreater(float(mid.mean()), float(hard.mean()))
        self.assertTrue(np.allclose(hard, 0.16, atol=1e-5))

    def test_smooth_hallucination_drops_confidence_not_structure(self):
        # Shared bar: LQ and H200 have the same contour. H200 only changes
        # interior intensity, so Sobel agreement stays high while |H200-LQ|
        # is large — the fish-head interior case that v1 cannot suppress.
        lq = np.zeros((128, 128, 3), dtype=np.uint8)
        lq[:, 20:108] = 90
        h200 = lq.copy()
        h200[:, 20:108] = 200
        struct = compute_structure_mask(rgb_to_y(lq), rgb_to_y(h200))
        conf = compute_residual_confidence(lq, h200, blur_sigma=1.0)
        final = combine_masks(struct, conf, gamma=1.0)
        interior = (slice(40, 88), slice(40, 88))
        outside = (slice(0, 16), slice(0, 16))
        self.assertGreater(float(struct[interior].mean()), 0.70)
        self.assertLess(float(conf[interior].mean()), 0.20)
        self.assertGreater(float(conf[outside].mean()), 0.80)
        self.assertLess(float(final[interior].mean()), float(struct[interior].mean()) - 0.40)
        self.assertLess(float(final[interior].mean()), float(final[outside].mean()))


class ExperimentArtifactTests(unittest.TestCase):
    def test_run_experiment_writes_v2_masks_heatmaps_and_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dirs = {name: root / name for name in ("LQ", "H50", "H200", "GT")}
            for directory in dirs.values():
                directory.mkdir()
            lq = np.full((64, 80, 3), 40, dtype=np.uint8)
            lq[:, 20:28] = 180
            gt = lq.copy()
            h50 = np.clip(lq.astype(np.int16) + 8, 0, 255).astype(np.uint8)
            h200 = lq.copy()
            h200[:, 52:64] = 210
            h200[20:44, 8:24] = 160
            for name, array in (("LQ", lq), ("H50", h50), ("H200", h200), ("GT", gt)):
                Image.fromarray(array).save(dirs[name] / "case1.png")
            out = root / "results" / "fusion_v2"
            result = run_experiment(
                dirs["LQ"],
                dirs["H50"],
                dirs["H200"],
                out,
                gt_dir=dirs["GT"],
                compute_lpips=False,
                cases=("case1",),
            )
            self.assertEqual(result["cases"], ["case1"])
            self.assertTrue((out / "fusion" / "A_struct" / "case1.png").is_file())
            self.assertTrue((out / "fusion" / "B_struct" / "case1.png").is_file())
            self.assertTrue((out / "fusion" / "A_conf" / "case1.png").is_file())
            self.assertTrue((out / "fusion" / "B_conf" / "case1.png").is_file())
            self.assertTrue((out / "fusion" / "A_conf2" / "case1.png").is_file())
            self.assertTrue((out / "fusion" / "A_conf05" / "case1.png").is_file())
            self.assertTrue((out / "masks" / "case1_struct.png").is_file())
            self.assertTrue((out / "masks" / "case1_conf.png").is_file())
            self.assertTrue((out / "masks" / "case1_final_conf.png").is_file())
            self.assertTrue((out / "masks" / "case1_final_conf2.png").is_file())
            self.assertTrue((out / "heatmaps" / "case1_residual.png").is_file())
            self.assertTrue((out / "comparison" / "case1.png").is_file())
            self.assertTrue((out / "metrics.csv").is_file())
            with (out / "metrics.csv").open(newline="", encoding="utf-8") as handle:
                methods = {row["method"] for row in csv.DictReader(handle) if row["case"] != "Average"}
            self.assertTrue(
                {
                    "LQ",
                    "HYPIR-50",
                    "HYPIR-200",
                    "fusion_A",
                    "fusion_B",
                    "fusion_A_conf",
                    "fusion_B_conf",
                    "fusion_A_conf2",
                    "fusion_B_conf2",
                    "fusion_A_conf05",
                    "fusion_B_conf05",
                }.issubset(methods)
            )
            self.assertFalse((root / "results" / "fusion_v1").exists())


if __name__ == "__main__":
    unittest.main()
