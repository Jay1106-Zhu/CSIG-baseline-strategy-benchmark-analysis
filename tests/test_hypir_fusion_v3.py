import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from baseline.experiments.hypir_fusion_v1.fusion import rgb_to_y
from baseline.experiments.hypir_fusion_v3.fusion import (
    BAND_SCALES,
    PYRAMID_LEVELS,
    decompose_y,
    fuse_multiband,
    reconstruct_y,
)
from baseline.experiments.hypir_fusion_v3.experiment import run_experiment


class LaplacianPyramidTests(unittest.TestCase):
    def test_reconstruct_matches_original_y(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(20, 220, size=(96, 128)).astype(np.float32)
        bands = decompose_y(y)
        self.assertEqual(set(bands), {"low", "mid", "high"})
        rebuilt = reconstruct_y(bands)
        self.assertTrue(np.allclose(rebuilt, y, atol=1e-3, rtol=0))

    def test_constant_image_has_near_zero_high_band(self):
        y = np.full((64, 80), 90.0, dtype=np.float32)
        bands = decompose_y(y)
        self.assertLess(float(np.abs(bands["high"]).mean()), 1e-3)
        self.assertLess(float(np.abs(bands["mid"]).mean()), 1e-3)
        self.assertTrue(np.allclose(bands["low"], 90.0, atol=1e-3))

    def test_band_scale_labels_are_documented(self):
        self.assertEqual(PYRAMID_LEVELS, 3)
        self.assertIn("high", BAND_SCALES)
        self.assertIn("mid", BAND_SCALES)
        self.assertIn("low", BAND_SCALES)


class MultiBandFusionTests(unittest.TestCase):
    def test_chroma_stays_from_lq(self):
        lq = np.zeros((32, 40, 3), dtype=np.uint8)
        lq[..., 0] = 40
        lq[..., 1] = 180
        h50 = np.full((32, 40, 3), 90, dtype=np.uint8)
        h200 = np.full((32, 40, 3), 200, dtype=np.uint8)
        fused = fuse_multiband(lq, h50, h200, lq)
        lq_ycc = __import__("cv2").cvtColor(lq, __import__("cv2").COLOR_RGB2YCrCb).astype(np.float32)
        out_ycc = __import__("cv2").cvtColor(
            np.clip(np.rint(fused), 0, 255).astype(np.uint8),
            __import__("cv2").COLOR_RGB2YCrCb,
        ).astype(np.float32)
        self.assertLess(float(np.abs(out_ycc[:, :, 1] - lq_ycc[:, :, 1]).mean()), 1.5)
        self.assertLess(float(np.abs(out_ycc[:, :, 2] - lq_ycc[:, :, 2]).mean()), 1.5)

    def test_high_from_h200_changes_y_when_sources_differ(self):
        lq = np.full((48, 64, 3), 40, dtype=np.uint8)
        h50 = lq.copy()
        h200 = lq.copy()
        h200[:, 20:28] = 220
        fused = fuse_multiband(lq, h50, h200, lq)
        self.assertGreater(float(np.abs(rgb_to_y(fused)[:, 20:28] - 40).mean()), 1.0)


class ExperimentArtifactTests(unittest.TestCase):
    def test_run_experiment_writes_v3_without_touching_v1(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dirs = {name: root / name for name in ("LQ", "H50", "H200", "GT", "texture", "fusionA", "v1", "v2")}
            for directory in dirs.values():
                directory.mkdir()
            lq = np.full((64, 80, 3), 40, dtype=np.uint8)
            gt = np.full((64, 80, 3), 80, dtype=np.uint8)
            h50 = np.full((64, 80, 3), 50, dtype=np.uint8)
            h200 = np.full((64, 80, 3), 200, dtype=np.uint8)
            texture = np.full((64, 80, 3), 60, dtype=np.uint8)
            fusion_a = np.full((64, 80, 3), 55, dtype=np.uint8)
            Image.fromarray(lq).save(dirs["LQ"] / "case1.png")
            Image.fromarray(gt).save(dirs["GT"] / "case1.png")
            Image.fromarray(h50).save(dirs["H50"] / "case1.png")
            Image.fromarray(h200).save(dirs["H200"] / "case1.png")
            Image.fromarray(texture).save(dirs["texture"] / "case1.png")
            Image.fromarray(fusion_a).save(dirs["fusionA"] / "case1.png")
            sentinel = dirs["v1"] / "keep.png"
            Image.fromarray(lq).save(sentinel)
            out = root / "fusion_v3"
            result = run_experiment(
                dirs["LQ"],
                dirs["H50"],
                dirs["H200"],
                out,
                gt_dir=dirs["GT"],
                texture_dir=dirs["texture"],
                fusion_v1_dir=dirs["fusionA"],
                compute_lpips=False,
                cases=("case1",),
            )
            self.assertEqual(result["cases"], ["case1"])
            self.assertTrue((out / "fusion" / "B" / "case1.png").is_file())
            self.assertTrue((out / "fusion" / "C" / "case1.png").is_file())
            self.assertTrue((out / "fusion" / "D01" / "case1.png").is_file())
            self.assertTrue((out / "fusion" / "D02" / "case1.png").is_file())
            self.assertTrue((out / "bands" / "case1_H200_high.png").is_file())
            self.assertTrue((out / "metrics.csv").is_file())
            self.assertTrue(sentinel.is_file())
            self.assertEqual(list(dirs["v2"].iterdir()), [])
            with (out / "metrics.csv").open(newline="", encoding="utf-8") as handle:
                methods = {row["method"] for row in csv.DictReader(handle) if row["case"] != "Average"}
            self.assertTrue(
                {
                    "LQ",
                    "HYPIR-50",
                    "HYPIR-200",
                    "texture_selective_h200",
                    "fusion_v1_A",
                    "multi_B",
                    "multi_C",
                    "multi_D01",
                    "multi_D02",
                }.issubset(methods)
            )


if __name__ == "__main__":
    unittest.main()
