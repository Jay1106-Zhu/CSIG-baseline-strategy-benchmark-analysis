import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from baseline.experiments.hypir_fusion_v1 import (
    DEFAULT_SCENE_ALPHA,
    compute_structure_mask,
    fuse_scheme_a,
    fuse_scheme_b,
    rgb_to_y,
    scene_alpha_for_case,
)
from baseline.experiments.hypir_fusion_v1.experiment import run_experiment


def _bar_image(height: int, width: int, columns: tuple[int, int], value: int = 220) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    start, end = columns
    image[:, start:end] = value
    return image


class StructureMaskTests(unittest.TestCase):
    def test_aligned_structure_keeps_high_mask(self):
        lq = _bar_image(256, 256, (60, 76))
        h200 = _bar_image(256, 256, (60, 76), value=255)
        mask, diagnostics = compute_structure_mask(rgb_to_y(lq), rgb_to_y(h200), return_diagnostics=True)
        self.assertEqual(mask.shape, (256, 256))
        self.assertTrue(np.isfinite(mask).all())
        self.assertGreaterEqual(float(mask.min()), 0.0)
        self.assertLessEqual(float(mask.max()), 1.0)
        aligned = mask[:, 58:78]
        self.assertGreater(float(aligned.mean()), 0.55)
        self.assertGreater(float(diagnostics["cos"].mean()), 0.4)

    def test_new_h200_contour_lowers_mask(self):
        lq = _bar_image(256, 256, (60, 76))
        h200 = lq.copy()
        h200[:, 180:196] = 220
        mask = compute_structure_mask(rgb_to_y(lq), rgb_to_y(h200))
        original_edge = float(mask[:, 58:64].mean())
        novel_edge = float(mask[:, 178:184].mean())
        self.assertLess(novel_edge, original_edge)
        self.assertLess(novel_edge, 0.60)
        self.assertGreater(original_edge - novel_edge, 0.30)

    def test_flat_region_with_no_new_edge_stays_high(self):
        lq = np.full((128, 128, 3), 40, dtype=np.uint8)
        h200 = np.full((128, 128, 3), 48, dtype=np.uint8)
        mask = compute_structure_mask(rgb_to_y(lq), rgb_to_y(h200))
        self.assertGreater(float(mask.mean()), 0.85)


class YCbCrFusionTests(unittest.TestCase):
    def test_scheme_a_follows_residual_formula_on_y(self):
        lq = np.full((16, 16, 3), 40, dtype=np.uint8)
        h200 = np.full((16, 16, 3), 140, dtype=np.uint8)
        mask = np.ones((16, 16), dtype=np.float32)
        fused = fuse_scheme_a(lq, h200, mask, alpha=0.25)
        y = rgb_to_y(fused)
        self.assertTrue(np.allclose(y, 65.0, atol=1.5))

    def test_scheme_b_uses_h50_as_base(self):
        lq = np.full((16, 16, 3), 40, dtype=np.uint8)
        h50 = np.full((16, 16, 3), 80, dtype=np.uint8)
        h200 = np.full((16, 16, 3), 180, dtype=np.uint8)
        mask = np.ones((16, 16), dtype=np.float32)
        fused = fuse_scheme_b(lq, h50, h200, mask, alpha=0.50)
        y = rgb_to_y(fused)
        self.assertTrue(np.allclose(y, 130.0, atol=1.5))

    def test_chroma_is_taken_from_lq_not_h200(self):
        lq = np.full((24, 24, 3), 80, dtype=np.uint8)
        h200 = np.zeros((24, 24, 3), dtype=np.uint8)
        h200[..., 0] = 220
        h200[..., 1] = 40
        h200[..., 2] = 40
        mask = np.ones((24, 24), dtype=np.float32)
        fused = fuse_scheme_a(lq, h200, mask, alpha=0.40)
        channels = fused.reshape(-1, 3).astype(np.float32)
        rg = np.abs(channels[:, 0] - channels[:, 1]).mean()
        gb = np.abs(channels[:, 1] - channels[:, 2]).mean()
        self.assertLess(rg, 8.0)
        self.assertLess(gb, 8.0)
        self.assertGreater(float(rgb_to_y(fused).mean()), float(rgb_to_y(lq).mean()))

    def test_zero_mask_returns_base(self):
        lq = np.full((8, 8, 3), 30, dtype=np.uint8)
        h50 = np.full((8, 8, 3), 90, dtype=np.uint8)
        h200 = np.full((8, 8, 3), 200, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.float32)
        fused_a = fuse_scheme_a(lq, h200, mask, 0.30)
        fused_b = fuse_scheme_b(lq, h50, h200, mask, 0.30)
        self.assertTrue(np.allclose(rgb_to_y(fused_a), rgb_to_y(lq), atol=1.5))
        self.assertTrue(np.allclose(rgb_to_y(fused_b), rgb_to_y(h50), atol=1.5))


class SceneAlphaTests(unittest.TestCase):
    def test_manual_scene_table(self):
        self.assertEqual(DEFAULT_SCENE_ALPHA["text"], 0.25)
        self.assertEqual(DEFAULT_SCENE_ALPHA["book"], 0.30)
        self.assertEqual(DEFAULT_SCENE_ALPHA["clock"], 0.30)
        self.assertGreaterEqual(DEFAULT_SCENE_ALPHA["bird"], 0.10)
        self.assertLessEqual(DEFAULT_SCENE_ALPHA["bird"], 0.15)
        self.assertGreaterEqual(DEFAULT_SCENE_ALPHA["plant"], 0.05)
        self.assertLessEqual(DEFAULT_SCENE_ALPHA["plant"], 0.10)

    def test_case_mapping_and_override(self):
        self.assertEqual(scene_alpha_for_case("case1"), ("text", 0.25))
        self.assertEqual(scene_alpha_for_case("case2"), ("book", 0.30))
        self.assertEqual(scene_alpha_for_case("case5"), ("clock", 0.30))
        scene, alpha = scene_alpha_for_case("case4")
        self.assertEqual(scene, "plant")
        self.assertAlmostEqual(alpha, 0.08)
        self.assertEqual(scene_alpha_for_case("case4", overrides={"plant": 0.05}), ("plant", 0.05))
        self.assertEqual(scene_alpha_for_case("case99", default_alpha=0.15), ("unknown", 0.15))


class ExperimentArtifactTests(unittest.TestCase):
    def test_run_experiment_writes_fusion_mask_heatmap_and_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dirs = {name: root / name for name in ("LQ", "H50", "H200", "GT")}
            for directory in dirs.values():
                directory.mkdir()
            lq = _bar_image(64, 80, (20, 28), value=180)
            gt = lq.copy()
            h50 = np.clip(lq.astype(np.int16) + 8, 0, 255).astype(np.uint8)
            h200 = lq.copy()
            h200[:, 52:60] = 200
            for name, array in (("LQ", lq), ("H50", h50), ("H200", h200), ("GT", gt)):
                Image.fromarray(array).save(dirs[name] / "case1.png")
            out = root / "results" / "fusion_v1"
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
            self.assertTrue((out / "fusion" / "A" / "case1.png").is_file())
            self.assertTrue((out / "fusion" / "B" / "case1.png").is_file())
            self.assertTrue((out / "masks" / "case1.png").is_file())
            self.assertTrue((out / "heatmaps" / "case1_residual.png").is_file())
            self.assertTrue((out / "heatmaps" / "case1_suppressed.png").is_file())
            self.assertTrue((out / "comparison" / "case1.png").is_file())
            self.assertTrue((out / "metrics.csv").is_file())
            with (out / "metrics.csv").open(newline="", encoding="utf-8") as handle:
                methods = {row["method"] for row in csv.DictReader(handle) if row["case"] != "Average"}
            self.assertTrue({"LQ", "HYPIR-50", "HYPIR-200", "fusion_A", "fusion_B"}.issubset(methods))


if __name__ == "__main__":
    unittest.main()
