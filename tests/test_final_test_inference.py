import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from baseline.experiments.structure_local_restoration import compute_weight_map, fuse_image
from baseline.experiments.final_test_inference.run_final_test import (
    FROZEN,
    assert_formula_frozen,
    build_hypir_command,
    inventory_test_dir,
    save_jpg,
    texture_selective_fuse,
)


class FrozenConfigTests(unittest.TestCase):
    def test_upscale_is_one_and_not_default_four(self):
        self.assertEqual(FROZEN["upscale"], 1)
        self.assertEqual(FROZEN["model_t"], 200)
        self.assertEqual(FROZEN["coeff_t"], 200)
        self.assertEqual(FROZEN["seed"], 231)
        self.assertEqual(FROZEN["captioner"], "empty")
        cmd = build_hypir_command(Path("python"), Path("lq"), Path("out"), Path("HYPIR"))
        joined = " ".join(cmd)
        self.assertIn("--upscale", cmd)
        self.assertIn("1", cmd)
        self.assertNotIn("--upscale 4", joined.replace("  ", " "))
        upscale_val = cmd[cmd.index("--upscale") + 1]
        self.assertEqual(upscale_val, "1")
        self.assertEqual(cmd[cmd.index("--captioner") + 1], "empty")

    def test_formula_source_matches_audit(self):
        assert_formula_frozen()
        src = inspect.getsource(compute_weight_map)
        self.assertIn("0.05 + 0.46 * texture * (0.35 + 0.65 * protect)", src)
        self.assertIn("np.clip(weight, 0.0, 0.60)", src)
        fuse_src = inspect.getsource(fuse_image)
        self.assertIn("weights[..., None] * hypir_array", fuse_src)
        self.assertIn("(1.0 - weights[..., None]) * lq_array", fuse_src)

    def test_wrapper_does_not_import_case_to_scene(self):
        import baseline.experiments.final_test_inference.run_final_test as mod

        src = inspect.getsource(mod)
        self.assertNotIn("from baseline.experiments.hypir_fusion_v1", src)
        self.assertNotIn("scene_alpha_for_case", src)


class InventoryTests(unittest.TestCase):
    def test_inventory_rejects_wrong_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.fromarray(np.zeros((32, 40, 3), dtype=np.uint8)).save(root / "case1.jpg", format="JPEG")
            with self.assertRaises(SystemExit):
                inventory_test_dir(root)

    def test_inventory_accepts_case1_to_100(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for i in range(1, 101):
                Image.fromarray(np.full((16, 16, 3), i % 255, dtype=np.uint8)).save(root / f"case{i}.jpg", format="JPEG")
            report = inventory_test_dir(root)
            self.assertEqual(report["count"], 100)
            self.assertEqual(report["missing"], [])
            self.assertEqual(report["duplicate"], [])


class FusionWriteTests(unittest.TestCase):
    def test_texture_selective_writes_matching_jpg(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lq = np.zeros((24, 32, 3), dtype=np.uint8)
            lq[:, 10:14] = 200
            h200 = np.full((24, 32, 3), 80, dtype=np.uint8)
            Image.fromarray(lq).save(root / "lq.jpg", format="JPEG")
            Image.fromarray(h200).save(root / "h200.png")
            out = root / "case1.jpg"
            info = texture_selective_fuse(root / "lq.jpg", root / "h200.png", out)
            self.assertTrue(out.is_file())
            with Image.open(out) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (32, 24))
            self.assertLessEqual(info["max_weight"], 0.60)
            self.assertGreaterEqual(info["min_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
