from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from baseline_bakeoff.benchmark import discover_benchmark_cases, resize_for_perceptual


class BenchmarkContractTests(unittest.TestCase):
    def test_discover_benchmark_cases_matches_lq_gt_and_model_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for directory in (root / "lq", root / "gt", root / "diffir", root / "hypir"):
                directory.mkdir()
            image = Image.new("RGB", (20, 10), (10, 20, 30))
            image.save(root / "lq" / "case1_lq.jpg")
            image.save(root / "gt" / "case1_gt.jpg")
            image.save(root / "gt" / "case1_lq.jpg")
            image.save(root / "diffir" / "case1_lq.png")
            image.save(root / "hypir" / "case1_lq.png")

            cases = discover_benchmark_cases(root / "lq", root / "gt", {"DiffIR": root / "diffir", "HYPIR": root / "hypir"})

            self.assertEqual([case.case for case in cases], ["case1"])
            self.assertEqual(cases[0].outputs["DiffIR"].name, "case1_lq.png")

    def test_resize_for_perceptual_caps_max_side_without_changing_aspect(self) -> None:
        image = Image.new("RGB", (4096, 3072), (1, 2, 3))

        resized = resize_for_perceptual(image, max_side=1024)

        self.assertEqual(resized.size, (1024, 768))
