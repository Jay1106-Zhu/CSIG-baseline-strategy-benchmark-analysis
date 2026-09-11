import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from baseline.experiments.hypir_process_control_v1.process_control import (
    INFERENCE_IS_SINGLE_STEP,
    PARAMETER_MAP,
    SELECTED_CONTROLS,
    coeff_t_eps_scale,
    existing_coeff_t_levels,
)
from baseline.experiments.hypir_process_control_v1.experiment import run_experiment


class ParameterMapTests(unittest.TestCase):
    def test_inference_is_single_step_with_no_cfg(self):
        self.assertTrue(INFERENCE_IS_SINGLE_STEP)
        self.assertEqual(PARAMETER_MAP["sampling_steps"]["controls_generation_freedom"], False)
        self.assertEqual(PARAMETER_MAP["guidance_scale"]["present_in_code"], False)
        self.assertEqual(PARAMETER_MAP["noise_injection"]["present_in_code"], False)

    def test_coeff_t_is_the_selected_x0_scale_control(self):
        self.assertEqual(SELECTED_CONTROLS, ("coeff_t",))
        info = PARAMETER_MAP["coeff_t"]
        self.assertEqual(info["used_in"], "SD2Enhancer.forward_generator")
        self.assertTrue(info["controls_generation_freedom"])
        self.assertIn("pred_original_sample", info["actual_effect"])

    def test_model_t_is_unet_timestep_not_denoise_strength(self):
        info = PARAMETER_MAP["model_t"]
        self.assertTrue(info["present_in_code"])
        self.assertEqual(info["controls_generation_freedom"], False)
        self.assertIn("timestep", info["actual_effect"].casefold())

    def test_lora_scale_and_guidance_are_not_wired(self):
        self.assertEqual(PARAMETER_MAP["lora_scale"]["present_in_code"], False)
        self.assertEqual(PARAMETER_MAP["guidance_scale"]["present_in_code"], False)

    def test_coeff_t_eps_scale_increases_with_timestep(self):
        s50 = coeff_t_eps_scale(50)
        s100 = coeff_t_eps_scale(100)
        s200 = coeff_t_eps_scale(200)
        self.assertGreater(s100, s50)
        self.assertGreater(s200, s100)
        self.assertGreater(s50, 0.0)

    def test_existing_levels_include_h50_h200_and_intermediates(self):
        self.assertEqual(existing_coeff_t_levels(), (50, 75, 100, 150, 200))


class OfflineExperimentTests(unittest.TestCase):
    def test_run_experiment_reuses_pngs_and_does_not_touch_fusion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lq_dir = root / "lq"
            gt_dir = root / "gt"
            texture_dir = root / "texture"
            fusion_dir = root / "fusion_v1"
            fusion_v2_dir = root / "fusion_v2"
            for directory in (lq_dir, gt_dir, texture_dir, fusion_dir, fusion_v2_dir):
                directory.mkdir()
            coeff_dirs = {}
            lq = np.full((32, 40, 3), 40, dtype=np.uint8)
            gt = np.full((32, 40, 3), 80, dtype=np.uint8)
            Image.fromarray(lq).save(lq_dir / "case1.png")
            Image.fromarray(gt).save(gt_dir / "case1.png")
            Image.fromarray(np.full((32, 40, 3), 50, dtype=np.uint8)).save(texture_dir / "case1.png")
            Image.fromarray(np.full((32, 40, 3), 55, dtype=np.uint8)).save(fusion_dir / "case1.png")
            for value in (50, 75, 100, 150, 200):
                directory = root / f"coeff_t_{value}"
                directory.mkdir()
                Image.fromarray(np.full((32, 40, 3), 40 + value // 5, dtype=np.uint8)).save(directory / "case1.png")
                coeff_dirs[value] = directory
            out = root / "process_control"
            result = run_experiment(
                lq_dir=lq_dir,
                gt_dir=gt_dir,
                coeff_dirs=coeff_dirs,
                out_dir=out,
                texture_dir=texture_dir,
                fusion_v1_dir=fusion_dir,
                compute_lpips=False,
                cases=("case1",),
            )
            self.assertEqual(result["cases"], ["case1"])
            self.assertTrue((out / "metrics.csv").is_file())
            self.assertTrue((out / "comparison" / "case1.png").is_file())
            self.assertFalse(result.get("ran_inference", True))
            self.assertEqual(list(fusion_dir.iterdir())[0].name, "case1.png")
            self.assertEqual(list(fusion_v2_dir.iterdir()), [])
            with (out / "metrics.csv").open(newline="", encoding="utf-8") as handle:
                methods = {row["method"] for row in csv.DictReader(handle) if row["case"] != "Average"}
            self.assertTrue(
                {
                    "LQ",
                    "coeff_t_50",
                    "coeff_t_75",
                    "coeff_t_100",
                    "coeff_t_150",
                    "coeff_t_200",
                    "texture_selective_h200",
                    "fusion_v1_A",
                }.issubset(methods)
            )

    def test_run_experiment_refuses_inference_flag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "out"
            with self.assertRaises(RuntimeError):
                run_experiment(
                    lq_dir=Path(temp_dir),
                    gt_dir=Path(temp_dir),
                    coeff_dirs={},
                    out_dir=out,
                    allow_inference=True,
                )


if __name__ == "__main__":
    unittest.main()
