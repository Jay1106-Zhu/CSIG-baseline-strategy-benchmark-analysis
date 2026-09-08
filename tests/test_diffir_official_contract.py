from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from baseline_bakeoff.runners.run_diffir_official import load_yaml, network_kwargs, stage_csig_val_pairs


class DiffIROfficialContractTests(unittest.TestCase):
    def test_stage_csig_val_pairs_uses_matching_basenames(self) -> None:
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "src"
            dest = Path(temp_dir) / "csig_val"
            source.mkdir()
            Image.new("RGB", (4, 4), (1, 2, 3)).save(source / "case1_lq.jpg")
            Image.new("RGB", (4, 4), (4, 5, 6)).save(source / "case1_gt.jpg")
            for case in ("case2", "case3", "case4", "case5"):
                Image.new("RGB", (4, 4)).save(source / f"{case}_lq.jpg")
                Image.new("RGB", (4, 4)).save(source / f"{case}_gt.jpg")

            staged = stage_csig_val_pairs(source, dest)

            self.assertTrue((dest / "lq" / "case1.jpg").is_file())
            self.assertTrue((dest / "gt" / "case1.jpg").is_file())
            self.assertEqual(staged["case1"], dest / "lq" / "case1.jpg")
            self.assertFalse((dest / "lq" / "case1_lq.jpg").exists())

    def test_yaml_keeps_official_network_and_seed(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        opt = load_yaml(repo / "baseline_bakeoff" / "options" / "test_DiffIRS2_csig.yml")

        self.assertEqual(opt["model_type"], "DiffIRS2Model")
        self.assertEqual(opt["manual_seed"], 0)
        self.assertEqual(opt["scale"], 4)
        self.assertEqual(opt["network_g"]["type"], "DiffIRS2")
        self.assertEqual(opt["network_g"]["timesteps"], 4)
        self.assertEqual(opt["path"]["param_key_g"], "params_ema")
        kwargs = network_kwargs(opt)
        self.assertNotIn("type", kwargs)
        self.assertEqual(kwargs["timesteps"], 4)
