from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from PIL import Image

from baseline_bakeoff.runners.run_diffir import (
    discover_inputs,
    infer_lq,
    make_run_manifest,
    run_with_oom_retry,
    tensor_to_rgb_image,
)


def _write_rgb(path: Path, size: tuple[int, int] = (7, 5)) -> None:
    pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    pixels[..., 0] = 17
    pixels[..., 1] = 31
    pixels[..., 2] = 47
    Image.fromarray(pixels).save(path)


class DiffIRRunnerContractTests(unittest.TestCase):
    def test_discover_inputs_decodes_by_content_and_ignores_gt(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temporary_root = Path(temp_dir)
            input_dir = temporary_root / "input"
            input_dir.mkdir()
            _write_rgb(input_dir / "case2_lq.jpg")
            _write_rgb(input_dir / "case1_lq.png")
            _write_rgb(input_dir / "case1_gt.jpg")

            paths = discover_inputs(input_dir)

            self.assertEqual([path.name for path in paths], ["case1_lq.png", "case2_lq.jpg"])

    def test_infer_lq_does_not_accept_or_read_gt(self) -> None:
        seen_shapes: list[tuple[int, ...]] = []

        class IdentityModel:
            def __call__(self, lq: torch.Tensor) -> torch.Tensor:
                seen_shapes.append(tuple(lq.shape))
                return lq

        lq = torch.zeros((3, 5, 7), dtype=torch.float32)
        output = infer_lq(IdentityModel(), lq)

        self.assertEqual(tuple(output.shape), (1, 3, 5, 7))
        self.assertEqual(seen_shapes, [(1, 3, 5, 7)])

    def test_tensor_to_rgb_image_preserves_dimensions_and_rgb(self) -> None:
        tensor = torch.zeros((1, 3, 5, 7), dtype=torch.float32)
        image = tensor_to_rgb_image(tensor)

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (7, 5))

    def test_oom_retry_uses_fixed_tiles_and_preserves_dimensions(self) -> None:
        calls: list[tuple[int, int]] = []

        class OOMOnceModel:
            def __call__(self, lq: torch.Tensor) -> torch.Tensor:
                calls.append(tuple(lq.shape[-2:]))
                if len(calls) == 1:
                    raise RuntimeError("CUDA out of memory")
                return lq

        lq = torch.zeros((1, 3, 512, 512), dtype=torch.float32)
        output, retry = run_with_oom_retry(OOMOnceModel(), lq)

        self.assertEqual(tuple(output.shape), (1, 3, 512, 512))
        self.assertEqual(calls, [(512, 512), (512, 512)])
        self.assertEqual(retry, {"oom_retry": True, "tile": 512, "overlap": 128})

    def test_oom_retry_normalizes_chw_input_before_tiling(self) -> None:
        class OOMOnceModel:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, lq: torch.Tensor) -> torch.Tensor:
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("CUDA out of memory")
                return lq

        output, retry = run_with_oom_retry(OOMOnceModel(), torch.zeros((3, 512, 512), dtype=torch.float32))

        self.assertEqual(tuple(output.shape), (1, 3, 512, 512))
        self.assertEqual(retry["tile"], 512)

    def test_manifest_contains_diffir_provenance_and_excludes_gt(self) -> None:
        manifest = make_run_manifest(
            checkpoint=Path("Deblurring-DiffIRS2.pth"),
            checkpoint_sha256="abc123",
            source_commit="293f86c",
            input_dir=Path("input"),
            output_dir=Path("output"),
            config={"timesteps": 4, "tile": None},
            environment={"torch": "2.11.0+cu128"},
        )

        self.assertEqual(manifest["model"], "DiffIR Motion Deblurring")
        self.assertEqual(manifest["checkpoint_sha256"], "abc123")
        self.assertEqual(manifest["config"]["timesteps"], 4)
        self.assertNotIn("gt_dir", manifest)
