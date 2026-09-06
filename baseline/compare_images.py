"""Create side-by-side visual comparisons for HYPIR evaluation results.

This utility only reads the input and HYPIR output images. It does not import
or modify HYPIR, and any resizing is applied to the comparison copy only.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
_CASE_RE = re.compile(r"case(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ImagePair:
    """A matched input/output pair with the same filename stem."""

    stem: str
    input_path: Path
    output_path: Path


@dataclass(frozen=True)
class ComparisonResult:
    """Metadata for one generated comparison image."""

    comparison_path: Path
    scale: float
    input_original_size: tuple[int, int]
    output_original_size: tuple[int, int]
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    input_box: tuple[int, int, int, int]
    output_box: tuple[int, int, int, int]
    input_label: str
    output_label: str
    ground_truth_size: Optional[tuple[int, int]] = None
    ground_truth_box: Optional[tuple[int, int, int, int]] = None
    ground_truth_label: str = ""

    @property
    def input_center(self) -> tuple[int, int]:
        left, top, right, bottom = self.input_box
        return ((left + right - 1) // 2, (top + bottom - 1) // 2)

    @property
    def output_center(self) -> tuple[int, int]:
        left, top, right, bottom = self.output_box
        return ((left + right - 1) // 2, (top + bottom - 1) // 2)

    @property
    def ground_truth_center(self) -> Optional[tuple[int, int]]:
        if self.ground_truth_box is None:
            return None
        left, top, right, bottom = self.ground_truth_box
        return ((left + right - 1) // 2, (top + bottom - 1) // 2)


@dataclass
class ProcessSummary:
    """Batch processing counts and paths, suitable for CLI reporting."""

    generated: int = 0
    generated_paths: list[Path] = field(default_factory=list)
    skipped_missing_input: list[str] = field(default_factory=list)
    skipped_missing_ground_truth: list[str] = field(default_factory=list)
    skipped_invalid: list[str] = field(default_factory=list)


def _sort_key(path: Path) -> tuple[int, object, str]:
    match = _CASE_RE.fullmatch(path.stem)
    if match:
        return (0, int(match.group(1)), path.name.casefold())
    return (1, path.stem.casefold(), path.name.casefold())


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS),
        key=_sort_key,
    )


def _by_stem(paths: Iterable[Path]) -> dict[str, Path]:
    """Build a deterministic, case-insensitive stem map."""

    result: dict[str, Path] = {}
    for path in paths:
        result.setdefault(path.stem.casefold(), path)
    return result


def _case_key(stem: str) -> str:
    """Normalize names such as case1_lq and case1_gt to case1."""

    normalized = stem.casefold()
    for suffix in ("_lq", "_gt", "_input", "_output"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _by_case_key(paths: Iterable[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in paths:
        result.setdefault(_case_key(path.stem), path)
    return result


def match_image_pairs(input_dir: Path | str, output_dir: Path | str) -> list[ImagePair]:
    """Return all output images that have an input with the same stem."""

    input_paths = _image_files(Path(input_dir))
    output_paths = _image_files(Path(output_dir))
    input_by_stem = _by_stem(input_paths)
    pairs = [
        ImagePair(output_path.stem, input_by_stem[output_path.stem.casefold()], output_path)
        for output_path in output_paths
        if output_path.stem.casefold() in input_by_stem
    ]
    return pairs


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        # Apply an embedded orientation for display, while leaving source files untouched.
        return ImageOps.exif_transpose(source).convert("RGB")


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _fit_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, font_size: int, bold: bool = False) -> tuple[str, ImageFont.ImageFont]:
    """Shrink, then ellipsize, metadata so it stays inside its panel."""

    size = max(10, font_size)
    while size >= 10:
        font = _load_font(size, bold)
        if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
            return text, font
        size -= 1
    font = _load_font(10, bold)
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text, font
    suffix = "..."
    available = max(1, max_width - draw.textbbox((0, 0), suffix, font=font)[2])
    prefix = ""
    for char in text:
        candidate = prefix + char
        if draw.textbbox((0, 0), candidate, font=font)[2] > available:
            break
        prefix = candidate
    return prefix + suffix, font


def _draw_header(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    role: str,
    details: str,
    header_height: int,
) -> str:
    left, _top, right, _bottom = box
    panel_width = right - left
    role_text, role_font = _fit_text(draw, role, panel_width, 24, bold=True)
    details_text, details_font = _fit_text(draw, details, panel_width, 17)
    role_bounds = draw.textbbox((0, 0), role_text, font=role_font)
    details_bounds = draw.textbbox((0, 0), details_text, font=details_font)
    center_x = (left + right) / 2
    role_y = max(4, (header_height - (role_bounds[3] - role_bounds[1] + details_bounds[3] - details_bounds[1] + 8)) // 2)
    draw.text((center_x, role_y), role_text, font=role_font, fill=(25, 25, 25), anchor="ma")
    draw.text((center_x, role_y + role_bounds[3] - role_bounds[1] + 8), details_text, font=details_font, fill=(70, 70, 70), anchor="ma")
    return f"{role} | {details}"


def create_comparison(
    input_path: Path | str,
    output_path: Path | str,
    comparison_path: Path | str,
    *,
    ground_truth_path: Path | str | None = None,
    max_panel_width: Optional[int] = 2048,
    max_panel_height: Optional[int] = 2048,
    margin: int = 24,
    gap: int = 24,
) -> ComparisonResult:
    """Create a two- or three-panel PNG using one scale factor for every image."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    comparison_path = Path(comparison_path)
    input_image = _load_rgb(input_path)
    output_image = _load_rgb(output_path)
    ground_truth_image = _load_rgb(Path(ground_truth_path)) if ground_truth_path is not None else None
    input_original_size = input_image.size
    output_original_size = output_image.size
    ground_truth_original_size = ground_truth_image.size if ground_truth_image is not None else None

    images = [input_image, output_image] + ([ground_truth_image] if ground_truth_image is not None else [])
    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)
    scale_limits = [1.0]
    if max_panel_width is not None:
        if max_panel_width <= 0:
            raise ValueError("max_panel_width must be positive")
        scale_limits.append(max_panel_width / max_width)
    if max_panel_height is not None:
        if max_panel_height <= 0:
            raise ValueError("max_panel_height must be positive")
        scale_limits.append(max_panel_height / max_height)
    scale = min(scale_limits)
    input_size = (max(1, round(input_image.width * scale)), max(1, round(input_image.height * scale)))
    output_size = (max(1, round(output_image.width * scale)), max(1, round(output_image.height * scale)))
    if input_size != input_image.size:
        input_image = input_image.resize(input_size, Image.Resampling.LANCZOS)
    if output_size != output_image.size:
        output_image = output_image.resize(output_size, Image.Resampling.LANCZOS)
    ground_truth_size = None
    if ground_truth_image is not None:
        ground_truth_size = (max(1, round(ground_truth_image.width * scale)), max(1, round(ground_truth_image.height * scale)))
        if ground_truth_size != ground_truth_image.size:
            ground_truth_image = ground_truth_image.resize(ground_truth_size, Image.Resampling.LANCZOS)

    header_height = 78
    panel_sizes = [input_size, output_size] + ([ground_truth_size] if ground_truth_size is not None else [])
    canvas_width = margin * 2 + sum(size[0] for size in panel_sizes) + gap * (len(panel_sizes) - 1)
    canvas_height = header_height + max(size[1] for size in panel_sizes) + margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    input_box = (margin, header_height, margin + input_size[0], header_height + input_size[1])
    output_left = margin + input_size[0] + gap
    output_box = (output_left, header_height, output_left + output_size[0], header_height + output_size[1])
    input_details = f"{input_path.name} | {input_original_size[0]}x{input_original_size[1]}"
    output_details = f"{output_path.name} | {output_original_size[0]}x{output_original_size[1]}"
    input_label = _draw_header(draw, input_box, "Input", input_details, header_height)
    output_label = _draw_header(draw, output_box, "HYPIR Output", output_details, header_height)
    ground_truth_box = None
    ground_truth_label = ""
    if ground_truth_image is not None and ground_truth_size is not None and ground_truth_original_size is not None:
        ground_truth_left = output_box[2] + gap
        ground_truth_box = (ground_truth_left, header_height, ground_truth_left + ground_truth_size[0], header_height + ground_truth_size[1])
        ground_truth_details = f"{Path(ground_truth_path).name} | {ground_truth_original_size[0]}x{ground_truth_original_size[1]}"
        ground_truth_label = _draw_header(draw, ground_truth_box, "Ground Truth", ground_truth_details, header_height)
    canvas.paste(input_image, (input_box[0], input_box[1]))
    canvas.paste(output_image, (output_box[0], output_box[1]))
    if ground_truth_image is not None and ground_truth_box is not None:
        canvas.paste(ground_truth_image, (ground_truth_box[0], ground_truth_box[1]))
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(comparison_path, format="PNG")
    return ComparisonResult(
        comparison_path=comparison_path,
        scale=scale,
        input_original_size=input_original_size,
        output_original_size=output_original_size,
        input_size=input_size,
        output_size=output_size,
        input_box=input_box,
        output_box=output_box,
        input_label=input_label,
        output_label=output_label,
        ground_truth_size=ground_truth_size,
        ground_truth_box=ground_truth_box,
        ground_truth_label=ground_truth_label,
    )


def process_comparisons(
    input_dir: Path | str,
    output_dir: Path | str,
    comparison_dir: Path | str,
    *,
    ground_truth_dir: Path | str | None = None,
    max_panel_width: Optional[int] = 2048,
    max_panel_height: Optional[int] = 2048,
) -> ProcessSummary:
    """Generate comparisons for every matching output and report skipped files."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    comparison_dir = Path(comparison_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    input_by_stem = _by_stem(_image_files(input_dir))
    ground_truth_by_case = _by_case_key(_image_files(Path(ground_truth_dir))) if ground_truth_dir is not None else {}
    summary = ProcessSummary()
    for output_path in _image_files(output_dir):
        input_path = input_by_stem.get(output_path.stem.casefold())
        if input_path is None:
            summary.skipped_missing_input.append(output_path.name)
            continue
        target = comparison_dir / f"{output_path.stem}_compare.png"
        ground_truth_path = ground_truth_by_case.get(_case_key(output_path.stem))
        if ground_truth_dir is not None and ground_truth_path is None:
            summary.skipped_missing_ground_truth.append(output_path.name)
            continue
        try:
            create_comparison(
                input_path,
                output_path,
                target,
                ground_truth_path=ground_truth_path,
                max_panel_width=max_panel_width,
                max_panel_height=max_panel_height,
            )
        except (OSError, ValueError, Image.UnidentifiedImageError) as exc:
            summary.skipped_invalid.append(f"{output_path.name}: {exc}")
            continue
        summary.generated += 1
        summary.generated_paths.append(target)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    baseline_dir = Path(__file__).resolve().parent
    parser.add_argument("--input-dir", type=Path, default=baseline_dir / "evaluation_input")
    parser.add_argument("--output-dir", type=Path, default=baseline_dir / "evaluation_output" / "result")
    parser.add_argument("--ground-truth-dir", type=Path, default=None, help="Optional GT directory; case1_lq matches case1_gt")
    parser.add_argument("--comparison-dir", type=Path, default=baseline_dir / "comparison")
    parser.add_argument("--max-panel-width", type=int, default=2048)
    parser.add_argument("--max-panel-height", type=int, default=2048)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.input_dir.is_dir():
        print(f"Input directory does not exist: {args.input_dir}")
        return 2
    if not args.output_dir.is_dir():
        print(f"Output directory does not exist: {args.output_dir}")
        return 2
    summary = process_comparisons(
        args.input_dir,
        args.output_dir,
        args.comparison_dir,
        ground_truth_dir=args.ground_truth_dir,
        max_panel_width=args.max_panel_width,
        max_panel_height=args.max_panel_height,
    )
    print(f"Generated {summary.generated} comparison image(s) in {args.comparison_dir}")
    if summary.skipped_missing_input:
        print("Skipped outputs without matching input: " + ", ".join(summary.skipped_missing_input))
    if summary.skipped_missing_ground_truth:
        print("Skipped outputs without matching ground truth: " + ", ".join(summary.skipped_missing_ground_truth))
    if summary.skipped_invalid:
        print("Skipped unreadable outputs: " + "; ".join(summary.skipped_invalid))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
