"""HYPIR output-space controlled fusion v1."""

from .fusion import (
    CASE_TO_SCENE,
    DEFAULT_SCENE_ALPHA,
    SCENE_NOTES,
    compute_structure_mask,
    fuse_detail,
    fuse_scheme_a,
    fuse_scheme_b,
    rgb_to_y,
    rgb_to_ycbcr,
    scene_alpha_for_case,
    ycbcr_to_rgb,
)

__all__ = [
    "CASE_TO_SCENE",
    "DEFAULT_SCENE_ALPHA",
    "SCENE_NOTES",
    "compute_structure_mask",
    "fuse_detail",
    "fuse_scheme_a",
    "fuse_scheme_b",
    "rgb_to_y",
    "rgb_to_ycbcr",
    "scene_alpha_for_case",
    "ycbcr_to_rgb",
]
