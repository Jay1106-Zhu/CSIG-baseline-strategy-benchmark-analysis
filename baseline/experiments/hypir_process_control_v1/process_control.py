"""HYPIR inference-parameter map, taken from this repo's code only.

Inference is one UNet epsilon pass plus DDPMScheduler.pred_original_sample.
There is no sampling loop. coeff_t scales how far that x0 step moves off LQ.
"""
from __future__ import annotations

import numpy as np


INFERENCE_IS_SINGLE_STEP = True

# SD2 scheduler_config.json in this checkout: scaled_linear, 1000 steps.
_SCHEDULER_BETA_START = 0.00085
_SCHEDULER_BETA_END = 0.012
_SCHEDULER_STEPS = 1000

# Already-completed single-variable sweep. Do not re-infer these.
EXISTING_COEFF_T_LEVELS: tuple[int, ...] = (50, 75, 100, 150, 200)
SELECTED_CONTROLS: tuple[str, ...] = ("coeff_t",)

PARAMETER_MAP: dict[str, dict[str, object]] = {
    "coeff_t": {
        "defined_in": "HYPIR/test.py --coeff_t → SD2Enhancer.__init__",
        "used_in": "SD2Enhancer.forward_generator",
        "actual_effect": (
            "Passed as timestep to DDPMScheduler.step(eps, coeff_t, z_in). "
            "Only .pred_original_sample is kept: "
            "x0 = (z_in - sqrt(1-α_t)*eps) / sqrt(α_t). "
            "Larger t → larger sqrt(1-α_t) → more of the UNet residual is applied."
        ),
        "controls_generation_freedom": True,
        "present_in_code": True,
        "already_swept": True,
        "suitable": True,
        "note": "This is the only real generation-strength knob. H50=50, H200=200.",
    },
    "model_t": {
        "defined_in": "HYPIR/test.py --model_t → SD2Enhancer.__init__",
        "used_in": "SD2Enhancer.prepare_inputs",
        "actual_effect": (
            "Integer filled into UNet timestep embedding: G(z_in, timesteps=model_t, text). "
            "Does not change the scheduler x0 formula. LoRA was trained at model_t=200."
        ),
        "controls_generation_freedom": False,
        "present_in_code": True,
        "already_swept": True,
        "suitable": False,
        "note": "Lowering model_t with coeff_t=200 made PSNR worse. Not denoise strength.",
    },
    "sampling_steps": {
        "defined_in": None,
        "used_in": None,
        "actual_effect": "No denoising loop. forward_generator calls UNet once.",
        "controls_generation_freedom": False,
        "present_in_code": False,
        "already_swept": False,
        "suitable": False,
        "note": "Not an img2img sampler.",
    },
    "noise_injection": {
        "defined_in": None,
        "used_in": None,
        "actual_effect": "z_in is VAE-encoded LQ * scaling_factor. No q_sample / add_noise.",
        "controls_generation_freedom": False,
        "present_in_code": False,
        "already_swept": False,
        "suitable": False,
        "note": None,
    },
    "start_end_timestep": {
        "defined_in": None,
        "used_in": None,
        "actual_effect": "No schedule. Single (model_t, coeff_t) pair.",
        "controls_generation_freedom": False,
        "present_in_code": False,
        "already_swept": False,
        "suitable": False,
        "note": None,
    },
    "guidance_scale": {
        "defined_in": None,
        "used_in": None,
        "actual_effect": "No unconditional pass, no CFG mix.",
        "controls_generation_freedom": False,
        "present_in_code": False,
        "already_swept": False,
        "suitable": False,
        "note": None,
    },
    "lora_scale": {
        "defined_in": None,
        "used_in": None,
        "actual_effect": (
            "LoRA is add_adapter + load_state_dict. test.py does not call set_adapter_scale. "
            "lora_rank / lora_modules only describe the loaded weight file."
        ),
        "controls_generation_freedom": False,
        "present_in_code": False,
        "already_swept": False,
        "suitable": False,
        "note": "Do not invent a scale this round; coeff_t already exists.",
    },
    "lora_rank": {
        "defined_in": "HYPIR/test.py --lora_rank",
        "used_in": "SD2Enhancer.init_generator LoraConfig",
        "actual_effect": "Must match HYPIR_sd2.pth. Changing it without new weights fails load.",
        "controls_generation_freedom": False,
        "present_in_code": True,
        "already_swept": False,
        "suitable": False,
        "note": None,
    },
    "prompt": {
        "defined_in": "HYPIR/test.py --captioner / --txt_dir",
        "used_in": "SD2Enhancer.prepare_inputs → CLIPTextModel",
        "actual_effect": "Cross-attention condition. Current baseline uses EmptyCaptioner ('').",
        "controls_generation_freedom": False,
        "present_in_code": True,
        "already_swept": False,
        "suitable": False,
        "note": "E4 already showed empty-prompt mapping is locked. Not a strength knob.",
    },
    "seed": {
        "defined_in": "HYPIR/test.py --seed",
        "used_in": "accelerate.utils.set_seed; VAE encode uses latent_dist.sample()",
        "actual_effect": "E4: four seeds pairwise PSNR 39.5–39.9 dB. Not sampling variance.",
        "controls_generation_freedom": False,
        "present_in_code": True,
        "already_swept": True,
        "suitable": False,
        "note": None,
    },
    "patch_size_stride": {
        "defined_in": "HYPIR/test.py --patch_size --stride",
        "used_in": "BaseEnhancer.enhance tiled VAE/UNet",
        "actual_effect": "Tile size for 4K. Not generation strength.",
        "controls_generation_freedom": False,
        "present_in_code": True,
        "already_swept": False,
        "suitable": False,
        "note": None,
    },
    "upscale": {
        "defined_in": "HYPIR/test.py --upscale",
        "used_in": "BaseEnhancer.enhance bicubic before VAE",
        "actual_effect": "Current runs use factor=1. Resolution, not strength.",
        "controls_generation_freedom": False,
        "present_in_code": True,
        "already_swept": False,
        "suitable": False,
        "note": None,
    },
    "wavelet_reconstruction": {
        "defined_in": "HYPIR/utils/common.py wavelet_reconstruction",
        "used_in": "BaseEnhancer.enhance after VAE decode",
        "actual_effect": "Always on: generated high-frequency + LQ low-frequency/color. No CLI flag.",
        "controls_generation_freedom": False,
        "present_in_code": True,
        "already_swept": False,
        "suitable": False,
        "note": "Not an exposed control. Do not invent a mix weight this round.",
    },
    "scheduler": {
        "defined_in": "SD2Enhancer.init_scheduler",
        "used_in": "forward_generator scheduler.step",
        "actual_effect": "DDPMScheduler.from_pretrained(.../scheduler). prediction_type=epsilon.",
        "controls_generation_freedom": False,
        "present_in_code": True,
        "already_swept": False,
        "suitable": False,
        "note": "Hard-coded class. Not a runtime knob.",
    },
    "degradation_conditioning": {
        "defined_in": None,
        "used_in": None,
        "actual_effect": "No degradation encoder at inference. LQ enters only as VAE latent.",
        "controls_generation_freedom": False,
        "present_in_code": False,
        "already_swept": False,
        "suitable": False,
        "note": None,
    },
}


def existing_coeff_t_levels() -> tuple[int, ...]:
    return EXISTING_COEFF_T_LEVELS


def scaled_linear_alphabar(
    num_train_timesteps: int = _SCHEDULER_STEPS,
    beta_start: float = _SCHEDULER_BETA_START,
    beta_end: float = _SCHEDULER_BETA_END,
) -> np.ndarray:
    """Match diffusers scaled_linear betas used by this SD2 scheduler config."""
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=np.float64) ** 2
    return np.cumprod(1.0 - betas)


def coeff_t_eps_scale(timestep: int) -> float:
    """Multiplier of eps in x0 = (z - scale*eps)/sqrt(α_t). Larger → more move off LQ."""
    if not 0 <= int(timestep) < _SCHEDULER_STEPS:
        raise ValueError(f"timestep must be in [0, {_SCHEDULER_STEPS}), got {timestep}")
    alpha_bar = float(scaled_linear_alphabar()[int(timestep)])
    return float(((1.0 - alpha_bar) / alpha_bar) ** 0.5)


def default_coeff_dirs(project_root) -> dict[int, object]:
    from pathlib import Path

    root = Path(project_root)
    return {
        value: root / "baseline" / "experiments" / f"coeff_t_{value}" / "output" / "result"
        for value in EXISTING_COEFF_T_LEVELS
    }
