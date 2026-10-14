"""Unified TiM-T2I pipeline construction (model + DC-AE + Gemma text encoder).

A thin wrapper over the helpers TiM already provides in `tim.models.vae` and
`tim.models.utils.text_encoders`. Loads everything in the dtypes / on the devices
that the training and sampling code expect.
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from omegaconf import OmegaConf

from tim.models.utils.text_encoders import load_text_encoder, encode_prompt
from tim.models.vae import (
    get_dc_ae,
    get_sd_vae,
    dc_ae_decode,
    sd_vae_decode,
)
from tim.schedulers.transition import TransitionSchedule
from tim.utils.misc_utils import instantiate_from_config, init_from_ckpt


# TiM's config names its VAE and text encoder by local paths (`checkpoints/...`).
# When those folders are absent we fall back to $TIM_REPO/<path>, then to the
# Hugging Face Hub copies of the same weights.
_HUB_IDS = {
    "dc-ae-f32c32-sana-1.1-diffusers": "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
    "gemma-3-1b-it": "google/gemma-3-1b-it",
}


def resolve_tim_asset(path: str) -> str:
    """Return a loadable location for a VAE / text-encoder path from TiM's config."""
    if os.path.isdir(path):
        return path
    repo = os.environ.get("TIM_REPO")
    if repo and os.path.isdir(os.path.join(repo, path)):
        return os.path.join(repo, path)
    return _HUB_IDS.get(os.path.basename(os.path.normpath(path)), path)


@dataclass
class TiMBundle:
    """Everything we need to run TiM end-to-end (sampling, fine-tune, eval)."""
    model: torch.nn.Module           # TiM (instantiated from config; weights loaded)
    scheduler: TransitionSchedule    # for sampling + transport-level utilities
    vae: torch.nn.Module             # DC-AE (or SD-VAE if config asks)
    decode_func: callable            # vae-aware latent → image decode
    text_encoder: torch.nn.Module    # Gemma-3-1B-it (frozen)
    tokenizer: object                # text tokenizer
    null_cap_feat: torch.Tensor      # cached embedding of the empty caption (for CFG)
    null_cap_mask: torch.Tensor      # cached attention mask of the empty caption
    config: object                   # the loaded OmegaConf config
    device: torch.device
    weight_dtype: torch.dtype
    spatial_downsample: int          # latent_h = image_h // spatial_downsample (DC-AE: 32, SD-VAE: 8)


def build_tim_bundle(
    config_path: str,
    ckpt_path: str,
    device: torch.device,
    weight_dtype: torch.dtype = torch.bfloat16,
    slice_vae: bool = True,
    load_text_encoder_to_device: bool = True,
) -> TiMBundle:
    """Construct everything needed for inference / training on TiM-T2I.

    Args:
        config_path: yaml config (e.g. `configs/t2i/tim_xl_p1_t2i.yaml`).
        ckpt_path: path to `t2i_model.bin`.
        device: torch device for the velocity model + VAE + text encoder.
        weight_dtype: dtype for the velocity model + VAE + text encoder. The
                      sampler upcasts to fp64 internally for trajectory
                      integration. The model itself runs in `weight_dtype`.
        slice_vae: if True, use partial VAE slicing during decode (mandatory for
                   1024×1024+ to fit memory).
        load_text_encoder_to_device: load Gemma to GPU. Setting False keeps it on
                                     CPU, useful when pre-encoding captions on a
                                     dataloader process and not needing inference.

    Returns:
        TiMBundle with all components on `device`.
    """
    config = OmegaConf.load(config_path)
    model_config = config.model
    model_config.vae_dir = resolve_tim_asset(model_config.vae_dir)
    model_config.text_encoder_dir = resolve_tim_asset(model_config.text_encoder_dir)

    # VAE
    if "dc-ae" in model_config.vae_dir:
        vae = get_dc_ae(model_config.vae_dir, dtype=torch.float32, device=device)
        spatial_downsample = 32
        # Tiling lets DC-AE decode at high resolution without OOM.
        vae.enable_tiling(2560, 2560, 2560, 2560)
        decode_func = functools.partial(dc_ae_decode, vae, slice_vae=slice_vae)
    elif "sd-vae" in model_config.vae_dir:
        vae = get_sd_vae(model_config.vae_dir, dtype=torch.float32, device=device)
        spatial_downsample = 8
        decode_func = functools.partial(sd_vae_decode, vae, slice_vae=slice_vae)
    else:
        raise ValueError(f"Unrecognised VAE dir: {model_config.vae_dir}")

    # Text encoder (Gemma-3-1B-it for the public T2I checkpoint)
    text_device = device if load_text_encoder_to_device else "cpu"
    text_encoder, tokenizer = load_text_encoder(
        text_encoder_dir=model_config.text_encoder_dir,
        device=text_device,
        weight_dtype=weight_dtype,
    )
    null_cap_feat, null_cap_mask = encode_prompt(
        tokenizer, text_encoder, device, weight_dtype,
        [""], model_config.use_last_hidden_state,
        max_seq_length=model_config.max_seq_length,
    )

    # Velocity model (TiM)
    model = instantiate_from_config(model_config.network).to(device=device, dtype=weight_dtype)
    init_from_ckpt(model, checkpoint_dir=ckpt_path, ignore_keys=None, verbose=True)

    # Transport + transition scheduler
    transport = instantiate_from_config(model_config.transport)
    scheduler = TransitionSchedule(
        transport=transport,
        **OmegaConf.to_container(model_config.transition_loss),
    )

    return TiMBundle(
        model=model,
        scheduler=scheduler,
        vae=vae,
        decode_func=decode_func,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        null_cap_feat=null_cap_feat,
        null_cap_mask=null_cap_mask,
        config=config,
        device=device,
        weight_dtype=weight_dtype,
        spatial_downsample=spatial_downsample,
    )


def encode_captions(
    bundle: TiMBundle,
    captions: list[str],
    truncate_to_max_length_in_batch: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a list of captions using the bundle's frozen text encoder.

    If `truncate_to_max_length_in_batch=True`, returns (cap_feats, cap_mask)
    truncated to the longest non-pad sequence in the batch, minor speedup for
    cross-attention on shorter prompts.
    """
    cap_feats, cap_mask = encode_prompt(
        bundle.tokenizer, bundle.text_encoder, bundle.device, bundle.weight_dtype,
        captions, bundle.config.model.use_last_hidden_state,
        max_seq_length=bundle.config.model.max_seq_length,
    )
    if truncate_to_max_length_in_batch:
        max_len = int(cap_mask.sum(dim=-1).max().item())
        cap_feats = cap_feats[:, :max_len]
        cap_mask = cap_mask[:, :max_len]
    return cap_feats, cap_mask


def latent_size(bundle: TiMBundle, height: int, width: int) -> Tuple[int, int]:
    return height // bundle.spatial_downsample, width // bundle.spatial_downsample


def decode_to_image(bundle: TiMBundle, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents to images in [0, 1] using the bundle's VAE.

    Returns (B, 3, H, W) float32 in [0, 1].

    NOTE: Uses `bundle.decode_func` which (for DC-AE) wraps the decode in
    `torch.no_grad()`. That's correct for sampling but BREAKS reward-gradient
    fine-tuning (the gradient chain through latent → image → reward dies).
    Use `decode_to_image_with_grad` instead when computing reward gradients.
    """
    images = bundle.decode_func(latents.to(torch.float32))
    # DC-AE / SD-VAE decode returns images in [-1, 1].
    return torch.clamp((images + 1) / 2.0, 0.0, 1.0)


def decode_to_image_with_grad(bundle: TiMBundle, latents: torch.Tensor) -> torch.Tensor:
    """Same as `decode_to_image` but preserves the autograd graph through the VAE.

    Required for reward-gradient fine-tuning. Implemented inline (not via
    `bundle.decode_func`) because TiM's `dc_ae_decode` / `sd_vae_decode` wrap
    their entire bodies in `torch.no_grad()`.
    """
    if hasattr(bundle.vae, "_decode") and hasattr(bundle.vae, "std") and hasattr(bundle.vae, "mean"):
        # DC-AE: undo the BN norm (z = latents * std + mean) then decode.
        z = latents.to(torch.float32) * bundle.vae.std + bundle.vae.mean
        decoded = bundle.vae._decode(z)
    else:
        # SD-VAE: scaling factor handling
        scaling = getattr(bundle.vae.config, "scaling_factor", 1.0)
        shift = getattr(bundle.vae.config, "shift_factor", 0.0)
        z = latents.to(torch.float32) / scaling + shift
        decoded = bundle.vae.decode(z, return_dict=False)[0]
    return torch.clamp((decoded + 1) / 2.0, 0.0, 1.0)
