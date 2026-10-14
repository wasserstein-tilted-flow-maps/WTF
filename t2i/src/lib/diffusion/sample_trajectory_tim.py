"""TiM-native trajectory sampler.

Drop-in replacement for `sample_trajectory_sd3.sample_trajectory`. Uses only
the API surface that `lib.tim_adapter.TimPipeline` exposes (no SD3-pipeline
internals like `check_inputs`, `prepare_latents`, `_execution_device`).

Returns the same 8-tuple as the SD3 version so the caller in
`lib/training/sampler.py` doesn't need to change.

Convention
----------
TiM uses continuous t ∈ [0, 1] with t=1 → noise, t=0 → data.
`scheduler.sigmas` is length N+1 going 1.0 → 0.0.
`scheduler.timesteps` is length N (the first N sigmas).
Euler step: latent <- latent + (sigma_next - sigma_curr) * velocity.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import torch


def _default_image_size(pipeline) -> int:
    """Image edge length, in pixels. Drives latent shape via spatial_downsample."""
    # TiM-T2I default is 1024×1024. Use the bundle config's sample_size if set.
    cfg = getattr(pipeline.bundle, "config", None)
    if cfg is not None:
        ss = getattr(cfg.model, "sample_size", None)
        if ss is not None:
            return int(ss)
    return 1024


@torch.inference_mode()
def sample_trajectory(
    self,
    tsfm_model_to_use,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 8,
    timesteps: List[int] = None,
    guidance_scale: float = 5.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "image",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    guidance_rexp=-1.0,
    reward_fn=None,
    prompt_metadata=None,
    return_output=True,
):
    bundle = self.bundle
    device = bundle.device

    # 1. Resolve image size and latent shape (for DC-AE: spatial_downsample=32).
    img_h = height or _default_image_size(self)
    img_w = width or _default_image_size(self)
    spatial = bundle.spatial_downsample
    lat_h, lat_w = img_h // spatial, img_w // spatial

    # TiM has 32 latent channels for DC-AE, 4 for SD-VAE. Read from the
    # transformer's expected in_channels via the bundle config.
    latent_channels = int(bundle.config.model.network.params.get("in_channels", 32))

    # 2. Encode prompts (cond + uncond).
    if isinstance(prompt, str):
        prompt = [prompt]
    batch_size = len(prompt)
    do_cfg = guidance_scale > 1.0

    prompt_embeds_, negative_prompt_embeds_, pooled_, neg_pooled_ = self.encode_prompt(
        prompt=prompt, do_classifier_free_guidance=do_cfg, device=device
    )
    prompt_embeds = prompt_embeds_
    negative_prompt_embeds = negative_prompt_embeds_
    pooled_prompt_embeds = pooled_
    negative_pooled_prompt_embeds = neg_pooled_

    if do_cfg:
        prompt_embeds_comb = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds_comb = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )
    else:
        prompt_embeds_comb = prompt_embeds
        pooled_prompt_embeds_comb = pooled_prompt_embeds

    # 3. Build random initial latents at t=1 (pure noise).
    sched = self.scheduler
    sched.set_timesteps(num_inference_steps, device=device)
    sigmas = sched.sigmas       # length N+1, 1.0 -> 0.0
    timesteps_seq = sched.timesteps  # length N
    n_steps = len(timesteps_seq)

    if latents is None:
        latents = torch.randn(
            (batch_size * num_images_per_prompt, latent_channels, lat_h, lat_w),
            device=device,
            dtype=prompt_embeds.dtype,
            generator=generator,
        )

    # 4. Euler denoising on [1.0, 0.0].
    all_latents = [latents]
    all_outputs = []

    for i, t in enumerate(timesteps_seq):
        sigma_curr = sigmas[i]
        sigma_next = sigmas[i + 1]
        dsigma = (sigma_next - sigma_curr).to(latents.dtype)

        # CFG: double the latent + timestep for cond/uncond.
        if do_cfg:
            latent_in = torch.cat([latents] * 2, dim=0)
        else:
            latent_in = latents
        t_batch = t.to(latent_in.dtype).expand(latent_in.shape[0])

        velocity = tsfm_model_to_use(
            hidden_states=latent_in,
            timestep=t_batch,
            encoder_hidden_states=prompt_embeds_comb,
            pooled_projections=pooled_prompt_embeds_comb,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]

        if do_cfg:
            v_uncond, v_cond = velocity.chunk(2, dim=0)
            velocity = v_uncond + guidance_scale * (v_cond - v_uncond)

        latents = latents + dsigma * velocity
        all_latents.append(latents)
        if return_output:
            all_outputs.append(velocity.detach())

    # 5. Decode final latents to image (no autograd needed for sampling).
    if output_type == "latent":
        image = latents
    else:
        # Decode in the [0, 1] domain (vae.decode in the adapter returns [-1, 1]).
        decoded = self.vae.decode(latents.to(torch.float32), return_dict=False)[0]
        image = ((decoded + 1) * 0.5).clamp(0, 1).detach()

    if return_output:
        return (
            image, all_latents, timesteps_seq,
            prompt_embeds, pooled_prompt_embeds,
            negative_prompt_embeds, negative_pooled_prompt_embeds,
            all_outputs,
        )
    else:
        return (
            image, all_latents, timesteps_seq,
            prompt_embeds, pooled_prompt_embeds,
            negative_prompt_embeds, negative_pooled_prompt_embeds,
            None,
        )
