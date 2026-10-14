"""TiM-T2I substrate adapters that expose the SD3-pipeline API surface that
`vggflow` is written against. Drop-in replacements:

  * `TimPipeline`        -> StableDiffusion3Pipeline
  * `TimTransformer`     -> SD3Transformer2DModel
  * `TimSchedulerAdapter`-> FlowMatchEulerDiscreteScheduler
  * `TimVAEAdapter`      -> SD3 VAE

Conventions
-----------
SD3 conventions used by vggflow:
  - `transformer(hidden_states, timestep, encoder_hidden_states,
                 pooled_projections, joint_attention_kwargs, return_dict=False)`
    -> `(velocity,)` so caller does `[0]`.
  - `pipeline.scheduler.sigmas` is a length-(N+1) tensor with `sigmas[-1] == 0`.
  - `pipeline.scheduler.timesteps` is a length-N tensor.
  - SD3 timestep ∈ [0, 1000] but vggflow's algorithm uses
    `sigma = timestep / 1000.0` everywhere, so internally everything is in
    [0, 1]. We use TiM's native t ∈ [0, 1] and pretend `timestep == sigma`
    (no division by 1000), which is consistent.
  - `pipeline.vae.decode(latents, return_dict=False)[0]` -> image in [-1, 1].
  - `pipeline.vae.config.scaling_factor` / `shift_factor`: vggflow's
    reward_gradient.py reverses these before calling decode. We expose
    `scaling_factor=1.0` and no `shift_factor`, so that pre-decode arithmetic
    is a no-op. The actual DC-AE undo-norm happens inside our `vae.decode`.
  - `pipeline.encode_prompt(prompts)` -> 4-tuple
    `(prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds)`.
    TiM has no pooled embeds. We return zero-dim tensors as placeholders since
    the TiM model ignores `pooled_projections`.
  - `pipeline.joint_attention_kwargs` -> None.

TiM call signature: `model(x, t, r, y, attn_type='flash_attn')`. Diagonal
velocity is `r == t`.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

# TiM-substrate helpers live in this package (lib/tim_pipeline.py, lib/tim_lora.py).
from .tim_pipeline import (  # noqa: E402
    build_tim_bundle,
    encode_captions,
    latent_size,
    decode_to_image,
    decode_to_image_with_grad,
)
from .tim_lora import (  # noqa: E402
    apply_lora_to_tim,
    DEFAULT_LORA_TARGETS,
)
try:
    from .tim_lora import set_adapter_enabled  # noqa: E402
except ImportError:
    set_adapter_enabled = None
try:
    from .tim_lora import num_trainable_params  # noqa: E402
except ImportError:
    num_trainable_params = None


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler, exposes .sigmas, .timesteps, .set_timesteps()
# ─────────────────────────────────────────────────────────────────────────────

class TimSchedulerAdapter:
    """Mimics FlowMatchEulerDiscreteScheduler's sigma+timestep arrays.

    For TiM, t and sigma coincide (continuous, [0, 1]). We populate a uniform
    schedule from 1.0 down to (almost) 0.0; `sigmas` has the trailing 0.0 that
    vggflow's algorithm.py reads as `sigma_last`.
    """
    def __init__(self):
        self.sigmas: torch.Tensor = torch.zeros(2)
        self.timesteps: torch.Tensor = torch.zeros(1)
        self.num_inference_steps: int = 0

    def set_timesteps(self, num_inference_steps: int, device=None):
        self.num_inference_steps = num_inference_steps
        # Mirror SD3's flow-match convention but on [0, 1] (TiM-native).
        # sigmas: length N+1, going 1.0 -> 0.0
        # timesteps: length N, the first N entries of sigmas
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)
        if device is not None:
            sigmas = sigmas.to(device)
        self.sigmas = sigmas
        self.timesteps = sigmas[:-1].clone()


# ─────────────────────────────────────────────────────────────────────────────
# VAE, exposes .decode(), .encode(), .config{scaling_factor}, .requires_grad_(),
# .to(), .eval(), .config.shift_factor (absent → no shift)
# ─────────────────────────────────────────────────────────────────────────────

class TimVAEAdapter(nn.Module):
    """Wraps a TiM bundle's VAE so SD3-style call sites work unchanged.

    Exposes `decode(latents, return_dict=False) -> (image_in_minus1_to_1,)`.
    The DC-AE undo-norm (`z * std + mean`) is applied inside. We set
    `config.scaling_factor=1.0` so vggflow's pre-decode rescale is a no-op.
    """
    def __init__(self, bundle):
        super().__init__()
        self.bundle = bundle
        self.config = SimpleNamespace(
            scaling_factor=1.0,
            # No shift_factor attribute, caller checks hasattr() and skips.
        )
        # No trainable parameters.
        self._inner = bundle.vae

    def decode(self, latents, return_dict=False, *, with_grad: bool = False):
        if with_grad:
            img = decode_to_image_with_grad(self.bundle, latents.to(torch.float32))
        else:
            img = decode_to_image(self.bundle, latents.to(torch.float32))
        # Both helpers return [0, 1]; SD3 callers expect [-1, 1].
        img = (img * 2.0) - 1.0
        if return_dict:
            return SimpleNamespace(sample=img)
        return (img,)

    def encode(self, *args, **kwargs):
        raise NotImplementedError(
            "TimVAEAdapter.encode is intentionally not implemented — vggflow's "
            "value-approach generates from noise, never reconstructs from images."
        )

    def requires_grad_(self, flag: bool):
        for p in self._inner.parameters():
            p.requires_grad_(flag)
        return self

    def to(self, *args, **kwargs):
        # Bundle VAE is already on device; no-op the dtype changes since
        # decode_to_image_with_grad upcasts internally.
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Transformer, exposes the SD3 forward signature
# ─────────────────────────────────────────────────────────────────────────────

class TimTransformer(nn.Module):
    """Wraps a TiM model so it accepts the SD3 transformer call signature.

    SD3 call:
        transformer(hidden_states=x, timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=..., return_dict=False) -> (velocity,)

    TiM call:
        model(x, t, r, y, attn_type='flash_attn') -> velocity

    Mapping:
      * `hidden_states` -> `x`
      * `timestep`      -> `t`. TiM accepts a (B,) tensor in [0, 1]; we use
                          `timestep` as-is (already in [0, 1] under our
                          TimScheduler convention).
      * Diagonal velocity: `r = t`.
      * `encoder_hidden_states` -> `y` (Gemma cap_feats, opaque to TiM).
      * `pooled_projections`: ignored (TiM has no slot).
      * `joint_attention_kwargs`: ignored.
    """
    def __init__(self, tim_model: nn.Module, attn_type: str = "flash_attn"):
        super().__init__()
        self.inner = tim_model
        self.attn_type = attn_type

    @property
    def proj_out(self):
        # SD3's MMDiT exposes `.proj_out` as the final velocity-output Linear.
        # TiM's DiT puts the equivalent at `inner.final_layer.linear` (with an
        # upstream adaLN_modulation that's zero-initialized so modulate ~ id).
        # vggflow's `value_net_param='lora'/'copy'` paths perturb proj_out's
        # weight (`*=1e-5` to start the value head near zero), alias here so
        # they work without code changes elsewhere.
        return self.inner.final_layer.linear

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states=None,
        pooled_projections=None,
        joint_attention_kwargs=None,
        return_dict=False,
    ):
        x = hidden_states
        t = timestep
        if t.dim() == 0:
            t = t.expand(x.size(0))
        # bf16 model weights, cast input.
        target_dtype = next(p.dtype for p in self.inner.parameters() if not p.requires_grad)
        x = x.to(target_dtype)
        t = t.to(target_dtype)
        velocity = self.inner(x, t, t, encoder_hidden_states, attn_type=self.attn_type)
        if return_dict:
            return SimpleNamespace(sample=velocity)
        return (velocity,)

    # peft adapter API, vggflow disables/enables/sets adapters on transformer.
    def disable_adapters(self):
        if hasattr(self.inner, "disable_adapters"):
            self.inner.disable_adapters()
    def enable_adapters(self):
        if hasattr(self.inner, "enable_adapters"):
            self.inner.enable_adapters()
    def set_adapter(self, name):
        if hasattr(self.inner, "set_adapter"):
            self.inner.set_adapter(name)
    def add_adapter(self, *args, **kwargs):
        # Two calling conventions exist:
        #   diffusers/SD3 style: add_adapter(peft_config, adapter_name="...")
        #   peft.PeftModel    : add_adapter(adapter_name, peft_config)
        # vggflow uses the SD3 style. Our `self.inner` is a PeftModel. Translate.
        if len(args) == 1 and "adapter_name" in kwargs and not isinstance(args[0], str):
            peft_config = args[0]
            adapter_name = kwargs.pop("adapter_name")
            return self.inner.add_adapter(adapter_name, peft_config, **kwargs)
        return self.inner.add_adapter(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline, top-level container
# ─────────────────────────────────────────────────────────────────────────────

class TimPipeline:
    """Mimics StableDiffusion3Pipeline's interface enough for vggflow."""

    def __init__(self, bundle):
        self.bundle = bundle
        self.transformer = TimTransformer(bundle.model)
        self.vae = TimVAEAdapter(bundle)
        self.scheduler = TimSchedulerAdapter()
        # vggflow reads pipeline.joint_attention_kwargs in algorithm.py:198, keep None.
        self.joint_attention_kwargs = None
        # SD3-specific text encoders that vggflow sometimes touches. Provide stubs.
        self.text_encoder = SimpleNamespace(
            requires_grad_=lambda flag: None,
            to=lambda *a, **k: None,
        )
        self.text_encoder_2 = self.text_encoder
        self.text_encoder_3 = self.text_encoder
        self.safety_checker = None

    # SD3 pipeline API used by vggflow
    def to(self, device):
        return self  # bundle already on device

    def set_progress_bar_config(self, **kwargs):
        pass  # no-op

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError(
            "Use TimPipeline(bundle) — bundle is built from build_tim_bundle()."
        )

    def encode_prompt(
        self,
        prompt: List[str],
        prompt_2: Optional[List[str]] = None,
        prompt_3: Optional[List[str]] = None,
        do_classifier_free_guidance: bool = True,
        device=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (prompt_embeds, neg_prompt_embeds, pooled, neg_pooled)
        in SD3 layout. TiM's encoder gives a single (B, L, D) tensor. The
        "pooled" outputs are placeholder zeros since TiM has no pooled slot.
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        # truncate_to_max_length_in_batch=False so every batch returns the full
        # max_seq_length, collate_samples relies on torch.cat across batches.
        cap_feats, _mask = encode_captions(self.bundle, prompt, truncate_to_max_length_in_batch=False)
        B, L, D = cap_feats.shape
        if do_classifier_free_guidance:
            null = self.bundle.null_cap_feat[:, :L].expand(B, -1, -1).to(cap_feats.dtype)
            neg_prompt_embeds = null
        else:
            neg_prompt_embeds = torch.zeros_like(cap_feats)
        # Pooled projections, TiM doesn't use them; zero-shaped placeholders OK.
        pooled = torch.zeros(B, 1, device=cap_feats.device, dtype=cap_feats.dtype)
        neg_pooled = torch.zeros_like(pooled)
        return cap_feats, neg_prompt_embeds, pooled, neg_pooled


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_tim_pipeline(
    config_path: str,
    ckpt_path: str,
    device: torch.device,
) -> TimPipeline:
    """Construct a TimPipeline from a TiM YAML config + checkpoint.

    Caller must do `.scheduler.set_timesteps(N, device=device)` afterwards,
    matching how SD3 pipelines are set up.
    """
    bundle = build_tim_bundle(config_path, ckpt_path, device=device)
    return TimPipeline(bundle)
