"""Thin wrappers connecting FTFMVALUEDMFLoss to the TiM substrate.

`TiMLossAdapter` exposes the loss-class call signature `(x, t, r, y, tcf=,
t_min=, t_max=, return_logvar=False)` over TiM's `(x, t, r, y)`. It also
forces `attn_type='fused_attn'` (= SDPA) so `torch.func.jvp` works.

`RewardScoreShim` adapts our callable reward fn to expose `.score(images,
prompts)` (which is what the loss class calls).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TiMLossAdapter(nn.Module):
    """Wraps a peft-wrapped TiM model so it accepts the kwargs that
    `FTFMVALUEDMFLoss` passes (`tcf`, `t_min`, `t_max`, `return_logvar`).

    With token-conditioning OFF (the only mode supported here, since TiM has
    no slot for tcf), the extra kwargs are absorbed and discarded.

    `attn_type` defaults to `'flash_attn'` because we monkey-patch
    `tim.models.t2i.tim_model.flash_attn_func` to FA-JVP (which supports
    forward AD via custom Triton kernels) at startup. See `lib/fa_jvp/tim_patch.py`.
    `attn_type='fused_attn'` (SDPA) is the alternative that also supports forward-mode AD.
    """
    def __init__(self, peft_tim_model: nn.Module, attn_type: str = "flash_attn"):
        super().__init__()
        self.inner = peft_tim_model
        self.attn_type = attn_type

    def forward(self, x, t, r, y, **kwargs):
        if t.dim() > 1:
            t = t.flatten()
        if r.dim() > 1:
            r = r.flatten()
        # Cast input to base TiM weight dtype (LoRA is fp32, but base is bf16).
        target_dtype = next(p.dtype for p in self.inner.parameters() if not p.requires_grad)
        return self.inner(
            x.to(target_dtype),
            t.to(target_dtype),
            r.to(target_dtype),
            y,
            attn_type=self.attn_type,
        )

    # peft adapter API passthrough, used by trainer's adapter-toggle logic.
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
        return self.inner.add_adapter(*args, **kwargs)


class RewardScoreShim:
    """The loss class calls `reward_fn.score(images, prompts)`. Our reward
    callables (from vggflow's `lib/reward_func/rewards.py` or imscore directly)
    expose `__call__` only. This shim adds a `.score` method.
    """
    def __init__(self, reward_callable):
        self._fn = reward_callable

    def score(self, images, prompts):
        return self._fn(images, prompts)
