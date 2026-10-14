"""EMA via peft multi-adapter, avoids `deepcopy` (which breaks autograd
through peft layers)."""
from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig


class PeftEMA:
    """Maintains a frozen "ema" peft adapter alongside the trainable "default" adapter.

    On `update(decay)`: copies LoRA tensors from the "default" adapter into
    the "ema" adapter with EMA decay.
    On `__enter__` / `__exit__`: temporarily switches active adapter to "ema"
    so model forward uses EMA weights (for trajectory rollouts).

    EMA params are kept `requires_grad=False` permanently; only the in-place
    EMA-copy below updates them. peft's adapter-toggle would otherwise flip
    requires_grad and cause backward to allocate gradients for them.
    """
    def __init__(self, peft_model: nn.Module, lora_cfg: LoraConfig,
                 adapter_name: str = "ema"):
        self.peft_model = peft_model
        self.adapter_name = adapter_name
        self.peft_model.add_adapter(adapter_name, lora_cfg)
        self._copy_default_into_ema()
        for n, p in self.peft_model.named_parameters():
            if f".{self.adapter_name}." in n and "lora_" in n:
                p.requires_grad_(False)

    def _matching_param_pairs(self):
        # peft renames LoRA params to include the adapter name as a suffix:
        # default: "...lora_A.default.weight" / ema: "...lora_A.ema.weight"
        params = dict(self.peft_model.named_parameters())
        for name, p in params.items():
            if ".default." in name and "lora_" in name:
                ema_name = name.replace(".default.", f".{self.adapter_name}.")
                if ema_name in params:
                    yield p, params[ema_name]

    @torch.no_grad()
    def _copy_default_into_ema(self):
        for d, e in self._matching_param_pairs():
            e.data.copy_(d.data)

    @torch.no_grad()
    def update(self, decay: float):
        for d, e in self._matching_param_pairs():
            e.data.mul_(decay).add_(d.data, alpha=(1.0 - decay))

    def __enter__(self):
        self._prev = self.peft_model.active_adapter
        self.peft_model.set_adapter(self.adapter_name)
        return self.peft_model

    def __exit__(self, *exc):
        self.peft_model.set_adapter(self._prev)

    def state_dict(self):
        return {n: p.detach().cpu() for n, p in self.peft_model.named_parameters()
                if f".{self.adapter_name}." in n and "lora_" in n}

    def load_state_dict(self, sd: dict):
        params = dict(self.peft_model.named_parameters())
        for k, v in sd.items():
            if k in params:
                params[k].data.copy_(v.to(params[k].device))
