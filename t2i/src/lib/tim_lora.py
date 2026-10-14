"""Apply peft LoRA to a TiM model.

TiM has no built-in LoRA hooks (the `lora_hidden_size` field on `TiMBlock` controls a
SwiGLU bottleneck inside the AdaLN modulation MLP, *not* fine-tuning LoRA). We use peft
to inject standard rank-r adapters on the trainable Linear sub-modules in every
`TiMBlock`. By default we target both self-attention, cross-attention, and the SwiGLU
MLP, same coverage as DiT-style "all linear" LoRA, which matches what VGG-Flow uses on
SD3.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer


# Per-block Linear layers we wrap. These are the *terminal* sub-attribute names
# (peft `target_modules` matches by string suffix on the qualified module name).
DEFAULT_LORA_TARGETS: List[str] = [
    "attn.qkv",         # self-attention QKV projection (Linear D -> 3D)
    "attn.proj",        # self-attention output projection (Linear D -> D)
    "cross_attn.q",     # cross-attention Q from latent (Linear D -> D)
    "cross_attn.kv",    # cross-attention KV from text (Linear D -> 2D)
    "cross_attn.proj",  # cross-attention output projection (Linear D -> D)
    "mlp.fc1_g",        # SwiGLU gate proj (Linear D -> hidden)
    "mlp.fc1_x",        # SwiGLU value proj (Linear D -> hidden)
    "mlp.fc2",          # SwiGLU output proj (Linear hidden -> D)
]


def apply_lora_to_tim(
    model: nn.Module,
    rank: int = 8,
    alpha: int = 8,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    adapter_name: str = "default",
) -> nn.Module:
    """Wrap the TiM model with a peft LoRA adapter.

    Args:
        model: An instantiated `tim.models.t2i.tim_model.TiM` (or compatible).
        rank: LoRA rank `r`. VGG-Flow uses 8 by default. Bump to 32-128 if reward
              starts saturating early.
        alpha: LoRA scaling. Effective scale at inference is `alpha / rank`.
              Setting `alpha == rank` gives unit scale, which is the simplest case
              for ablations.
        dropout: LoRA dropout. 0.0 unless we see overfitting.
        target_modules: list of suffix matchers for which Linears get a LoRA.
                        Defaults to `DEFAULT_LORA_TARGETS` (all 8 per block).
        adapter_name: peft adapter name. Multiple named adapters can coexist on the
                      same model. Useful for compose-multiple-rewards inference.

    Returns:
        The model wrapped by peft, with LoRA layers active and only LoRA params
        in `requires_grad=True`.

    Notes:
        - peft's `get_peft_model` freezes everything except LoRA params automatically.
        - We do NOT touch non-attention/non-MLP layers (e.g., `t_embedder`,
          `delta_embedder`, `final_layer`, `x_embedder`). They remain frozen.
        - Pretrained checkpoint loading must happen BEFORE this call: peft renames
          the wrapped Linears so a strict `state_dict` load against the wrapped
          model would fail.
    """
    targets = target_modules if target_modules is not None else DEFAULT_LORA_TARGETS
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=targets,
        # Don't try to detect task type, TiM is a custom architecture,
        # peft's auto-detection doesn't know about it.
        task_type=None,
    )
    wrapped = get_peft_model(model, lora_cfg, adapter_name=adapter_name)
    return wrapped


def num_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lora_param_groups(model: nn.Module, lr: float, weight_decay: float = 0.0):
    """Yield param groups containing only LoRA params, suitable for AdamW."""
    return [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            "lr": lr,
            "weight_decay": weight_decay,
        }
    ]


def iter_lora_layers(model: nn.Module) -> Iterable[LoraLayer]:
    """Walk the model and yield every peft LoraLayer."""
    for module in model.modules():
        if isinstance(module, LoraLayer):
            yield module


def set_adapter_enabled(model: nn.Module, enabled: bool, adapter_name: str = "default") -> None:
    """Toggle whether a named LoRA adapter is active during forward.

    Used by VGG-Flow's algorithm: compute "reference velocity" with adapters DISABLED
    (returns the base flow exactly) and "current velocity" with adapters ENABLED.
    """
    if enabled:
        model.set_adapter(adapter_name)
        try:
            model.enable_adapter_layers()
        except AttributeError:
            pass
    else:
        try:
            model.disable_adapter_layers()
        except AttributeError:
            pass
