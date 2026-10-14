"""LoRA adapters for DMFT concept-slider training and inference.

Each LoRALinear wraps a frozen nn.Linear and holds a *named* dict of adapters.
Every adapter is a (A, B) low-rank pair with its own scale η and enabled flag:

    forward(x) = W x + b + Σ_name (α_name / r_name) · η_name · (B_name A_name) x

  * Training trains a single named adapter (default: "default") with η = 1
    fixed; A is Kaiming-init, B is zero-init so LoRA(x) = 0 at step 0 and the
    base forward is preserved bit-for-bit.
  * At inference you can `add_adapter("hps")`, load its state dict, then
    `add_adapter("imreward")` and load another, scales combine additively,
    so multiple reward sliders compose for free.
  * `lora_disabled(model)` temporarily zeros all adapters, which is what the
    training loss uses to get a pure-base forward for `v_teacher` without
    deepcopying the backbone.
"""
import math
from contextlib import contextmanager
from typing import Sequence, Tuple, List, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_LORA_TARGETS: Tuple[str, ...] = (
    "attn.qkv",
    "attn.proj",
    "mlp.fc1",
    "mlp.fc2",
)

DEFAULT_ADAPTER = "default"


class LoRAAdapter(nn.Module):
    """One low-rank pair (A, B) with an inference-time scale knob.

    Optional tcf conditioning (`tcf_dim > 0`): an AdaLN-style scale+shift on
    the rank-r hidden state h = A·x, parameterised by `tcf_gamma`, `tcf_beta`
    Linears that map a per-sample tcf embedding (D = `tcf_dim`) → (r,). Both
    are zero-initialised so at step 0 the modulation is identity (h_mod = h)
    and the adapter output is bit-identical to the un-conditioned LoRA. Once
    those weights start to learn, the LoRA's INTERNAL computation depends on
    tcf, i.e. BA·x is no longer just a tcf-scaled fixed direction. This is
    what the slider's eq. 50 promise (u_{s,t} conditioned on the reward
    scales) requires; the scalar `tcf_gate` alone can only express amplitude
    scaling.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: int,
        tcf_dim: int = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.alpha_over_rank = float(alpha) / float(rank)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Plain python floats, no GPU sync on read and trivial to toggle.
        self.scale: float = 1.0
        self.enabled: bool = True
        # Optional per-sample multiplicative gate on the LoRA OUTPUT.
        # MeanFlow-style: at gate=0 the LoRA contribution is structurally 0,
        # so tcf=0 → exact base (regardless of what (A,B) learned).
        # None means "no gate" (= scalar 1.0). Tensor of shape (B,) means
        # per-sample gate broadcast over feature dims at forward time.
        self.tcf_gate = None
        # Optional AdaLN-on-rank-r tcf conditioning (see class docstring).
        self.tcf_dim = int(tcf_dim)
        if self.tcf_dim > 0:
            self.tcf_gamma = nn.Linear(self.tcf_dim, rank, bias=True)
            self.tcf_beta  = nn.Linear(self.tcf_dim, rank, bias=True)
            nn.init.constant_(self.tcf_gamma.weight, 0)
            nn.init.constant_(self.tcf_gamma.bias,   0)
            nn.init.constant_(self.tcf_beta.weight,  0)
            nn.init.constant_(self.tcf_beta.bias,    0)
        else:
            self.tcf_gamma = None
            self.tcf_beta  = None
        # Transient: per-forward tcf embedding (B, tcf_dim). Set by the model
        # via `set_lora_tcf_embed(...)` before calling forward.
        self.tcf_embed = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.lora_A.to(x.dtype)
        b = self.lora_B.to(x.dtype)
        h = F.linear(x, a)  # (B, ..., r)
        if self.tcf_embed is not None and self.tcf_gamma is not None:
            te = self.tcf_embed.to(device=h.device, dtype=h.dtype)
            gamma = self.tcf_gamma(te)  # (B, r)
            beta  = self.tcf_beta(te)   # (B, r)
            # Broadcast (B, r) over any extra middle dims of h, e.g. (B, L, r).
            view = (-1,) + (1,) * (h.ndim - 2) + (h.shape[-1],)
            h = h * (1.0 + gamma.view(*view)) + beta.view(*view)
        out = (self.alpha_over_rank * self.scale) * F.linear(h, b)
        if self.tcf_gate is not None:
            gate = self.tcf_gate.to(device=out.device, dtype=out.dtype)
            out = gate.view(-1, *([1] * (out.ndim - 1))) * out
        return out


class LoRALinear(nn.Module):
    """Frozen Linear + named adapter dict. Residuals sum across enabled adapters."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.adapters = nn.ModuleDict()

    def add_adapter(self, name: str, rank: int, alpha: int, tcf_dim: int = 0) -> LoRAAdapter:
        assert name not in self.adapters, f"adapter {name!r} already exists"
        ad = LoRAAdapter(self.in_features, self.out_features, rank, alpha, tcf_dim=tcf_dim)
        self.adapters[name] = ad
        return ad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        for ad in self.adapters.values():
            if ad.enabled:
                out = out + ad(x)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Module-tree mutation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _collect_targets(root: nn.Module, suffixes: Sequence[str]) -> List[Tuple[nn.Module, str, nn.Linear]]:
    """(parent, attr, base_linear) for every nn.Linear whose qualified name
    ends in one of `suffixes`. Materialized so callers can mutate the tree."""
    hits: List[Tuple[nn.Module, str, nn.Linear]] = []
    for name, module in list(root.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(s) for s in suffixes):
            continue
        if "." in name:
            parent_name, _, attr = name.rpartition(".")
            parent = root.get_submodule(parent_name)
        else:
            parent, attr = root, name
        hits.append((parent, attr, module))
    return hits


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: int = 16,
    targets: Sequence[str] = DEFAULT_LORA_TARGETS,
    adapter_name: str = DEFAULT_ADAPTER,
    tcf_dim: int = 0,
) -> int:
    """Wrap every `targets` nn.Linear with a LoRALinear carrying one adapter
    named `adapter_name`. Freezes all non-LoRA params. Idempotent on re-call
    with a new adapter_name: existing LoRALinear wrappers just grow another
    adapter (needed for multi-slider inference).

    `tcf_dim`: if > 0, every newly-created adapter gets AdaLN-on-rank-r tcf
    conditioning (extra params: `tcf_gamma`, `tcf_beta`, both zero-init), and
    those params are kept trainable along with `lora_A`/`lora_B`. The model
    must call `set_lora_tcf_embed(self, embed)` once per forward to feed the
    per-sample tcf embedding (B, tcf_dim).

    Returns the number of adapter instances created (= wrapped layers on the
    first call; same number on subsequent calls that add another adapter)."""
    # First pass: wrap raw Linears that haven't been wrapped yet.
    new_wraps = _collect_targets(model, targets)
    for parent, attr, base in new_wraps:
        setattr(parent, attr, LoRALinear(base))

    # Second pass: add the named adapter to every LoRALinear whose inner base
    # matches one of the targets.
    n_added = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if not any(name.endswith(s) for s in targets):
                continue
            if adapter_name in module.adapters:
                continue
            module.add_adapter(adapter_name, rank=rank, alpha=alpha, tcf_dim=tcf_dim)
            n_added += 1

    # Freeze everything that isn't a LoRA-trainable param. The trainable set
    # is: lora_A, lora_B, and (if tcf-conditional) tcf_gamma/tcf_beta, all
    # under `.adapters.<name>.`. Plus the model-level `lora_tcf_embedder`
    # module (a small TimestepEmbedder living on the model root) that
    # produces the per-sample tcf embedding consumed by the per-adapter
    # AdaLN-on-rank-r modulation. Without unfreezing this, the embedder
    # would output a constant vector and the AdaLN modulation could not
    # actually depend on tcf.
    trainable_suffixes = (".lora_A", ".lora_B",
                          ".tcf_gamma.weight", ".tcf_gamma.bias",
                          ".tcf_beta.weight",  ".tcf_beta.bias")
    for name, p in model.named_parameters():
        is_adapter_trainable = (".adapters." in name
                                and any(name.endswith(s) for s in trainable_suffixes))
        is_tcf_embedder = ("lora_tcf_embedder." in name)
        if is_adapter_trainable or is_tcf_embedder:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    return n_added


# ──────────────────────────────────────────────────────────────────────────────
# Runtime controls
# ──────────────────────────────────────────────────────────────────────────────

def _iter_adapters(model: nn.Module, name: Optional[str] = None) -> Iterable[LoRAAdapter]:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            if name is None:
                for ad in module.adapters.values():
                    yield ad
            elif name in module.adapters:
                yield module.adapters[name]


def set_lora_scale(model: nn.Module, scale: float, name: Optional[str] = None) -> None:
    s = float(scale)
    for ad in _iter_adapters(model, name):
        ad.scale = s


def set_lora_enabled(model: nn.Module, enabled: bool, name: Optional[str] = None) -> None:
    flag = bool(enabled)
    for ad in _iter_adapters(model, name):
        ad.enabled = flag


def set_lora_tcf_gate(model: nn.Module, gate, name: Optional[str] = None) -> None:
    """Set the per-sample multiplicative gate on every LoRA adapter's output.

    `gate` is either None (no gating, equivalent to scalar 1.0) or a (B,)
    tensor that gets broadcast over feature dims at forward time. MeanFlow
    parameterization: at gate=0, the LoRA contribution is structurally 0
    regardless of what (A, B) learned, gives an exact off-switch when the
    gate is computed from tcf via e.g. `1 - 1/(1+tcf)`."""
    for ad in _iter_adapters(model, name):
        ad.tcf_gate = gate


def set_lora_tcf_embed(model: nn.Module, embed, name: Optional[str] = None) -> None:
    """Set the per-sample tcf embedding (B, tcf_dim) on every LoRA adapter.

    Used together with `apply_lora(..., tcf_dim=D)` so each adapter's
    AdaLN-on-rank-r modulation can produce tcf-conditional INTERNAL
    behaviour (not just amplitude scaling like `tcf_gate`). Pass None to
    clear and recover plain LoRA forward (h_mod = h)."""
    for ad in _iter_adapters(model, name):
        ad.tcf_embed = embed


@contextmanager
def lora_disabled(model: nn.Module, name: Optional[str] = None):
    """Temporarily turn (named or all) LoRA adapters off on `model`."""
    ads = list(_iter_adapters(model, name))
    prev = [ad.enabled for ad in ads]
    for ad in ads:
        ad.enabled = False
    try:
        yield model
    finally:
        for ad, f in zip(ads, prev):
            ad.enabled = f


# ──────────────────────────────────────────────────────────────────────────────
# Param iteration + checkpoint I/O
# ──────────────────────────────────────────────────────────────────────────────

_LORA_TRAINABLE_SUFFIXES: Tuple[str, ...] = (
    ".lora_A", ".lora_B",
    ".tcf_gamma.weight", ".tcf_gamma.bias",
    ".tcf_beta.weight",  ".tcf_beta.bias",
)


def lora_parameters(model: nn.Module, name: Optional[str] = None):
    """Iterator over LoRA-only parameters, A, B, and (if present) the
    AdaLN-on-rank-r tcf-conditioning weights tcf_gamma/tcf_beta. Optionally
    scoped to one adapter name. Also includes the model-level
    `lora_tcf_embedder.*` parameters when present (only when name is None,
    since the embedder is shared across all adapters on the model)."""
    needle = f".adapters.{name}." if name is not None else ".adapters."
    for pname, p in model.named_parameters():
        if needle in pname and any(pname.endswith(s) for s in _LORA_TRAINABLE_SUFFIXES):
            yield p
        elif name is None and "lora_tcf_embedder." in pname:
            yield p


def lora_state_dict(model: nn.Module, name: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """State dict containing only LoRA adapter weights.
    When `name` is given, filter to that adapter. Otherwise include all.
    Returned tensors are independent copies, safe to mutate the model afterwards."""
    needle = f".adapters.{name}." if name is not None else ".adapters."
    # .cpu().clone() guarantees independent storage; a bare .cpu() on a CPU
    # tensor aliases the parameter, so later training steps would overwrite
    # the "saved" state.
    out = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if needle in k and any(k.endswith(s) for s in _LORA_TRAINABLE_SUFFIXES)
    }
    # Also include the model-level lora_tcf_embedder.* (if present) when no
    # specific adapter name is requested. The embedder is shared across all
    # adapters, so it should travel with full-model state dicts but not with
    # per-adapter snapshots.
    if name is None:
        for k, v in model.state_dict().items():
            if "lora_tcf_embedder." in k:
                out[k] = v.detach().cpu().clone()
    return out


def load_lora_state_dict(
    model: nn.Module,
    sd: Dict[str, torch.Tensor],
    name: Optional[str] = None,
    strict: bool = True,
):
    """Load a LoRA state dict. If `name` is given, the saved keys' adapter
    namespace is rewritten on the fly so a checkpoint saved under e.g.
    ".adapters.default." can be loaded into ".adapters.hps." on inference."""
    if name is not None:
        remapped = {}
        for k, v in sd.items():
            # Rewrite ".adapters.<saved_name>." → ".adapters.<name>."
            idx = k.find(".adapters.")
            if idx < 0:
                remapped[k] = v
                continue
            after = k[idx + len(".adapters."):]
            # saved adapter name = up to next "."
            saved_name, _, rest = after.partition(".")
            remapped[k[:idx] + ".adapters." + name + "." + rest] = v
        sd = remapped
    result = model.load_state_dict(sd, strict=False)
    if strict and result.unexpected_keys:
        raise RuntimeError(f"Unexpected keys in LoRA state dict: {result.unexpected_keys}")
    return result


def num_lora_parameters(model: nn.Module, name: Optional[str] = None) -> int:
    return sum(p.numel() for p in lora_parameters(model, name))


def adapter_names(model: nn.Module) -> List[str]:
    """Names of adapters present in the model (order of first-seen traversal)."""
    seen: List[str] = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            for k in module.adapters.keys():
                if k not in seen:
                    seen.append(k)
    return seen
