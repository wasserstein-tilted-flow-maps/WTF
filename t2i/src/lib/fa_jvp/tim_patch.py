"""Monkey-patch TiM's `tim.models.t2i.tim_model.flash_attn_func` to use the
JVP-aware Flash Attention 2 from `lib.fa_jvp` (vendored from kyungmnlee/dmf).

Why monkey-patch instead of forking TiM:
- TiM is a third-party pre-trained model. Replacing its `Attention` class would
  require a weight-migration mess.
- The ImageNet lane builds its own model (DMFT) and selects FA-JVP at construction
  time via `Attention(attn_func="fa2")`; TiM's class signature cannot be changed
  that way.
- The dmf FA-JVP function IS a drop-in for stock `flash_attn_func`: same forward
  + backward via `flash_attn_2_cuda` extension; only the JVP path uses a custom
  Triton kernel. Forward-pass output should be byte-identical to stock.

Why a `dropout_p` shim:
- Stock `flash_attn_func` accepts `dropout_p`. dmf's version does not.
- TiM's config sets `attn_drop=0` (default; not overridden in
  `configs/t2i/tim_xl_p1_t2i.yaml`), so the call site always passes
  `dropout_p=0`. The shim asserts this.
"""
from __future__ import annotations

import torch.autograd.forward_ad as fwAD
from . import flash_attn_func as _fa_jvp_func



def _coerce_dual_dtypes(q, k, v):
    """Ensure primal+tangent of q,k,v all share the same dtype before the
    Triton kernel. TiM's RoPE multiplies bf16 q with fp32 freqs_cos/freqs_sin,
    then casts back via q.to(ori_dtype). Under torch.func.jvp, the cast may
    not propagate cleanly to the tangent, leaving tangent in fp32 while
    primal is bf16. The kernel's tl.dot then errors on mixed dtypes.

    We force-cast tangent to primal.dtype on a per-tensor basis. Since
    primal is the "trusted" dtype (the model's compute precision), this is
    the safe direction: down-cast precision on tangent to match.
    """
    import torch
    out = []
    for t in (q, k, v):
        primal, tangent = fwAD.unpack_dual(t)
        if tangent is None:
            out.append(t)
            continue
        if tangent.dtype != primal.dtype:
            tangent = tangent.to(primal.dtype)
            t = fwAD.make_dual(primal, tangent)
        out.append(t)
    return tuple(out)


def _flash_attn_func_with_dropout_shim(q, k, v, dropout_p=0.0, *args, **kwargs):
    """Drop-in wrapper for stock `flash_attn_func(q, k, v, dropout_p=...)`.

    Three jobs:
    1. Absorb `dropout_p` kwarg (dmf's FA-JVP doesn't accept it; TiM uses 0 always).
    2. Coerce primal/tangent dtypes to match (TiM's RoPE leaks fp32 into tangent).
    3. Pass through to the FA-JVP function.
    """
    import torch
    import torch.autograd.forward_ad as _fwAD

    assert dropout_p == 0.0, (
        f"FA-JVP shim assumes dropout_p=0 (TiM-T2I default). Got {dropout_p}."
    )

    q, k, v = _coerce_dual_dtypes(q, k, v)
    return _fa_jvp_func(q, k, v)


def apply_to_tim(model=None, dtype=None):
    """Monkey-patch TiM's flash_attn_func symbol AND the model's `get_rope`
    method so that RoPE `freqs_cos`/`freqs_sin` are cast to the model's compute
    dtype (bf16) before they multiply q/k.

    Why both patches are needed:
    - flash_attn_func patch: redirect to FA-JVP for JVP support.
    - RoPE dtype patch: TiM's RoPE buffers are fp32 by default. Multiplying
      bf16 q with fp32 freqs promotes to fp32, then `q.to(ori_dtype)` casts
      primal back to bf16, but under torch.func.jvp, the tangent does NOT
      get the cast (PyTorch fwAD quirk). Tangent stays fp32 while primal is
      bf16 → kernel dtype mismatch. Casting freqs to bf16 BEFORE the
      multiplication keeps the entire computation bf16 → no fp32 leak.

    Args:
        model: the TiM root model (`bundle.model` / `peft_model.base_model.model`).
            If passed, we patch its `get_rope` method. If None, only flash_attn
            patch is applied (caller must guarantee freqs are already in compute dtype).
        dtype: target dtype for RoPE freqs. Defaults to bf16 (TiM's compute dtype).

    Returns:
        dict with keys for restoring previous state via `revert_tim(...)`.
    """
    import torch
    import tim.models.t2i.tim_model as _tim_mod

    if dtype is None:
        dtype = torch.bfloat16

    state = {"flash_attn_func": _tim_mod.flash_attn_func, "model_get_rope": None, "model": model}
    _tim_mod.flash_attn_func = _flash_attn_func_with_dropout_shim

    if model is not None:
        prev_get_rope = model.get_rope
        _patch_called = [0]
        def patched_get_rope(self, h, w, attn_type):
            _patch_called[0] += 1
            fc, fs = prev_get_rope(h, w, attn_type)
            if _patch_called[0] == 1:
                print(f"  [tim_patch] RoPE tables cast from {fc.dtype} to {dtype}", flush=True)
            return fc.to(dtype), fs.to(dtype)
        import types
        model.get_rope = types.MethodType(patched_get_rope, model)
        state["model_get_rope"] = prev_get_rope
        state["_patch_call_counter"] = _patch_called

        # Patch q_norm / k_norm of every Attention block.
        # TiM uses nn.LayerNorm by default for q_norm, k_norm. nn.LayerNorm computes
        # mean/var in fp32 internally and does NOT propagate the dtype cast to the
        # JVP tangent, leaving tangent in fp32 while primal is bf16. The kernel
        # then errors on dtype mismatch.
        # Replace forward with a bf16-pure version that keeps both primal and
        # tangent in the model's compute dtype.
        import torch
        import torch.nn as nn
        patched_norms = []
        for module in model.modules():
            for child_name in ("q_norm", "k_norm"):
                if hasattr(module, child_name):
                    norm_module = getattr(module, child_name)
                    if isinstance(norm_module, nn.LayerNorm):
                        orig_forward = norm_module.forward
                        ln_eps = norm_module.eps
                        ln_weight = norm_module.weight
                        ln_bias = norm_module.bias
                        ln_norm_shape = norm_module.normalized_shape
                        def make_pure(eps, weight, bias, norm_shape):
                            def pure_layernorm_forward(x):
                                # Match nn.LayerNorm but stay in input dtype throughout
                                in_dtype = x.dtype
                                mean = x.mean(dim=tuple(range(-len(norm_shape), 0)), keepdim=True)
                                var = ((x - mean) ** 2).mean(dim=tuple(range(-len(norm_shape), 0)), keepdim=True)
                                normed = (x - mean) * torch.rsqrt(var + eps)
                                if weight is not None:
                                    normed = normed * weight.to(in_dtype)
                                if bias is not None:
                                    normed = normed + bias.to(in_dtype)
                                return normed
                            return pure_layernorm_forward
                        norm_module.forward = make_pure(ln_eps, ln_weight, ln_bias, ln_norm_shape)
                        patched_norms.append((norm_module, orig_forward))
        state["patched_norms"] = patched_norms
        print(f"  [tim_patch] patched {len(patched_norms)} q_norm/k_norm LayerNorm modules to bf16-pure", flush=True)

    return state


def revert_tim(state):
    """Undo the monkey-patch (for testing).

    `state` is the dict returned by `apply_to_tim(...)`.
    Backwards-compatible: also accepts a bare callable (the old return type).
    """
    import tim.models.t2i.tim_model as _tim_mod
    if callable(state):  # a bare flash_attn_func
        _tim_mod.flash_attn_func = state
        return
    _tim_mod.flash_attn_func = state["flash_attn_func"]
    if state.get("model") is not None and state.get("model_get_rope") is not None:
        state["model"].get_rope = state["model_get_rope"]
    for norm_module, orig_forward in state.get("patched_norms", []):
        norm_module.forward = orig_forward
