# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

from functools import partial

import torch
from torch import Tensor
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed
from .lora import set_lora_tcf_gate, set_lora_tcf_embed
from .layers import (
    TimestepEmbedder,
    LabelEmbedder,
    DiTBlock,
    FinalLayer,
    get_2d_sincos_pos_embed,
    timestep_embedding
)


def _as_1d(v, B: int, device, dtype):
    """Coerce scalar / (B,) / (B,1,1,1) → (B,) tensor for TimestepEmbedder."""
    if isinstance(v, (int, float)):
        return torch.full((B,), float(v), device=device, dtype=dtype)
    v = v.to(device=device, dtype=dtype)
    return v.reshape(-1)


"""Decoupled MeanFlow Transformer (DMFT)"""
class DMFT(nn.Module):
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_cfg_embedding: bool = True,
        num_classes: int = 1000,
        dmf_depth: int = 8,
        use_logvar: bool = False,
        use_token_cond: bool = False,
        # Unused by the adaLN path.
        num_tcf_tokens: int = 4,
        num_interval_tokens: int = 2,
        **block_kwargs # fused_attn, qk_norm, post_ln
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.dmf_depth = dmf_depth
        self.use_logvar = use_logvar
        self.use_token_cond = use_token_cond

        # model
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, use_cfg_embedding)
        # Will use fixed sin-cos embedding:
        self.num_patches = self.x_embedder.num_patches
        # (input_size // patch_size)**2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), requires_grad=False
        )
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        if self.use_logvar:
            self.logvar_t_embedder = partial(timestep_embedding, dim=128)
            self.logvar_linear  = nn.Linear(256, 1)

        # Token-cond conditioning via adaLN (replaces the earlier prefix-token
        # design that didn't differentiate in practice). Each of tcf, t_min,
        # t_max goes through its own TimestepEmbedder → zero-init Linear and is
        # added into the DiT block modulation signal `c` (same pathway as t/r/y).
        # adaLN gives conditioning ~100% of the block's bandwidth instead of the
        # ~3% attention share the prefix tokens got, which is what blocked tcf
        # differentiation in the tokens run. Zero-init on the final Linear
        # preserves the pretrained backbone exactly at step 0; gradient still
        # flows into that Linear (∂L/∂W ≠ 0 even when W = 0), so the
        # conditioning pathway comes online within the first few steps.
        if self.use_token_cond:
            self.tcf_embedder  = TimestepEmbedder(hidden_size)
            self.tmin_embedder = TimestepEmbedder(hidden_size)
            self.tmax_embedder = TimestepEmbedder(hidden_size)
            self.tcf_linear  = nn.Linear(hidden_size, hidden_size)
            self.tmin_linear = nn.Linear(hidden_size, hidden_size)
            self.tmax_linear = nn.Linear(hidden_size, hidden_size)
            # Separate trainable embedder used SOLELY by the LoRA's
            # AdaLN-on-rank-r tcf modulation (`apply_lora(..., tcf_dim=D)`).
            # Lives outside the backbone-conditioning path so the frozen
            # backbone is never influenced by tcf, only the trainable LoRA's
            # internal computation is. Receives raw tcf (better resolution
            # across [0, 100] than the bounded `tcf_in = 1 − 1/(1+tcf)` used
            # for the backbone cond_embed pathway).
            self.lora_tcf_embedder = TimestepEmbedder(hidden_size)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.use_logvar:
            nn.init.constant_(self.logvar_linear.weight, 0)
            nn.init.constant_(self.logvar_linear.bias, 0)

        if self.use_token_cond:
            # Embedders: standard small-Gaussian init on both MLP layers so they
            # have a non-zero "useful range" once the downstream Linear picks
            # up signal.
            nn.init.normal_(self.tcf_embedder.mlp[0].weight,  std=0.02)
            nn.init.normal_(self.tcf_embedder.mlp[2].weight,  std=0.02)
            nn.init.normal_(self.tmin_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.tmin_embedder.mlp[2].weight, std=0.02)
            nn.init.normal_(self.tmax_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.tmax_embedder.mlp[2].weight, std=0.02)
            # Output projection into the adaLN condition, zero-initialized so step 0 matches the pretrained model.
            nn.init.constant_(self.tcf_linear.weight,  0)
            nn.init.constant_(self.tcf_linear.bias,    0)
            nn.init.constant_(self.tmin_linear.weight, 0)
            nn.init.constant_(self.tmin_linear.bias,   0)
            nn.init.constant_(self.tmax_linear.weight, 0)
            nn.init.constant_(self.tmax_linear.bias,   0)
            # LoRA-only tcf embedder with small Gaussian init. Step 0 matches the base model through the
            # zero-initialized tcf_gamma and tcf_beta inside each LoRA adapter.
            nn.init.normal_(self.lora_tcf_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.lora_tcf_embedder.mlp[2].weight, std=0.02)

    def unpatchify(self, x: Tensor) -> Tensor:
        c, p = self.out_channels, self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def _cond_embed(self, tcf, t_min, t_max, B: int, device, dtype):
        """Build the additive conditioning vector (B, 1, D) from (tcf, t_min, t_max).

        tcf ∈ [0, ∞) is mapped through 1 − 1/(1+tcf) ∈ [0, 1) before the
        sinusoidal embedder (matches imeanflow's 1 − 1/w reparameterization).
        t_min, t_max ∈ [0, 1] are passed through directly. Each embedder output
        is projected by a zero-init Linear, so this returns 0 at init.
        """
        tcf_1d  = _as_1d(tcf,   B, device, dtype)
        tmin_1d = _as_1d(t_min, B, device, dtype)
        tmax_1d = _as_1d(t_max, B, device, dtype)
        tcf_in  = 1.0 - 1.0 / (1.0 + tcf_1d.clamp(min=0.0))
        tcf_embed  = self.tcf_linear(self.tcf_embedder(tcf_in))    # (B, D)
        tmin_embed = self.tmin_linear(self.tmin_embedder(tmin_1d)) # (B, D)
        tmax_embed = self.tmax_linear(self.tmax_embedder(tmax_1d)) # (B, D)
        cond = (tcf_embed + tmin_embed + tmax_embed).unsqueeze(1)  # (B, 1, D)
        return cond

    def forward(
        self,
        x,
        t,
        r,
        y,
        tcf=0.0,
        t_min=0.0,
        t_max=1.0,
        return_logvar: bool = False,
    ):
        t, r = t.reshape(-1), r.reshape(-1)

        x = self.x_embedder(x) + self.pos_embed  # (B, L, D)
        B, L, D = x.shape

        t_embed = self.t_embedder(t).reshape(B, -1, D)
        y_embed = self.y_embedder(y).reshape(B, -1, D)

        # Conditioning gets injected into the adaLN pathway, same channel as
        # t/r/y, via a zero-init additive term. See _cond_embed for details.
        cond_embed = 0.0
        if self.use_token_cond:
            cond_embed = self._cond_embed(tcf, t_min, t_max, B, x.device, x.dtype)
            tcf_1d = _as_1d(tcf, B, x.device, x.dtype).reshape(B)
            # (1) Multiplicative gate on the LoRA OUTPUT: raw `tcf`. Gives the
            # structural anchor `output(tcf=0) ≡ base flow` (gate=0 zeros out
            # the LoRA contribution regardless of what (A,B,γ,β) learned) and
            # bakes in the linear-in-tcf prior matched to the loss target's
            # tcf-scaling `g_hat ∝ tcf·(B/G)·∇R`.
            lora_gate = tcf_1d.clamp(min=0.0)
            set_lora_tcf_gate(self, lora_gate)
            # (2) Per-sample tcf embedding consumed by each LoRA adapter's
            # internal AdaLN-on-rank-r modulation (`apply_lora(..., tcf_dim)`):
            # makes BA·x itself tcf-CONDITIONAL, i.e. the LoRA's effective
            # direction can shift with tcf, not just its amplitude. This is
            # the structural change that lets `model(x; tcf=k)` reproduce the
            # behaviour of a model trained at fixed tcf=k (eq. 50 in the
            # paper), which the gate alone cannot deliver.
            lora_tcf_embed = self.lora_tcf_embedder(tcf_1d.clamp(min=0.0))  # (B, D)
            set_lora_tcf_embed(self, lora_tcf_embed)

        c_t = t_embed + y_embed + cond_embed

        # Encoder blocks
        for _, block in enumerate(self.blocks[:self.dmf_depth]):
            x = block(x, c_t)
        # Decoder blocks
        r_embed = self.t_embedder(r).reshape(B, -1, D)
        c_r = r_embed + y_embed + cond_embed
        for _, block in enumerate(self.blocks[self.dmf_depth:]):
            x = block(x, c_r)

        x = self.final_layer(x, c_r)
        x = self.unpatchify(x)
        if return_logvar:
            logvar = self.logvar_linear(
                torch.cat([self.logvar_t_embedder(t), self.logvar_t_embedder(r)], dim=1)
            )
            logvar = logvar.view(-1, *[1] * (x.ndim - 1))
            return x, logvar
        return x


DMFT_XL_2 = partial(DMFT, depth=28, hidden_size=1152, patch_size=2, num_heads=16)
DMFT_XL_4 = partial(DMFT, depth=28, hidden_size=1152, patch_size=4, num_heads=16)
DMFT_XL_8 = partial(DMFT, depth=28, hidden_size=1152, patch_size=8, num_heads=16)
DMFT_L_2  = partial(DMFT, depth=24, hidden_size=1024, patch_size=2, num_heads=16)
DMFT_L_4  = partial(DMFT, depth=24, hidden_size=1024, patch_size=4, num_heads=16)
DMFT_L_8  = partial(DMFT, depth=24, hidden_size=1024, patch_size=8, num_heads=16)
DMFT_M_2  = partial(DMFT, depth=16, hidden_size=1024, patch_size=2, num_heads=16)
DMFT_M_4  = partial(DMFT, depth=16, hidden_size=1024, patch_size=4, num_heads=16)
DMFT_M_8  = partial(DMFT, depth=16, hidden_size=1024, patch_size=8, num_heads=16)
DMFT_B_2  = partial(DMFT, depth=12, hidden_size=768, patch_size=2, num_heads=12)
DMFT_B_4  = partial(DMFT, depth=12, hidden_size=768, patch_size=4, num_heads=12)
DMFT_B_8  = partial(DMFT, depth=12, hidden_size=768, patch_size=8, num_heads=12)
DMFT_S_2  = partial(DMFT, depth=12, hidden_size=384, patch_size=2, num_heads=6)
DMFT_S_4  = partial(DMFT, depth=12, hidden_size=384, patch_size=4, num_heads=6)
DMFT_S_8  = partial(DMFT, depth=12, hidden_size=384, patch_size=8, num_heads=6)

DMFT_models = {
    "DMFT-XL/2": DMFT_XL_2,  "DMFT-XL/4": DMFT_XL_4,  "DMFT-XL/8": DMFT_XL_8,
    "DMFT-L/2":  DMFT_L_2,   "DMFT-L/4":  DMFT_L_4,   "DMFT-L/8":  DMFT_L_8,
    "DMFT-B/2":  DMFT_B_2,   "DMFT-B/4":  DMFT_B_4,   "DMFT-B/8":  DMFT_B_8,
    "DMFT-S/2":  DMFT_S_2,   "DMFT-S/4":  DMFT_S_4,   "DMFT-S/8":  DMFT_S_8,
}
