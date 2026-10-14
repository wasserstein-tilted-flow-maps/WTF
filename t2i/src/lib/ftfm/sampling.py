"""Sampling + grid utilities for FTFM eval events.

They use TiM's native scheduler (`bundle.scheduler.sample`).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


@torch.no_grad()
def sample_with_cfg(model, bundle, cap_feats, *,
                    num_steps: int, cfg_scale: float,
                    height: int, width: int, seed: int, device) -> torch.Tensor:
    """Run TiM's transition sampler against the given (LoRA-wrapped) model.
    Returns final latents (B, C, Hl, Wl) in fp32."""
    null = bundle.null_cap_feat[:, :cap_feats.size(1)].expand(cap_feats.size(0), -1, -1)
    Hl, Wl = (height // bundle.spatial_downsample, width // bundle.spatial_downsample)
    in_ch = bundle.config.model.network.params.in_channels
    g = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn((cap_feats.size(0), in_ch, Hl, Wl),
                    generator=g, device=device, dtype=bundle.weight_dtype)
    samples = bundle.scheduler.sample(
        model, cap_feats, null, z, T_max=1.0, T_min=0.0,
        num_steps=num_steps, cfg_scale=cfg_scale, cfg_low=0.0, cfg_high=1.0,
    )
    return samples[-1].to(torch.float32)


@torch.no_grad()
def latents_to_uint8_grid(decode_to_image_fn, bundle, latents: torch.Tensor,
                          n_cols: int = 8) -> np.ndarray:
    """Decode + tile (B, C, Hl, Wl) latents into a (n_rows*H, n_cols*W, 3) uint8
    grid. Default n_cols=8 → 8×8 with B=64. Pads with zeros if B isn't a
    multiple of n_cols.

    `decode_to_image_fn(bundle, latents) -> images in [0, 1]` is the
    no-grad version of the decoder (saves memory during eval).
    """
    imgs = decode_to_image_fn(bundle, latents.to(bundle.weight_dtype))
    imgs = imgs.clamp(0, 1).cpu().float().numpy()
    imgs = (imgs.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    B, H, W, _ = imgs.shape
    n_rows = (B + n_cols - 1) // n_cols
    pad = n_rows * n_cols - B
    if pad > 0:
        imgs = np.concatenate([imgs, np.zeros((pad, H, W, 3), dtype=np.uint8)], axis=0)
    rows = [np.concatenate(list(imgs[r*n_cols:(r+1)*n_cols]), axis=1) for r in range(n_rows)]
    return np.concatenate(rows, axis=0)


def parse_nfes(s: str) -> list[int]:
    """Parse comma-separated NFE list, e.g. '1,30' → [1, 30]."""
    return [int(x) for x in s.split(",") if x.strip()]
