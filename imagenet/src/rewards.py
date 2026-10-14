import torch
import torch.nn as nn


class RednessReward(nn.Module):
    """
    Trivial differentiable reward: how "red" an image is.

    Score per image = mean(R) − 0.5·(mean(G) + mean(B)), with channels in [0, 1].
    Range ≈ [−0.75, +1.0]. Prompts are ignored. Kept in the signature to match
    HPSv2 / ImageReward / PickScorer interfaces so it's drop-in compatible
    with the reward_fn.score(images, prompts) call in train.py.
    """

    def score(self, images: torch.Tensor, prompts=None) -> torch.Tensor:
        # images: [B, 3, H, W], expected in [0, 1] after the (x+1)/2 step.
        r = images[:, 0].mean(dim=(-1, -2))
        g = images[:, 1].mean(dim=(-1, -2))
        b = images[:, 2].mean(dim=(-1, -2))
        return r - 0.5 * (g + b)


class BluenessReward(nn.Module):
    """Mirror of RednessReward for the blue channel: B − 0.5·(R + G)."""

    def score(self, images: torch.Tensor, prompts=None) -> torch.Tensor:
        r = images[:, 0].mean(dim=(-1, -2))
        g = images[:, 1].mean(dim=(-1, -2))
        b = images[:, 2].mean(dim=(-1, -2))
        return b - 0.5 * (r + g)
