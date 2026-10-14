"""Online diversity metrics for paper-faithful logging.

Implements DreamSim + CLIP diversity exactly as the VGG-Flow paper §5.1 describes:

  "compute the variance of latent features (both DreamSim features [22] and
   CLIP features [29, 53]) extracted from a batch of generated images
   (we use a batch of size 16) to measure sample diversity."

We use **mean pairwise L2 distance over normalized feature vectors** as the
"variance" measure, matching microsoft/soc-fine-tuning-sd's `metrics.py:do_clip_score_diversity`
and `do_dreamsim_diversity` (lines 49-72 + 100-130), those are the only
released reference implementations we have.

CLIP: re-uses the openai/clip-vit-large-patch14 already loaded by aesthetic_scorer.
DreamSim: uses the actual `dreamsim` PyPI package (paper-faithful). The package
itself loads the DreamSim ensemble checkpoint via PEFT, which requires
**peft<0.19** in our env (peft 0.19 imports `EmbeddingParallel` from
`transformers.integrations.tensor_parallel`, but transformers >=4.50 removed it).
requirements.txt pins peft 0.18.1 accordingly. The LoRA APIs used here
(LoraConfig, get_peft_model) are stable across this range.

Also usable during training to log `diversity_clip` and `diversity_dreamsim` for the on-policy batch.
"""
from __future__ import annotations
from typing import Optional, Dict, List
import torch
import torch.nn.functional as F


def _mean_pairwise_l2_squared(features: torch.Tensor) -> torch.Tensor:
    """Mean of squared pairwise L2 distances over a batch of (already-normalized) feature vectors.

    For N feature vectors, returns:
        (1 / (N(N-1))) * sum_{i != j} ||f_i - f_j||²

    This matches the MS soc-fine-tuning-sd diversity definition (their `do_clip_score_diversity`
    sums upper triangle and divides by N(N-1)) and the paper's "variance of latent features"
    interpretation.

    Args:
        features: (N, D) tensor, feature vectors. Should already be L2-normalized for CLIP-style diversity.
    Returns:
        scalar tensor, diversity score.
    """
    N = features.size(0)
    if N < 2:
        return features.new_zeros(())
    # ||f_i - f_j||² = ||f_i||² + ||f_j||² - 2 ⟨f_i, f_j⟩
    sq_norms = (features * features).sum(dim=-1)            # (N,)
    gram = features @ features.T                             # (N, N)
    sq_dists = sq_norms.unsqueeze(0) + sq_norms.unsqueeze(1) - 2 * gram   # (N, N)
    sq_dists = sq_dists.clamp(min=0)                         # numerical safety
    # Sum over off-diagonal entries / (N(N-1))
    mask = torch.ones_like(sq_dists) - torch.eye(N, device=sq_dists.device, dtype=sq_dists.dtype)
    diversity = (sq_dists * mask).sum() / (N * (N - 1))
    return diversity


class DiversityComputer:
    """Computes CLIP + DreamSim diversity on a batch of images (RGB float in [0, 1]).

    Lazy-loads models on first call to keep import cheap. CLIP can reuse an existing
    HuggingFace CLIPModel (e.g. from aesthetic_scorer); DreamSim is loaded fresh
    via `from dreamsim import dreamsim`.

    Usage:
        computer = DiversityComputer(device="cuda")
        # Local-batch embedding (small, returns features for cross-rank gather):
        clip_features = computer.embed_clip(images)        # (B, 768)
        dreamsim_features = computer.embed_dreamsim(images)# (B, 1792)
        # Global diversity on rank 0:
        score_clip = mean_pairwise_l2_squared(global_clip_features)
        score_dreamsim = mean_pairwise_l2_squared(global_dreamsim_features)
    """

    def __init__(
        self,
        device: torch.device,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        dreamsim_model_name: str = "ensemble",
    ):
        # `dreamsim_model_name='ensemble'` matches paper §5.1, uses the full
        # DreamSim ensemble (DINO ViT-B/16 + CLIP ViT-B/32 + OpenCLIP ViT-B/32,
        # all fine-tuned on NIGHTS perceptual-similarity data). This requires
        # `peft<0.19` (requirements.txt pins 0.18.1); see the module docstring.
        self.device = device
        self.clip_model_name = clip_model_name
        self.dreamsim_model_name = dreamsim_model_name

        self._clip_model = None
        self._dreamsim_model = None
        self._dreamsim_preprocess = None

    # ─── CLIP embed ───────────────────────────────────────────────────────
    def _ensure_clip(self):
        if self._clip_model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor
        self._clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device).eval()
        self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        for p in self._clip_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def embed_clip(self, images: torch.Tensor) -> torch.Tensor:
        """Embed (B, 3, H, W) images in [0, 1] → (B, 768) L2-normalized CLIP features.

        Uses CLIPProcessor's standard pipeline (resize to 224×224, mean/std normalize).
        Returns features detached on the same device as images.
        """
        self._ensure_clip()
        # CLIPProcessor wants PIL or numpy. We have float [0,1] tensor on device.
        # Run a manual preprocess: resize 224 + ImageNet-CLIP norm.
        # CLIP-L norm: mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        x = images.to(self.device).float()
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        feats = self._clip_model.get_image_features(pixel_values=x)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feats.detach()

    # ─── DreamSim embed (paper-faithful: full ensemble) ──────────────────
    def _ensure_dreamsim(self):
        if self._dreamsim_model is not None:
            return
        import os
        from dreamsim import dreamsim
        cache_dir = os.environ.get(
            "DREAMSIM_CACHE",
            os.path.expanduser("~/.cache/dreamsim"),
        )
        os.makedirs(cache_dir, exist_ok=True)
        self._dreamsim_model, self._dreamsim_preprocess = dreamsim(
            pretrained=True,
            cache_dir=cache_dir,
            dreamsim_type=self.dreamsim_model_name,
            device=str(self.device),
        )
        self._dreamsim_model = self._dreamsim_model.to(self.device).eval()
        for p in self._dreamsim_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def embed_dreamsim(self, images: torch.Tensor) -> torch.Tensor:
        """Embed (B, 3, H, W) images in [0, 1] → DreamSim ensemble features.

        Uses the actual DreamSim model (paper-faithful per VGG-Flow §5.1).
        Default `dreamsim_type='ensemble'` = DINO + CLIP + OpenCLIP ensemble fine-tuned
        on NIGHTS. Embeds via `model.embed(x)`; we then L2-normalize and use the same
        mean-pairwise-L2-distance diversity metric as MS soc-fine-tuning-sd's
        `do_dreamsim_diversity`.

        Requires peft<0.19 (requirements.txt pins 0.18.1) for dreamsim's PEFT loader.
        """
        self._ensure_dreamsim()
        x = images.to(self.device).float()
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
        # DreamSim uses CLIP-style normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        feats = self._dreamsim_model.embed(x)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feats.detach()

    # ─── Compute both, returns scalar diversities ────────────────────────
    @torch.no_grad()
    def compute_diversity(
        self,
        images: torch.Tensor,
        gather_fn: Optional[callable] = None,
    ) -> Dict[str, float]:
        """Embed images locally, optionally gather features cross-rank, compute diversity.

        Args:
          images: (B_local, 3, H, W) on any device, in [0, 1] range
          gather_fn: optional callable `(features) -> gathered_features` for DDP all-gather.
                     If None, computes diversity on local batch only.

        Returns:
          dict with keys 'clip', 'dreamsim', float scalars.
        """
        clip_feats = self.embed_clip(images)
        ds_feats = self.embed_dreamsim(images)

        if gather_fn is not None:
            clip_feats = gather_fn(clip_feats)
            ds_feats = gather_fn(ds_feats)

        return {
            "clip": _mean_pairwise_l2_squared(clip_feats).item(),
            "dreamsim": _mean_pairwise_l2_squared(ds_feats).item(),
        }


# ─── Paper-faithful within-prompt diversity ───────────────────────────────
@torch.no_grad()
def compute_within_prompt_diversity(
    pipeline,
    transformer,
    prompts: list,            # M prompts
    K: int,                    # samples per prompt
    diversity_computer: DiversityComputer,
    num_inference_steps: int,
    guidance_scale: float,
    device: torch.device,
) -> dict:
    """Generate K samples per prompt for M prompts, return mean within-prompt diversity.

    Paper §5.1 anti-mode-collapse signal: for a fixed prompt, do samples vary or
    collapse to one mode? Computes mean pairwise L2² over (K samples per prompt),
    then averages across the M prompts.

    Args:
      pipeline: TiM pipeline with a `sample_trajectory` method.
      transformer: the LoRA-wrapped TiM model (or `transformer.module` if DDP).
      prompts: list of M prompts (strings).
      K: samples per prompt (paper convention: 16; our quick-monitor: 4).
      diversity_computer: shared DiversityComputer (DreamSim + CLIP).
      num_inference_steps: 20 (paper).
      guidance_scale: paper 5.0; TiM substrate 2.5.

    Returns:
      {"clip": float, "dreamsim": float}, averaged across M prompts.
    """
    M = len(prompts)
    if M == 0 or K < 2:
        return {"clip": 0.0, "dreamsim": 0.0}

    # Batch all M*K samples in one sample_trajectory call by repeating each prompt K times.
    # Avoids M separate calls (each has prompt-encode + scheduler-setup overhead).
    expanded_prompts = []
    for p in prompts:
        expanded_prompts.extend([p] * K)

    # `sample_trajectory` is a standalone function imported by `lib.training.sampler`,
    # not a method on the pipeline (despite its `self` first arg). Same call style as
    # `lib/training/sampler.py:95`.
    from lib.diffusion.sample_trajectory_tim import sample_trajectory
    outputs = sample_trajectory(
        pipeline,
        transformer,
        prompt=expanded_prompts,
        negative_prompt=None,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type="image",
        return_output=False,
    )
    # Output tuple: (images, latents, timesteps, prompt_embeds, ...)
    images = outputs[0]   # (M*K, 3, H, W)

    # Embed all M*K images at once (cheap)
    clip_feats = diversity_computer.embed_clip(images.to(device))         # (M*K, 768)
    ds_feats = diversity_computer.embed_dreamsim(images.to(device))       # (M*K, dim)

    # Group samples by prompt and compute within-prompt diversity per group
    div_clip_per_prompt = []
    div_ds_per_prompt = []
    for i in range(M):
        cf = clip_feats[i * K : (i + 1) * K]
        df = ds_feats[i * K : (i + 1) * K]
        div_clip_per_prompt.append(_mean_pairwise_l2_squared(cf).item())
        div_ds_per_prompt.append(_mean_pairwise_l2_squared(df).item())

    return {
        "clip": sum(div_clip_per_prompt) / M,
        "dreamsim": sum(div_ds_per_prompt) / M,
    }


# ─── Helper for DDP all-gather of feature tensors ─────────────────────────
def make_all_gather_fn(world_size: int):
    """Returns a callable that all-gathers a tensor across ranks and returns the concatenation
    on every rank. Returns identity if world_size=1 (single-GPU)."""
    if world_size == 1:
        return lambda x: x

    import torch.distributed as dist

    def gather(x: torch.Tensor) -> torch.Tensor:
        gathered = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(gathered, x.contiguous())
        return torch.cat(gathered, dim=0)

    return gather


# ─── Self-test ────────────────────────────────────────────────────────────
def _selftest():
    """Local sanity test: feeding the same image gives diversity≈0; feeding random gives positive."""
    print("=== Diversity computer self-test ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARN: CUDA not available; CLIP load will be slow on CPU. Skipping full test.")
        return

    computer = DiversityComputer(device)
    # 4 identical images → diversity should be ~0
    same = torch.full((4, 3, 256, 256), 0.5, device=device)
    d_same = computer.compute_diversity(same)
    print(f"  identical images: clip={d_same['clip']:.4e}, dreamsim={d_same['dreamsim']:.4e}")
    assert d_same["clip"] < 1e-4 and d_same["dreamsim"] < 1e-4

    # 4 random images → diversity should be positive
    diff = torch.rand(4, 3, 256, 256, device=device)
    d_diff = computer.compute_diversity(diff)
    print(f"  random images:    clip={d_diff['clip']:.4f}, dreamsim={d_diff['dreamsim']:.4f}")
    assert d_diff["clip"] > 0.01 and d_diff["dreamsim"] > 0.01
    print("  ✓ self-test passed")


if __name__ == "__main__":
    _selftest()
