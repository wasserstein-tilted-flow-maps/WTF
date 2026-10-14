"""Standard T2I evaluation: quality (HPS/PickScore/ImageReward) + diversity (CLIP/DreamSim).

Eval protocol:
- Quality: 1000 generations sampled with replacement from 100 prompts → mean reward.
- Diversity: first 32 prompts × K=16 samples per prompt → mean CLIP+DreamSim pairwise diversity.

Supports two LoRA checkpoint formats:
- training checkpoint: .pt with {"lora": state_dict, ...}  (t2i/src/train.py output)
- peft:    pytorch_lora_weights.safetensors    (e.g. VGG-Flow checkpoints)

Runs on one or more GPUs under torchrun (eval.py at the repository root launches it). Each sample's
noise depends on its GPU rank, so use the same GPU count to reproduce a result exactly.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Make `lib.*` imports work when run via torchrun from any cwd.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.distributed as dist

# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", type=str, required=True,
                   help="Path to LoRA ckpt: .pt (ftfm) or pytorch_lora_weights.safetensors (vggflow), "
                        "or a directory containing the latter.")
    p.add_argument("--prompts-file", type=str,
                   default="lib/assets/HPDv2/sample100_photo50_paint50.txt")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--config", type=str, required=True,
                   help="TiM config yaml, e.g. /path/to/TiM/configs/t2i/tim_xl_p1_t2i.yaml")
    p.add_argument("--ckpt-base", type=str, required=True,
                   help="TiM base ckpt, e.g. /path/to/TiM/checkpoints/t2i_model.bin")
    p.add_argument("--lora-rank", type=int, default=16,
                   help="LoRA rank used at training time (16 for the reported WTF run).")
    p.add_argument("--num-quality", type=int, default=1000)
    p.add_argument("--num-div-prompts", type=int, default=32)
    p.add_argument("--num-div-samples", type=int, default=16)
    p.add_argument("--nfe", type=int, default=50)
    p.add_argument("--nfes", type=str, default=None,
                   help="Comma-separated list of NFEs to sweep in one process "
                        "(loads TiM + LoRA + rewards once, iterates over NFEs). "
                        "Overrides --nfe when set.")
    p.add_argument("--cfg-scale", type=float, default=2.5)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-lora", action="store_true",
                   help="Skip loading LoRA — evaluate the base model only.")
    p.add_argument("--rewards", type=str, default="hps,pickscore,imagereward",
                   help="Comma-separated subset of {hps, pickscore, imagereward}.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def init_dist():
    """torchrun-aware init. Returns (rank, world_size, local_rank).

    If WORLD_SIZE env > 1 (torchrun), uses NCCL across ranks.
    Otherwise falls back to a single-rank group so dist.barrier() in reward fns works.
    """
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29575")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", world_size=1, rank=0)
    return 0, 1, 0


# --------------------------------------------------------------------------- #
def load_lora(model, ckpt_path: str, lora_rank: int):
    """Apply LoRA + load weights. Auto-detects ftfm .pt vs vggflow safetensors."""
    from peft.utils import set_peft_model_state_dict
    from lib.tim_adapter import apply_lora_to_tim

    # Resolve path: if directory, look for safetensors inside.
    p = Path(ckpt_path)
    if p.is_dir():
        st = p / "pytorch_lora_weights.safetensors"
        pt = list(p.glob("*.pt"))
        if st.exists():
            ckpt_file = str(st)
        elif pt:
            ckpt_file = str(pt[0])
        else:
            raise FileNotFoundError(f"No LoRA weights found in {p}")
    else:
        ckpt_file = str(p)

    # Canonical wrapper (uses lib/tim_lora.py:DEFAULT_LORA_TARGETS, alpha==rank).
    peft_model = apply_lora_to_tim(model, rank=lora_rank, alpha=lora_rank,
                                    adapter_name="default")

    # Load weights
    if ckpt_file.endswith(".safetensors"):
        from safetensors.torch import load_file
        raw = load_file(ckpt_file, device="cpu")
        # Strip "transformer." prefix if vggflow saver added it
        lora_sd = {k[len("transformer."):] if k.startswith("transformer.") else k: v
                   for k, v in raw.items()}
        result = set_peft_model_state_dict(peft_model, lora_sd, adapter_name="default")
        print(f"[lora] vggflow safetensors loaded: missing={len(getattr(result, 'missing_keys', []))}, "
              f"unexpected={len(getattr(result, 'unexpected_keys', []))}")
    elif ckpt_file.endswith(".pt"):
        raw = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        if "lora" not in raw:
            raise KeyError(f"{ckpt_file}: expected key 'lora' in checkpoint dict, got {list(raw.keys())[:5]}")
        # ftfm saves keys WITH the `base_model.model.` peft prefix (raw state_dict()
        # filtered by `.default.` + `lora_`). Use the raw load path so prefixed keys
        # match peft_model.state_dict() directly. Filter to LoRA-only to keep the
        # base TiM weights untouched.
        # DUAL-ADAPTER: ckpts may contain BOTH ".default." (dynamics adapter A) and
        # ".map." (flow-map adapter B) LoRA tensors. The eval model only instantiates
        # the "default" adapter, so ".map." keys would be UNEXPECTED and raise below.
        # We evaluate adapter A (the deployed dynamics) => keep ".default." only.
        # Backward-compatible: single-adapter ckpts contain only ".default." already.
        lora_sd = {k: v for k, v in raw["lora"].items()
                   if "lora_" in k and ".default." in k}
        _n_map = sum(1 for k in raw["lora"] if ".map." in k)
        if _n_map:
            print("[lora] dual-adapter ckpt: ignoring %d .map. tensors (evaluating adapter A/default)" % _n_map)
        result = peft_model.load_state_dict(lora_sd, strict=False)
        # `missing_keys` will include all base-TiM keys + the unused "ema"-adapter LoRA
        # keys (we only loaded "default"). What matters is no `default` LoRA keys are
        # missing and zero unexpected.
        unexpected = list(result.unexpected_keys)
        loaded_default = sum(1 for k in lora_sd if "default" in k)
        print(f"[lora] ftfm .pt loaded from step {raw.get('steps', raw.get('step', '?'))}: "
              f"loaded_default_lora_tensors={loaded_default}, unexpected={len(unexpected)}"
              f"{' '+repr(unexpected[:3]) if unexpected else ''}")
        if unexpected:
            raise RuntimeError(f"unexpected LoRA keys when loading ftfm ckpt: {unexpected[:5]}")
    else:
        raise ValueError(f"Unknown ckpt format: {ckpt_file}")

    return peft_model


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "eval.log")

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    rank, world, local_rank = init_dist()
    is_main = (rank == 0)

    if is_main:
        log(f"args: {vars(args)}")
        log(f"distributed: rank={rank}/{world} local_rank={local_rank}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(f"cuda:{local_rank}")

    # ── Build TiM pipeline ──────────────────────────────────────────────────
    from lib.tim_adapter import build_tim_pipeline, encode_captions, decode_to_image
    from lib.ftfm.sampling import sample_with_cfg
    from lib.metrics import DiversityComputer, _mean_pairwise_l2_squared
    from lib.fa_jvp.tim_patch import apply_to_tim

    log("building TiM pipeline...")
    pipeline = build_tim_pipeline(config_path=args.config, ckpt_path=args.ckpt_base, device=device)
    bundle = pipeline.bundle
    apply_to_tim(model=bundle.model, dtype=bundle.weight_dtype)
    log(f"bundle ready (latent {bundle.spatial_downsample}x ds)")

    # ── Apply LoRA (or skip for base eval) ──────────────────────────────────
    model = bundle.model
    if not args.no_lora:
        log(f"loading LoRA from {args.ckpt_path}")
        model = load_lora(bundle.model, args.ckpt_path, lora_rank=args.lora_rank)
        bundle.model = model
        model.eval()
    else:
        log("--no-lora: evaluating base model")
    model.eval()

    # ── Read prompts ────────────────────────────────────────────────────────
    repo_root = Path(__file__).resolve().parent.parent
    pf = args.prompts_file
    if not os.path.isabs(pf):
        pf = str(repo_root / pf)
    with open(pf) as f:
        prompts_all = [ln.strip() for ln in f if ln.strip()]
    if len(prompts_all) < 100:
        log(f"WARN: prompts file has {len(prompts_all)} lines, expected 100")
    log(f"loaded {len(prompts_all)} prompts from {pf}")

    # ── Resolve NFEs to sweep ───────────────────────────────────────────────
    if args.nfes:
        nfe_list = [int(x) for x in args.nfes.split(",") if x.strip()]
    else:
        nfe_list = [args.nfe]
    if is_main:
        log(f"NFEs to sweep: {nfe_list}")

    # ── Load reward fns ONCE (re-used across all NFEs) ──────────────────────
    rewards_to_compute = [r.strip() for r in args.rewards.split(",") if r.strip()]
    reward_fns = {}
    if "hps" in rewards_to_compute:
        from lib.reward_func.rewards import hpscore
        if is_main: log("loading HPSv2...")
        reward_fns["hps"] = hpscore(dtype=torch.float32, device=device)
    if "pickscore" in rewards_to_compute:
        from lib.reward_func.rewards import pickscore
        if is_main: log("loading PickScore...")
        reward_fns["pickscore"] = pickscore(dtype=torch.float32, device=device)
    if "imagereward" in rewards_to_compute:
        # ImageReward PyPI package imports `apply_chunking_to_forward` from
        # transformers.modeling_utils, but that helper moved to transformers.pytorch_utils
        # in newer transformers releases. Patch it back so ImageReward's BLIP model loads.
        try:
            import transformers.modeling_utils as _tmu
            from transformers import pytorch_utils as _tpu
            for _name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices",
                           "prune_linear_layer"):
                if not hasattr(_tmu, _name) and hasattr(_tpu, _name):
                    setattr(_tmu, _name, getattr(_tpu, _name))
        except Exception as e:
            if is_main: log(f"  (could not patch transformers.modeling_utils: {e})")
        try:
            from lib.reward_func.rewards import imagereward
            if is_main: log("loading ImageReward...")
            reward_fns["imagereward"] = imagereward(dtype=torch.float32, device=device)
        except Exception as e:
            if is_main: log(f"WARN: ImageReward unavailable, skipping: {e}")

    # ── Pre-compute quality prompts (shared across NFEs) ────────────────────
    rng = random.Random(args.seed)
    quality_indices_full = [rng.randrange(len(prompts_all)) for _ in range(args.num_quality)]
    my_quality_idx = list(range(rank, args.num_quality, world))
    quality_prompts = [prompts_all[quality_indices_full[i]] for i in my_quality_idx]
    bs = args.batch_size
    n_local = len(quality_prompts)

    # ── Pre-load DiversityComputer (shared across NFEs) ─────────────────────
    if is_main: log("loading DiversityComputer (CLIP + DreamSim)...")
    div_computer = DiversityComputer(device=device)

    div_prompts_all = prompts_all[:args.num_div_prompts]
    my_div_idx = list(range(rank, len(div_prompts_all), world))
    my_div_prompts = [(i, div_prompts_all[i]) for i in my_div_idx]
    K = args.num_div_samples

    # ── Loop over NFEs ──────────────────────────────────────────────────────
    for current_nfe in nfe_list:
        if is_main:
            log(f"=== NFE={current_nfe} ===")
        quality_scores_local = {k: [] for k in reward_fns}
        t0 = time.time()
        for batch_start in range(0, n_local, bs):
            batch_prompts = quality_prompts[batch_start:batch_start + bs]
            if not batch_prompts:
                break
            cap_feats, _ = encode_captions(bundle, batch_prompts, truncate_to_max_length_in_batch=False)
            seed = args.seed * 100003 + rank * 7919 + batch_start
            with torch.no_grad():
                lat = sample_with_cfg(
                    model, bundle, cap_feats,
                    num_steps=current_nfe, cfg_scale=args.cfg_scale,
                    height=args.height, width=args.width, seed=seed, device=device,
                )
                imgs = decode_to_image(bundle, lat.to(bundle.weight_dtype)).clamp(0, 1)
            for name, fn in reward_fns.items():
                with torch.no_grad():
                    scores, _ = fn(imgs, batch_prompts, {})
                quality_scores_local[name].extend(scores.detach().cpu().float().tolist())
            if is_main and (batch_start // bs) % 5 == 0:
                elapsed = time.time() - t0
                done = batch_start + len(batch_prompts)
                log(f"  [NFE={current_nfe}] quality rank0 {done}/{n_local} ({elapsed:.0f}s elapsed)")
        if is_main:
            log(f"  [NFE={current_nfe}] quality gen complete on rank0 ({time.time()-t0:.0f}s); gathering...")

        # All-gather scores across ranks (rank 0 aggregates)
        quality_summary = {}
        for name in reward_fns:
            if world > 1:
                gathered = [None] * world
                dist.all_gather_object(gathered, quality_scores_local[name])
            else:
                gathered = [quality_scores_local[name]]
            if is_main:
                all_scores = [s for sub in gathered for s in sub]
                arr = np.array(all_scores)
                quality_summary[name] = {
                    "mean": float(arr.mean()),
                    "sem": float(arr.std() / np.sqrt(len(arr))),
                    "n": int(len(arr)),
                }
        if is_main:
            log(f"  [NFE={current_nfe}] quality means: {quality_summary}")

        # ── Diversity eval ──────────────────────────────────────────────────
        local_clip = {}
        local_dreamsim = {}
        t1 = time.time()
        for n, (pi, prompt) in enumerate(my_div_prompts):
            cap_feats, _ = encode_captions(bundle, [prompt] * K, truncate_to_max_length_in_batch=False)
            all_imgs = []
            for s in range(0, K, bs):
                sub = cap_feats[s:s + bs]
                seed = args.seed * 17 + pi * 1009 + s
                with torch.no_grad():
                    lat = sample_with_cfg(
                        model, bundle, sub,
                        num_steps=current_nfe, cfg_scale=args.cfg_scale,
                        height=args.height, width=args.width, seed=seed, device=device,
                    )
                    imgs = decode_to_image(bundle, lat.to(bundle.weight_dtype)).clamp(0, 1)
                all_imgs.append(imgs)
            all_imgs = torch.cat(all_imgs, dim=0)
            with torch.no_grad():
                clip_feat = div_computer.embed_clip(all_imgs)
                ds_feat = div_computer.embed_dreamsim(all_imgs)
                clip_div = _mean_pairwise_l2_squared(clip_feat).item()
                ds_div = _mean_pairwise_l2_squared(ds_feat).item()
            local_clip[pi] = clip_div
            local_dreamsim[pi] = ds_div

        if world > 1:
            gathered_clip = [None] * world
            gathered_dreamsim = [None] * world
            dist.all_gather_object(gathered_clip, local_clip)
            dist.all_gather_object(gathered_dreamsim, local_dreamsim)
        else:
            gathered_clip = [local_clip]
            gathered_dreamsim = [local_dreamsim]

        diversity_summary = None
        if is_main:
            all_clip = {}
            all_ds = {}
            for d in gathered_clip:
                all_clip.update(d)
            for d in gathered_dreamsim:
                all_ds.update(d)
            ordered_clip = [all_clip[i] for i in range(len(div_prompts_all))]
            ordered_ds = [all_ds[i] for i in range(len(div_prompts_all))]
            diversity_summary = {
                "clip_pairwise_l2sq": {
                    "mean": float(np.mean(ordered_clip)),
                    "sem": float(np.std(ordered_clip) / np.sqrt(len(ordered_clip))),
                    "n_prompts": int(len(ordered_clip)),
                    "K_per_prompt": int(K),
                    "per_prompt": [float(x) for x in ordered_clip],
                },
                "dreamsim_pairwise_l2sq": {
                    "mean": float(np.mean(ordered_ds)),
                    "sem": float(np.std(ordered_ds) / np.sqrt(len(ordered_ds))),
                    "n_prompts": int(len(ordered_ds)),
                    "K_per_prompt": int(K),
                    "per_prompt": [float(x) for x in ordered_ds],
                },
            }
            log(f"  [NFE={current_nfe}] CLIP avg: {diversity_summary['clip_pairwise_l2sq']['mean']:.4f}")
            log(f"  [NFE={current_nfe}] DreamSim avg: {diversity_summary['dreamsim_pairwise_l2sq']['mean']:.4f}")

        if is_main:
            results = {
                "args": vars(args),
                "quality": quality_summary,
                "diversity": diversity_summary,
                "world_size": world,
                "wallclock_seconds": time.time() - t0,
                "nfe": current_nfe,
            }
            out_json = os.path.join(args.output_dir, f"eval_results_nfe{current_nfe}.json")
            with open(out_json, "w") as f:
                json.dump(results, f, indent=2)
            log(f"  [NFE={current_nfe}] saved {out_json}")
        if world > 1:
            dist.barrier()

    if is_main:
        log("=== ALL NFEs DONE ===")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
