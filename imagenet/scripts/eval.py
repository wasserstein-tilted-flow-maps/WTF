"""ImageNet reward and diversity evaluation.

Generates K samples per class for a fixed class set and scores within-class diversity
(CLIP and DreamSim pairwise distances), then generates over a sampled class set and
scores reward (HPSv2, PickScore, ImageReward). Classes are held fixed per group so
that diversity is a within-class quantity. Runs single-process.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from diffusers.models import AutoencoderKL
from models.dmft import DMFT_models
from samplers import euler_sampler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="DMFT-XL/2")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--dmf-depth", type=int, default=20)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--n-groups", type=int, default=32)   # classes, mirrors T2I's 32 prompts
    p.add_argument("--k-per-group", type=int, default=16)  # K per class, mirrors T2I's K=16
    p.add_argument("--num-steps", type=int, default=50)    # NFE
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reward", default="hps", choices=["hps", "none"])
    p.add_argument("--div-classes", default="first", choices=["first", "random"],
                   help="'first' = first N classes; 'random' = a seeded random class set.")
    p.add_argument("--all-rewards", action="store_true",
                   help="also score PickScore + ImageReward (the paper reports all three)")
    p.add_argument("--reward-classes", type=int, default=0,
                   help="PROTOCOL: if >0, ALSO run a reward pass over this many RANDOMLY sampled classes "
                        "(spec says 1000), one sample per class, scoring HPSv2 on the generated images.")
    # sampler settings (defaults match the DMF sampler)
    p.add_argument("--g-min", type=float, default=0.0)
    p.add_argument("--g-max", type=float, default=1.0)
    p.add_argument("--shift", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--coeff-type", default="default")
    p.add_argument("--path-type", default="linear")
    p.add_argument("--tcf", type=float, default=0.0,
                   help="Reward-strength conditioning fed to a token-conditioned model "
                        "(trained with --use-token-cond). Ignored by unconditioned models. "
                        "For a conditioned model the metrics are a FUNCTION of this value.")
    p.add_argument("--allow-unexpected", action="store_true",
                   help="Waive the architecture check below. Only if you are certain.")
    p.add_argument("--t-min", type=float, default=0.0)
    p.add_argument("--t-max", type=float, default=1.0)
    a = p.parse_args()

    # ImageReward's loader calls into torch.distributed (the T2I harness always runs under an 8-way
    # torchrun, so a group always exists there). This eval is single-process, so create a trivial
    # world_size=1 group or ImageReward fails with "Default process group has not been initialized".
    import torch.distributed as _dist
    if not _dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29400 + (os.getpid() % 500)))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        try:
            _dist.init_process_group(backend="gloo", rank=0, world_size=1)
        except Exception as _e:
            print("[eval] dist init failed (ImageReward may be skipped):", _e)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    dev = a.device
    latent_size = a.resolution // 8

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(dev).eval()
    # Build the eval model with the architecture flags stored in the checkpoint.
    _ck_probe = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    _ck_args = _ck_probe.get("args", None)
    _use_tok = bool(getattr(_ck_args, "use_token_cond", False)) if _ck_args is not None else False
    del _ck_probe
    if _use_tok:
        print("[eval] checkpoint is TOKEN-CONDITIONED -> building model with use_token_cond=True", flush=True)
    model = DMFT_models[a.model](
        input_size=latent_size, num_classes=a.num_classes,
        use_cfg_embedding=True, dmf_depth=a.dmf_depth,
        attn_func="torch_sdpa", qk_norm=False, use_token_cond=_use_tok,
    )
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    key = "ema" if "ema" in sd else ("model" if "model" in sd else None)
    msg = model.load_state_dict(sd[key] if key else sd, strict=False)
    print(f"[eval] loaded {a.ckpt} via key={key!r}: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}", flush=True)
    # Load strictly. The only expected unexpected keys are the training-only logvar head.
    if len(msg.unexpected_keys) > 2 and not a.allow_unexpected:
        raise SystemExit(
            f"[eval] ABORT: {len(msg.unexpected_keys)} unexpected keys. The checkpoint carries modules this "
            f"eval model does not have, so strict=False would DROP them and evaluate a crippled model. "
            f"First few: {msg.unexpected_keys[:6]}. Pass --allow-unexpected only if certain.")
    model = model.to(dev).eval()

    from metrics import DiversityComputer, _mean_pairwise_l2_squared
    div = DiversityComputer(device=dev)

    # Rewards are scored on the same sampled images as diversity, so every reported number refers to the same
    # 32 x 16 images. Prompts are "A high-resolution, high-quality photograph of a <class>", as in training.
    # All three rewards use reward_scores.py, which matches the text-to-image harness.
    scorers = {}
    if a.reward == "hps":
        from reward_scores import hpscore, pickscore as _pick, imagereward as _imrew
        scorers["hps"] = hpscore(dtype=torch.float32, device=dev)
        if a.all_rewards:
            scorers["pickscore"] = _pick(dtype=torch.float32, device=dev)
            try:
                import transformers.modeling_utils as _tmu
                from transformers import pytorch_utils as _tpu
                for _n in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
                    if not hasattr(_tmu, _n) and hasattr(_tpu, _n):
                        setattr(_tmu, _n, getattr(_tpu, _n))
            except Exception as e:
                print("[eval] could not patch transformers.modeling_utils:", e)
            try:
                scorers["imagereward"] = _imrew(dtype=torch.float32, device=dev)
            except Exception as e:
                print("[eval] WARN ImageReward unavailable:", e)
    reward_fn = scorers.get("hps")

    from timm.data import ImageNetInfo
    _imnet = ImageNetInfo()
    PROMPT_PREFIX = "A high-resolution, high-quality photograph of a "

    g = torch.Generator(device=dev).manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    # Diversity uses the first 32 classes, 16 images per class. A random class set gives a
    # different mean because per-class DreamSim spans ~0.32-0.68, so absolute values are only
    # comparable at a fixed class set.
    if a.div_classes == "first":
        classes = np.arange(a.n_groups)
    else:
        classes = rng.choice(a.num_classes, size=a.n_groups, replace=False)

    clip_vals, ds_vals, rew_vals = [], [], []
    per_class = {}
    for gi, cls in enumerate(classes):
        z = torch.randn(a.k_per_group, 4, latent_size, latent_size, device=dev, generator=g)
        y = torch.full((a.k_per_group,), int(cls), device=dev, dtype=torch.long)
        samples = euler_sampler(
            model=model, latents=z, y=y, num_steps=a.num_steps, cfg_scale=a.cfg_scale,
            g_min=a.g_min, g_max=a.g_max, shift=a.shift, heun=False, gamma=a.gamma,
            coeff_type=a.coeff_type, path_type=a.path_type, tcf=a.tcf,
            t_min=a.t_min, t_max=a.t_max,
        ).to(torch.float32)
        imgs = vae.decode(samples / 0.18215).sample          # [-1,1]
        imgs = ((imgs + 1.0) / 2.0).clamp(0, 1)              # [0,1], as metrics expects
        # Same diversity computation as the text-to-image harness (t2i/src/scripts/eval.py).
        clip_feat = div.embed_clip(imgs)
        ds_feat = div.embed_dreamsim(imgs)
        c = _mean_pairwise_l2_squared(clip_feat).item()
        d = _mean_pairwise_l2_squared(ds_feat).item()
        clip_vals.append(float(c)); ds_vals.append(float(d))
        if reward_fn is not None:
            prompts = [PROMPT_PREFIX + _imnet.index_to_description(int(cls))] * imgs.shape[0]
            # PER-CLASS for ALL THREE rewards, so a per-class reward can be paired with the per-class
            # diversity of the SAME 16 images. This is the only way to ask WHERE an aggregate reward gap
            # lives: uniform across classes, or concentrated in the high-diversity ones.
            for _n, _sc in scorers.items():
                _v, _ = _sc(imgs, prompts, {})
                per_class.setdefault(_n, []).append(float(_v.float().mean()))
            rew_vals.append(per_class["hps"][-1])
        print(f"[eval] group {gi+1}/{a.n_groups} class={int(cls)} clip={clip_vals[-1]:.4f} ds={ds_vals[-1]:.4f}", flush=True)

    # Reward pass: randomly sample `reward_classes` classes, generate, score HPSv2
    rew_pass = []
    rew_all = {k: [] for k in ("hps", "pickscore", "imagereward")}
    if a.reward_classes > 0 and reward_fn is not None:
        rr = np.random.RandomState(a.seed + 1)
        rcls = rr.choice(a.num_classes, size=a.reward_classes, replace=True)
        B = a.k_per_group
        for i in range(0, len(rcls), B):
            chunk = rcls[i:i + B]
            z = torch.randn(len(chunk), 4, latent_size, latent_size, device=dev, generator=g)
            y = torch.tensor([int(c) for c in chunk], device=dev, dtype=torch.long)
            sm = euler_sampler(
                model=model, latents=z, y=y, num_steps=a.num_steps, cfg_scale=a.cfg_scale,
                g_min=a.g_min, g_max=a.g_max, shift=a.shift, heun=False, gamma=a.gamma,
                coeff_type=a.coeff_type, path_type=a.path_type, tcf=a.tcf,
                t_min=a.t_min, t_max=a.t_max).to(torch.float32)
            im = vae.decode(sm / 0.18215).sample
            im = ((im + 1.0) / 2.0).clamp(0, 1)
            pr = [PROMPT_PREFIX + _imnet.index_to_description(int(c)) for c in chunk]
            for _n, _s in scorers.items():
                _sc, _ = _s(im, pr, {})
                _sc = [float(x) for x in _sc.float().cpu()]
                if _n == "hps":
                    rew_pass.extend(_sc)
                rew_all[_n].extend(_sc)
            if (i // B) % 10 == 0:
                print(f"[eval] reward pass {i+len(chunk)}/{len(rcls)}", flush=True)

    def summ(v):
        v = np.asarray(v, dtype=np.float64)
        return {"mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(len(v))),
                "n_prompts": int(len(v)), "K_per_prompt": int(a.k_per_group),
                "per_prompt": [float(x) for x in v]}

    out = {"args": vars(a),
           "diversity": {"clip_pairwise_l2sq": summ(clip_vals),
                         "dreamsim_pairwise_l2sq": summ(ds_vals)}}
    if rew_vals:
        out["quality"] = {"hps_on_div_classes": summ(rew_vals)}
        # Paper protocol: every reward is averaged over the same 32 x 16 images used for diversity.
        for _n in ("pickscore", "imagereward"):
            if per_class.get(_n):
                out["quality"][f"{_n}_on_div_classes"] = summ(per_class[_n])
    if per_class:
        out["per_class"] = {"classes": [int(c) for c in classes],
                            "dreamsim": [float(x) for x in ds_vals],
                            "clip": [float(x) for x in clip_vals],
                            **{k: [float(x) for x in v] for k, v in per_class.items()}}
    if rew_pass:
        import numpy as _np
        _v = _np.asarray(rew_pass, dtype=_np.float64)
        out.setdefault("quality", {})["hps"] = {
            "mean": float(_v.mean()), "sem": float(_v.std(ddof=1) / _np.sqrt(len(_v))), "n": int(len(_v))}
        for _n, _vals in rew_all.items():
            if _n != "hps" and _vals:
                _a = _np.asarray(_vals, dtype=_np.float64)
                out["quality"][_n] = {"mean": float(_a.mean()),
                                      "sem": float(_a.std(ddof=1) / _np.sqrt(len(_a))), "n": int(len(_a))}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[eval] WROTE {a.out}")
    q = out.get("quality", {})
    print(f"[eval] clip {out['diversity']['clip_pairwise_l2sq']['mean']:.5f} "
          f"dreamsim {out['diversity']['dreamsim_pairwise_l2sq']['mean']:.5f}"
          + "".join(f" {k} {v['mean']:.5f}" for k, v in q.items() if k.endswith("_on_div_classes"))
          + (f" hps {q['hps']['mean']:.5f} (n={q['hps']['n']})" if rew_pass else ""))


if __name__ == "__main__":
    main()
