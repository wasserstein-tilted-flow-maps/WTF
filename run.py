#!/usr/bin/env python3
"""Training entry point for both benchmarks.

    python run.py t2i/configs/wtf_t2i.yaml
    python run.py t2i/configs/wtf_t2i.yaml --set objective.distill_weight=1.0 --dry-run
    python run.py imagenet/configs/wtf_imagenet.yaml --set optim.seed=1 --set optim.exp_name=wtf_imagenet_s1

The benchmark ("lane") is the top-level folder the config lives in (t2i/ or imagenet/); a config stored
elsewhere can name it with a top-level `lane: t2i` or `lane: imagenet` key.
"""
import argparse, os, shlex, subprocess, sys, yaml

GROUPS = ("substrate", "adaptation", "objective", "estimator", "guidance", "optim", "hjb", "runtime")
ROOT = os.path.dirname(os.path.abspath(__file__))

# Config key -> CLI flag. Keys absent from these tables are ignored, and every launch prints them.
T2I = {
    "substrate.ckpt": "--ckpt", "substrate.config": "--config",
    "substrate.height": "--height", "substrate.width": "--width",
    "substrate.attn_type": "--attn-type", "substrate.reward": "--reward",
    "adaptation.lora_rank": "--lora-rank", "adaptation.lora_alpha": "--lora-alpha",
    "objective.distill_weight": "--distill-weight", "objective.distill_mode": "--distill-mode",
    "objective.distill_method": "--distill-method",
    "objective.target_control_fraction": "--target-control-fraction",
    "objective.eta_mode": "--eta-mode",
    "estimator.rollout_steps": "--rollout-steps",
    "estimator.num_tau_samples": "--num-tau-samples",
    "estimator.xt_chain_steps": "--xt-chain-steps",
    "estimator.quantile_clip_pct": "--quantile-clip-pct",
    "estimator.frozen_model_mode": "--frozen-model-mode",
    "estimator.cfg_scale": "--cfg-scale",
    "estimator.w_rew_warmup_iters": "--w-rew-warmup-iters",
    "hjb.consist_weight": "--hjb-consist-weight",
    "hjb.boundary_weight": "--hjb-boundary-weight",
    "hjb.eps_time": "--hjb-eps-time",
    "optim.lr": "--lr", "optim.adam_beta2": "--adam-beta2",
    "optim.weight_decay": "--weight-decay",
    "optim.lr_warmup_steps": "--lr-warmup-steps", "optim.batch_size": "--batch-size",
    "optim.grad_accum": "--grad-accum-steps", "optim.loss_ema_decay": "--loss-ema-decay",
    "optim.out_dir": "--out-dir",
    "optim.num_iters": "--num-iters", "optim.checkpoint_steps": "--checkpoint-steps",
    "optim.sampling_steps": "--sampling-steps", "optim.seed": "--seed",
    "estimator.reward_jitter_pairs": "--reward-jitter-pairs",
    "estimator.reward_jitter_sigma": "--reward-jitter-sigma",
}
IMAGENET = {
    "substrate.model": "--model", "substrate.dmf_depth": "--dmf-depth",
    "substrate.pretrained": "--pretrained",
    "substrate.resolution": "--resolution",
    "objective.target_control_fraction": "--target-control-fraction",
    "objective.diag_prob": "--diag-prob", "objective.distill_method": "--distill-method",
    "objective.anchor_weight": "--anchor-weight",
    "estimator.num_tau_samples": "--num-tau-samples",
    "estimator.rollout_steps": "--rollout-steps",
    "guidance.g_type": "--g-type", "guidance.omega": "--omega",
    "guidance.g_min": "--g-min", "guidance.g_max": "--g-max",
    "guidance.cfg_prob": "--cfg-prob",
    "optim.lr": "--learning-rate", "optim.ema_decay": "--ema-decay",
    "optim.batch_size": "--batch-size", "optim.max_train_steps": "--max-train-steps",
    "optim.checkpointing_steps": "--checkpointing-steps",
    "optim.max_grad_norm": "--max-grad-norm", "optim.mixed_precision": "--mixed-precision",
    "optim.adam_beta1": "--adam-beta1", "optim.adam_beta2": "--adam-beta2",
    "optim.sampling_steps": "--sampling-steps",
    "optim.exp_name": "--exp-name", "optim.project_name": "--project-name",
    "runtime.attn_func": "--attn-func", "runtime.num_workers": "--num-workers",
    "optim.seed": "--seed",
    "substrate.data_dir": "--data-dir",
    "substrate.num_shards": "--num-shards",
}
# Consumed outside the flag tables, so not "ignored": optim.gpus sets --nproc_per_node /
# --num_processes, optim.master_port sets the rendezvous port, estimator.use_token_cond has its
# own emit-when-false rule below, and estimator.detach is unified across the two lanes.
HANDLED = {"optim.gpus", "optim.master_port", "estimator.use_token_cond", "estimator.detach"}

# Emitted as bare flags when true.
BOOL = {
    "adaptation.dual_adapter": "--dual-adapter",
    "estimator.cfg_detach_uncond": "--cfg-detach-uncond",
    "estimator.cost_at_tau": "--cost-at-tau",
    "estimator.use_logvar": "--use-logvar",
    "estimator.no_w_rew_norm": "--no-w-rew-norm",
}

def load_config(path):
    """Load a config, resolving a `defaults:` parent relative to the config's directory."""
    cfg = yaml.safe_load(open(path)) or {}
    parent = cfg.pop("defaults", None)
    if parent is None:
        return cfg
    base = load_config(os.path.join(os.path.dirname(os.path.abspath(path)), parent))
    return merge(base, cfg)

def merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out

def detect_lane(config_path, cfg):
    """Return "t2i" or "imagenet" for a config path (see the module docstring)."""
    if cfg.get("lane") in ("t2i", "imagenet"):
        return cfg["lane"]
    path = os.path.abspath(config_path)
    for lane in ("t2i", "imagenet"):
        lane_dir = os.path.join(ROOT, lane)
        if os.path.commonpath([path, lane_dir]) == lane_dir:
            return lane
    sys.exit(f"cannot tell the benchmark of {config_path}: keep it under t2i/ or imagenet/, "
             "or add a top-level `lane: t2i` or `lane: imagenet` key")

def missing_substrate(cfg, lane):
    """List the substrate paths that are unset or still contain an unexpanded ${VAR}."""
    keys = ("substrate.ckpt", "substrate.config") if lane == "t2i" else ("substrate.pretrained",)
    out = []
    for key in keys:
        v = dig(cfg, key)
        if v is None or "$" in os.path.expandvars(str(v)):
            out.append(f"{key} ({v})" if v is not None else key)
    return out

def dig(d, dotted):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur: return None
        cur = cur[k]
    return cur

def put(d, dotted, val):
    cur = d
    parts = dotted.split(".")
    for k in parts[:-1]: cur = cur.setdefault(k, {})
    for cast in (int, float):
        try: val = cast(val); break
        except ValueError: pass
    if val in ("true","True"): val = True
    if val in ("false","False"): val = False
    cur[parts[-1]] = val

def build(cfg, lane):
    argv, unmapped = [], []
    table = T2I if lane == "t2i" else IMAGENET
    for key, flag in table.items():
        v = dig(cfg, key)
        if v is not None: argv += [flag, os.path.expandvars(str(v))]
    for key, flag in BOOL.items():
        if dig(cfg, key) is True: argv.append(flag)
    # BooleanOptionalAction on the ImageNet side: emit the form that was asked for.
    if lane == "imagenet":
        v = dig(cfg, "estimator.use_token_cond")
        if v is True:  argv.append("--use-token-cond")
        if v is False: argv.append("--no-use-token-cond")
    # ── The detach lever: one spelling for both lanes ──────────────────────────────
    # Unified here because the underlying CLIs disagree: T2I exposes a BOOLEAN
    # (--detach-lookahead-velocity) because it runs rollout_steps=1, where the lookahead
    # is a single hop; ImageNet exposes an ENUM (--detach-lookahead {none,last,all}) because
    # rollout_steps=2 has two hops and last cuts only the second. See docs/CONFIGURATION.md.
    d = (dig(cfg, "estimator.detach") or "none")
    if lane == "t2i":
        if d == "last":
            sys.exit("estimator.detach=last is imagenet-only: the t2i CLI exposes detach as a "
                     "boolean, so it can only express none or all.")
        if d == "all": argv.append("--detach-lookahead-velocity")
    else:
        argv += ["--detach-lookahead", d]
    for k in GROUPS:
        for sub in (cfg.get(k) or {}):
            dotted = f"{k}.{sub}"
            if dotted not in table and dotted not in BOOL and dotted not in HANDLED:
                unmapped.append(dotted)
    unmapped += [k for k in cfg if k not in GROUPS and k != "lane"]
    return argv, unmapped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="override a config key, e.g. objective.distill_weight=3.0")
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    for ov in a.set:
        k, _, v = ov.partition("="); put(cfg, k, v)
    lane = detect_lane(a.config, cfg)
    argv, unmapped = build(cfg, lane)
    gpus = a.gpus or dig(cfg, "optim.gpus") or 8
    entry = os.path.join(ROOT, lane, "src", "train.py")
    # imagenet requires accelerate; t2i requires torchrun. They are not interchangeable.
    port = str(dig(cfg, "optim.master_port") or (29501 if lane == "imagenet" else 29500))
    if lane == "imagenet":
        cmd = ["accelerate", "launch", "--num_machines", "1", "--multi_gpu",
               "--num_processes", str(gpus), "--main_process_port", port, entry] + argv
    else:
        cmd = ["torchrun", "--nnodes", "1", "--nproc_per_node", str(gpus),
               "--master_port", port, entry] + argv
    print("lane:", lane)
    if unmapped:
        print("ignored keys:", ", ".join(sorted(unmapped)))
    print(" ".join(shlex.quote(c) for c in cmd))
    missing = missing_substrate(cfg, lane)
    if missing:
        hint = ("export TIM_CKPT and TIM_CONFIG" if lane == "t2i" else
                "export DMF_CKPT=/path/to/dmf_xl_2_256.pt; without it the zero-initialized output "
                "layer gives no training signal")
        msg = f"missing substrate path: {', '.join(missing)}. See docs/SETUP.md ({hint})."
        if not a.dry_run:
            sys.exit(msg)
        print("warning:", msg)
    if a.dry_run: return
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__": main()
