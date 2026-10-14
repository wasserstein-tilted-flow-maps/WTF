"""WTF reward fine-tuning of TiM-T2I (text-to-image) with LoRA.

Runs the WTF objective (`FTFMVALUEDMFLoss` in lib/ftfm/loss.py) on the TiM-T2I flow map. Training prompts
come from `lib.reward_func.prompts` (the VGG-Flow prompt sets): for `--reward hps`, the HPDv2 photo and
paintings prompts.

Launch through run.py, which maps a YAML config onto this CLI:
    python run.py t2i/configs/wtf_t2i.yaml
or directly:
    torchrun --nproc_per_node 8 t2i/src/train.py --config $TIM_CONFIG --ckpt $TIM_CKPT \
      --reward hps --out-dir exps/wtf_t2i ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time

import numpy as np

# On SIGTERM, finish the current iteration, save a checkpoint and exit, so a resumed run continues
# from the same step.
_SHUTDOWN_REQUESTED = False
def _sigterm_handler(signum, frame):
    global _SHUTDOWN_REQUESTED
    print(f"[signal] received {signal.Signals(signum).name} — will save ckpt and exit cleanly", flush=True)
    _SHUTDOWN_REQUESTED = True
signal.signal(signal.SIGTERM, _sigterm_handler)
signal.signal(signal.SIGUSR1, _sigterm_handler)  # optional pre-emption warning signal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig

# TiM substrate helpers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.tim_adapter import (
    build_tim_pipeline, encode_captions, decode_to_image,
    decode_to_image_with_grad, apply_lora_to_tim, set_adapter_enabled,
    DEFAULT_LORA_TARGETS, num_trainable_params,
)

# WTF objective and utilities.
from lib.ftfm.loss import FTFMVALUEDMFLoss, FTFMOCImFLoss
from lib.ftfm.peft_ema import PeftEMA
from lib.ftfm.wrappers import TiMLossAdapter, RewardScoreShim
from lib.ftfm.sampling import sample_with_cfg, latents_to_uint8_grid, parse_nfes


def setup_dist():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def is_main(rank: int) -> bool:
    return rank == 0


def make_reward_mix(spec: str, device: torch.device, standardize: bool = True):
    """Weighted mixture of reward models.

    spec: "hps:0.5,imagereward:0.5" -> 0.5 z(hps) + 0.5 z(imagereward), where each reward is
    standardized by a running mean and std.
    """
    parts = []
    for tok in spec.split(","):
        name, _, w = tok.partition(":")
        parts.append((name.strip(), float(w) if w else 1.0))
    fns = [(make_reward(n, device), w, n) for n, w in parts]
    state = {n: {"m": None, "v": None} for _, _, n in fns}
    beta = 0.99

    def fn(images, prompts):
        total = None
        for f, w, n in fns:
            r = f(images, prompts).float()
            if r.dim() > 1:
                r = r.reshape(r.shape[0], -1)
                assert r.shape[1] == 1, (
                    f"reward {n} returned {tuple(r.shape)}, expected one score per image")
                r = r[:, 0]
            st = state[n]
            bm = r.mean().detach()
            bv = r.var(unbiased=False).detach().clamp_min(1e-8)
            st["m"] = bm if st["m"] is None else beta * st["m"] + (1 - beta) * bm
            st["v"] = bv if st["v"] is None else beta * st["v"] + (1 - beta) * bv
            z = (r - st["m"]) / st["v"].sqrt() if standardize else r
            total = w * z if total is None else total + w * z
        return total

    return fn


def make_reward(name: str, device: torch.device):
    """Reward callable `fn(images, prompts) -> (B,)` for hps, aesthetic, pickscore or imagereward."""
    name = name.lower()
    if name == "hps":
        # HPSv2 v2.1 weights (xswu/HPSv2) through the VGG-Flow hpscore loader.
        from lib.reward_func.rewards import hpscore as _hpscore
        _fn_full = _hpscore(device=device)  # (images, prompts, metadata) -> (scores, {})
        def fn(images, prompts):
            rew, _ = _fn_full(images, prompts, [{} for _ in prompts])
            return rew.float()
    elif name == "aesthetic":
        # LAION aesthetic head (sac+logos+ava1-l14-linearMSE.pth) on CLIP-L, from lib/assets/.
        from lib.reward_func.aesthetic_scorer import AestheticScorer
        import torch as _t
        m = AestheticScorer(dtype=_t.float32, distributed=False).to(device).eval()
        def fn(images, prompts):
            return m(images).float()
    elif name == "pickscore":
        from imscore.pickscore.model import PickScorer
        m = PickScorer().to(device).eval()
        def fn(images, prompts):
            return m.score(images, prompts).float()
    elif name == "imagereward":
        from imscore.imreward.model import ImageReward
        m = ImageReward().to(device).eval()
        def fn(images, prompts):
            return m.score(images, prompts).float()
    else:
        raise NotImplementedError(f"unknown reward '{name}'; expected hps|aesthetic|pickscore|imagereward")
    return fn


def _trainable_lora_state(peft_model: nn.Module) -> dict:
    """Return the trainable LoRA ('default' adapter) state."""
    sd = peft_model.state_dict()
    keys = {n for n, p in peft_model.named_parameters()
            if "lora_" in n and (".default." in n or ".map." in n)}
    return {k: v.detach().cpu() for k, v in sd.items() if k in keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="TiM YAML config path")
    p.add_argument("--ckpt", required=True, help="TiM checkpoint path")
    p.add_argument("--reward", default="hps",
                   choices=["hps", "aesthetic", "pickscore", "imagereward"])
    p.add_argument("--reward-mix-raw", action="store_true",
                   help="Mix raw reward values instead of standardising each component. "
                        "Use when the weights are already calibrated to the raw scales.")
    p.add_argument("--reward-mix", default=None,
                   help="Weighted reward mixture, e.g. hps:0.5,imagereward:0.5. "
                        "Overrides --reward. Components are z-scored before mixing.")
    p.add_argument("--prompt-file", default=None,
                   help="Optional override. If omitted, prompts are loaded from "
                        "lib.reward_func.prompts (byte-identical with upstream vggflow): "
                        "aesthetic→simple_animals, hps→HPDv2 photo+paintings, "
                        "pickscore→pick_a_pic, imagereward→DrawBench.")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=8)
    # Optimizer defaults (the configs set the reported values).
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--adam-beta2", type=float, default=0.95)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--w-rew-warmup-iters", type=int, default=0,
                   help="If >0, freeze the loss-side EMAs (EMA[‖v‖], EMA[‖∇R‖]) "
                        "after this many training iters. Calibrates w_rew during the "
                        "warmup window and locks it — kills the autoamplifier death "
                        "spiral while preserving cross-reward scale calibration. "
                        "Recommended 50.")
    p.add_argument("--loss-ema-decay", type=float, default=0.98,
                   help="EMA decay for loss-side trackers (EMA[‖v_t‖], EMA[‖∇R‖]) "
                        "used to compute w_rew = TCF·EMA[‖v‖]/EMA[‖∇R‖]. Default 0.98 "
                        "(half-life ~35 iters) tracks runs fast but autoamplifies when "
                        "‖∇R‖ persistently drops (death spiral). Set to 0.999 (half-life "
                        "~700 iters) to slow the EMA enough that 1000-iter runs stay near "
                        "the iter-0 calibration — prevents the autoamplifier without "
                        "sacrificing scale invariance across rewards.")
    p.add_argument("--target-control-fraction", type=float, default=25.0)
    p.add_argument("--div-grep", type=int, default=1,
                   help="D2 control-variate: same-prompt group size (>=2 enables). Combined path only.")
    p.add_argument("--div-baseline", type=float, default=0.0,
                   help="D2 control-variate strength: grad_reward -= div_baseline * per-prompt group-mean.")
    p.add_argument("--reward-jitter-pairs", type=int, default=0,
                   help="Spatial smoothing: # antithetic pairs for jitter-averaging grad_reward (0=off; M=2*pairs).")
    p.add_argument("--reward-jitter-sigma", type=float, default=0.05,
                   help="Spatial smoothing perturbation sigma (multiplied by x_t std).")
    p.add_argument("--eta-mode", type=str, default="none",
                   choices=["none", "quad", "linear", "constant"],
                   help="Time-attenuation on reward gradient. 'none' = paper-faithful "
                        "(WTF Eq 41/48); 'quad' = (1-t)²; 'linear' = (1-t).")
    p.add_argument("--quantile-clip-pct", type=float, default=0.0,
                   help="Clip per-sample ‖∇R‖ at this batch-quantile. 0.0 disables.")
    p.add_argument("--beta-outer", type=float, default=1.0,
                   help="Outer scalar multiplying the whole g_hat target (default 1.0 = "
                        "paper-faithful).")
    p.add_argument("--hjb-consist-weight", type=float, default=0.0,
                   help="WTF-LEGAL port of vggflow HJB consistency onto our own u field (0=off).")
    p.add_argument("--hjb-eps-time", type=float, default=1e-4)
    p.add_argument("--hjb-boundary-weight", type=float, default=0.0)
    p.add_argument("--hjb-nu-adapter", action="store_true",
                   help="ABLATION-2 (DIAGNOSTIC ONLY, breaks critic-free): nu from adapter B.")
    p.add_argument("--detach-lookahead-velocity", action="store_true",
                   help="Detach the lookahead velocity (vggflow algorithm.py:63). Their detach DEFINES\n                        their estimator: reward-grad term = I*grad_r. Keeping our analytic pullback is\n                        MORE CORRECT and therefore LESS FAITHFUL — a bridge must reproduce the approximation.")
    p.add_argument("--diagonal-lookahead", action="store_true",
                   help="Reward lookahead uses the DIAGONAL velocity v(x,t,t) on adapter A instead of "
                        "an off-diagonal map jump v(x,t,0) on adapter B. vggflow-faithful "
                        "(algorithm.py:62-63). REQUIRED with --distill-weight 0 + --dual-adapter, where "
                        "B gets zero gradient and its jump would be a stale base-map endpoint. "
                        "Default OFF = bit-identical to previous behaviour.")
    p.add_argument("--xt-rollout-mode", type=str, default="map", choices=["map","veuler"],
                   help="veuler = velocity-Euler rollout instead of flow-map jumps.")
    p.add_argument("--xt-harvest-states", action="store_true",
                   help="Sample the training state from along the K-step rollout instead of only "
                        "its endpoint.")
    p.add_argument("--xt-chain-steps", type=int, default=1,
                   help="Build x_t by K chained map jumps from noise (1 = one jump).")
    p.add_argument("--fixed-force-ramp-iters", type=int, default=0,
                   help="With --no-w-rew-norm: linearly ramp w_rew 0->tcf over N iters "
                        "(fixes early-phase over-drive; keeps late self-annealing).")
    p.add_argument("--no-w-rew-norm", action="store_true",
                   help="Disable EMA-normalization of the reward-grad norm. "
                        "When set, w_rew = tcf (constant) instead of tcf · ‖v_base‖/‖∇R‖, "
                        "so g_hat scales like tcf · ‖∇R‖ — same form as vggflow's "
                        "β · ‖∇R‖ for direct β-calibration.")
    p.add_argument("--g-hat-scale", type=float, default=1.0,
                   help="Multiplier on g_hat target itself (NOT on the loss). "
                        "Tests equivalence with scaling the loss via β-outer.")
    p.add_argument("--w-rew-cap", type=float, default=0.0,
                   help="V5b (2026-05-13): clamp(w_rew, max=cap) when >0. "
                        "Bounds direction errors per step when no-norm + raw β. 0 = uncapped.")
    p.add_argument("--P-mean-t", type=float, default=0.4)
    p.add_argument("--P-mean-r", type=float, default=-1.2)
    p.add_argument("--cfg-prob", type=float, default=0.1)
    p.add_argument("--adaptive-p", type=float, default=1.0)
    p.add_argument("--num-tau-samples", type=int, default=1)
    p.add_argument("--num-t-samples", type=int, default=1,
                   help="V4: trajectory-multi-t. If >1, force-split L_control + L_distill "
                        "and call the L_control phase N_t times per micro-batch with "
                        "independent (t_val, τ) draws; backward averages over N_t. "
                        "Default 1 = legacy single-(t,x) per opt step (preserves G1 byte-for-byte).")
    p.add_argument("--distill-mode", type=str, default="dmf", choices=["dmf", "tim"],
                   help="'dmf' = current FTFM distill (L2 mean-flow only). 'tim' = TiM-training-faithful: "
                        "adds directional cosine loss, forces 10%% of distill samples to r=0 (consistency), "
                        "and applies 1/sqrt(t-r+σ_d) time weighting. Uses model diagonal as anchor "
                        "(not ground truth — fine-tuning regime).")
    p.add_argument("--distill-r0-prob", type=float, default=0.1,
                   help="Fraction of distill samples forced to r=0 (NFE=1 path). Active in any "
                        "distill_mode. Matches TiM consistency_ratio=0.1; raise to ~0.3-0.5 to "
                        "explicitly lift NFE=1 reward.")
    p.add_argument("--dual-diag-weight", type=float, default=1.0,
                   help="Dual-adapter: weight on ||v_B(x,t,t) - sg(v_A(x,t,t))||^2 (flow-map boundary condition).")
    p.add_argument("--dual-adapter", action="store_true",
                   help="Separate LoRA adapters: default=dynamics(control), map=distillation.")
    p.add_argument("--distill-longjump-alpha", type=float, default=0.0,
                   help="Selective up-weight on long-jump distill samples: scales per-sample "
                        "mean_flow_loss by (1 + α·|t-r|). α=0 disables (default). α∈[1,2] "
                        "lifts the (t≈1, r≈0) NFE=1 path without dampening small-jump samples.")
    p.add_argument("--no-cast-lora-fp32", action="store_true",
                   help="Disable casting LoRA params to fp32 (default: ON, matches vggflow). "
                        "Critical for Adam state precision with β2=0.999 — bf16 Adam state silently "
                        "rounds (1-β2)·g² to 0 around v≈1, breaking adaptive LR.")
    p.add_argument("--cfg-detach-uncond", action="store_true",
                   help="Detach v_uncond in the CFG forward so the reward gradient flows only through "
                        "the v_cond branch (as in VGG-Flow's CFG wrapper).")
    p.add_argument("--distill-time-sigma-d", type=float, default=1.0,
                   help="In tim distill mode: σ_d in 1/sqrt(t-r+σ_d) time weighting. TiM OT_FM uses 1.0.")
    p.add_argument("--jvp-tangent-clip-ratio", type=float, default=0.0,
                   help="Cap dv_dt_mf JVP tangent at k·||v_tgt_mf||. 0=disabled. "
                        "~5 stabilizes active mode against JVP runaway. "
                        "We observed dv_over_v_tgt=27× during LR=2e-3 crashes.")
    p.add_argument("--tangent-normalize-eps", type=float, default=0.0,
                   help="AYF-style soft tangent normalization: dv_dt → "
                        "dv_dt · ‖v_tgt‖ / (‖dv_dt‖ + eps·‖v_tgt‖). "
                        "0=disabled. eps=0.2 → cap ≈ 5×‖v_tgt‖, smoother than "
                        "jvp_tangent_clip_ratio. Applies to jvp/fd/fd_traj.")
    p.add_argument("--distill-method", type=str, default="jvp",
                   choices=["jvp", "fd", "fd_traj", "semigroup", "psd_shared"],
                   help="jvp (Eulerian, default) | fd (FD ∂F/∂t with x fixed) | "
                        "fd_traj (FD total dF/dt along frozen trajectory; JVP-equivalent direction) | "
                        "semigroup (PSD compositional consistency, WTF Eq. 15) | "
                        "psd_shared (semigroup distillation reusing the reward rollout).")
    p.add_argument("--fd-eps", type=float, default=1e-3,
                   help="Finite-difference step ε for distill_method=fd. "
                        "dv/dt ≈ (F(x, t+ε, r) − F(x, t, r))/ε.")
    # ── Optimal-control variant with iMF-style distillation (FTFMOCImFLoss, --use-imf-oc) ──
    p.add_argument("--use-imf-oc", action="store_true",
                   help="Use FTFMOCImFLoss (iMF-style OC, no data interpolant). "
                        "Overrides FTFMVALUEDMFLoss when set.")
    p.add_argument("--imf-oc-reward-scale", type=float, default=0.1,
                   help="Weight on reward term in iMF-OC loss (default 0.1).")
    p.add_argument("--imf-oc-lambda", type=float, default=1.0,
                   help="Weight λ_imf on the iMF mean-flow distill loss term.")
    p.add_argument("--imf-oc-anchor-weight", type=float, default=0.5,
                   help="Weight on teacher-anchor regularization in iMF-OC loss.")
    p.add_argument("--imf-oc-diag-prob", type=float, default=0.5,
                   help="Probability of using diagonal r=t for the regularization "
                        "anchor (vs off-diagonal r=r_mf).")
    p.add_argument("--imf-oc-rollout-steps", type=int, default=2, choices=[1, 2, 3],
                   help="Reward rollout depth: "
                        "1 = direct noise→x0 (1 jump); "
                        "2 = 1→t_mf→0 (paper-style 2-jump lookahead, default); "
                        "3 = 1→t_mf→r_mf→0 (3 jumps).")
    p.add_argument("--imf-oc-no-stopgrad", action="store_true",
                   help="If set, do NOT stop-grad on JVP outcome inside V_θ. "
                        "iMF default is stopgrad ON; turning OFF allows second-order "
                        "gradient flow through JVP backward.")
    p.add_argument("--imf-oc-adaptive", action="store_true",
                   help="Apply adaptive 1/loss weighting to L_iMF. iMF claims its "
                        "loss has stable variance, so adaptive weighting is "
                        "unnecessary; off by default.")
    p.add_argument("--imf-oc-tangent-normalize-eps", type=float, default=0.0,
                   help="AYF-style soft cap on du_dt inside V_θ construction. "
                        "Critical on TiM (substrate isn't FM-consistent → iMF identity "
                        "has huge du_dt residual). 0=off, 0.2 ≈ cap 5×‖v_diag‖.")
    p.add_argument("--imf-oc-w-rew-norm", action="store_true",
                   help="Enable w_rew gradient normalization (analog of FTFMVALUEDMFLoss "
                        "reward-grad EMA). Replaces -reward_scale·rewards with a surrogate "
                        "whose gradient on x_0 is scaled to TCF·‖v_diag‖_EMA / ‖∂r/∂x_0‖_EMA.")
    p.add_argument("--imf-oc-target-control-fraction", type=float, default=1.0,
                   help="TCF for w_rew normalization. 1.0 = reward force ≈ ‖v_diag‖.")
    p.add_argument("--imf-oc-w-rew-cap", type=float, default=0.0,
                   help="Hard cap on w_rew (0 = uncapped); prevents w_rew from growing without bound "
                        "when the reward-gradient EMA shrinks.")
    p.add_argument("--imf-oc-use-original-mf", action="store_true",
                   help="Ablation: use original mean-flow JVP tangent (e − x, conditional "
                        "velocity) instead of iMF Eq 12 (tangent = v_diag = u_θ(z, t, t)). "
                        "Tests whether iMF's diag-aware tangent is load-bearing.")
    p.add_argument("--lr-warmup-steps", type=int, default=0,
                   help="Linear LR warmup over first N opt steps. 0=disabled. "
                        "Recommended: 50 (~10%% of 500-step run).")
    p.add_argument("--lr-cosine-decay", action="store_true",
                   help="After warmup, cosine-decay LR to lr_min_ratio×base_lr by end.")
    p.add_argument("--lr-min-ratio", type=float, default=0.1,
                   help="LR floor as fraction of base_lr at end of cosine schedule.")
    p.add_argument("--distill-weight", type=float, default=1.0,
                   help="Outer scalar multiplying loss_distillation_adaptive in total loss. "
                        "Default 1.0 (legacy). >1 boosts distillation to keep off-diagonal flow map "
                        "consistent with rapid diagonal updates (control loss).")
    p.add_argument("--num-cost-mc", type=int, default=1,
                   help="Number of MC samples for the cost subgraph. K_cost=4 averages "
                        "∇_x∫₀^t ‖u‖² dτ over 4 random τ values. Cheap per-K (no decode, "
                        "no reward fwd/bwd) — about 30-50%% of reward-MC's per-K cost.")
    p.add_argument("--stretch-weight", type=float, default=0.0,
                   help="log-stretch entropy injection weight (0=off; in-framework diversity; no ruling)")
    p.add_argument("--stretch-sigma", type=float, default=0.1,
                   help="log-stretch perturbation scale x std(x_t); keep small 0.05-0.15")
    p.add_argument("--stretch-floor", type=float, default=1e-4,
                   help="clamp_min on stretch diff norm; guards log to -inf at collapse")
    p.add_argument("--rollout-steps", type=int, default=2, choices=[1, 2],
                   help="Reward rollout. 1 = one flow-map jump x_t→x_0; 2 = two jumps x_t→x_τ→x_0 "
                        "with the running cost at a random τ.")
    p.add_argument("--cost-at-tau", action="store_true",
                   help="With --rollout-steps 1: evaluate the running cost at the sampled τ (x_τ by one "
                        "flow-map jump), as in the paper's estimator, instead of at t.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--batch-size-control", type=int, default=None,
                   help="Per-rank batch for L_control + g_hat subgraphs. "
                        "If different from --batch-size, asymmetric backward "
                        "is enabled: L_control and L_distill are computed via "
                        "separate forward+backward calls with separately-sized "
                        "batches. Memory peak = max(L_ctrl phase, L_distill phase) "
                        "instead of sum. Default "
                        "= --batch-size (combined backward, current behavior).")
    p.add_argument("--batch-size-distill", type=int, default=None,
                   help="Per-rank batch for L_distill (JVP). Default = --batch-size.")
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="Gradient accumulation steps. Effective global batch = "
                        "batch_size · world_size · grad_accum_steps. Set to 4 to "
                        "match vggflow SD3 paper's effective batch=128 with "
                        "BATCH=4 per-GPU on 8 GPUs.")
    p.add_argument("--inner-sgd-steps", type=int, default=1,
                   help="Inner SGD steps per outer iter. Default 1 = standard. >1 = "
                        "vggflow-style inner loop: each inner step is a fresh "
                        "forward+backward+optimizer.step() instead of GA averaging. "
                        "Each inner step uses BS=batch_size (no GA inside). "
                        "Total actual SGD updates per outer = inner_sgd_steps. "
                        "Mutually exclusive with --grad-accum-steps>1 — choose one.")
    p.add_argument("--num-iters", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--wandb-project", default="ftfm_tim")
    p.add_argument("--wandb-name", default="")
    p.add_argument("--checkpoint-steps", type=int, default=1000)
    p.add_argument("--gcs-bucket", type=str, default="",
                   help="If set, copy each saved checkpoint to <gcs-bucket>/checkpoints/<step>.pt "
                        "in the background with `gsutil cp -n`.")
    p.add_argument("--sampling-steps", type=int, default=500)
    p.add_argument("--vis-num-prompts", type=int, default=8,
                   help="Per-rank prompts per set; world=8 → 64/set (8x8 grid).")
    p.add_argument("--vis-num-sets", type=int, default=3,
                   help="Number of prompt sets per sample event.")
    p.add_argument("--vis-num-div-prompts-per-rank", type=int, default=1,
                   help="L: number of distinct prompts per rank for the within-prompt "
                        "diversity probe. Total M = world*L. Default 1 → M=8 (high variance). "
                        "Set 4 → M=32 (matches eval). Each prompt costs K=16 samples; "
                        "diversity probe cost scales linearly with L but is still cheap "
                        "(only runs at sample events).")
    p.add_argument("--vis-nfes", default="1,2,4,8,30",
                   help="Comma-separated NFEs to render at sample events.")
    p.add_argument("--cfg-scale", type=float, default=2.5)
    p.add_argument("--resume-step", type=int, default=0)
    p.add_argument("--frozen-model-mode", default="ema",
                   choices=["ema", "active", "frozen_base"],
                   help="Which model is used for the trajectory rollout in cost/reward "
                        "subgraphs. 'ema' (default) = EMA-LoRA, paper-faithful for ftfm "
                        "but lags behind active LoRA at high ema_decay. 'active' = live "
                        "LoRA, matches vggflow's setup; reward signal is more responsive.")
    p.add_argument("--attn-type", default="flash_attn",
                   choices=["flash_attn", "fused_attn"],
                   help="Attention backend for TiM forward (incl. distill JVP). "
                        "'flash_attn' (default) → patched FA-JVP shim (~10 GB less peak "
                        "memory at BS=8 512², bf16 internal attention math). "
                        "'fused_attn' → PyTorch SDPA MATH backend (more memory, fp32 "
                        "internal attention math, more precise JVP tangent — was the "
                        "production-tested path before FA-JVP integration).")
    args = p.parse_args()

    rank, world, local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)

    use_wandb = False
    if is_main(rank):
        os.makedirs(args.out_dir, exist_ok=True)
        try:
            import wandb
            run_name = args.wandb_name or f"valuegrad_tim_{args.reward}_r{args.lora_rank}"
            # Each launch creates a new wandb run. Set WANDB_RESUME_ID to resume a specific run.
            init_kwargs = dict(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                dir=os.path.join(args.out_dir, "wandb"),
                settings=wandb.Settings(init_timeout=120),
            )
            resume_id = os.environ.get("WANDB_RESUME_ID")
            if resume_id:
                init_kwargs["id"] = resume_id
                init_kwargs["resume"] = "allow"
            wandb.init(**init_kwargs)
            use_wandb = True
        except Exception as e:
            print(f"[warn] wandb init failed: {e}")

    # ── TiM substrate ──────────────────────────────────────────────────────
    pipeline = build_tim_pipeline(config_path=args.config, ckpt_path=args.ckpt,
                                  device=device)
    bundle = pipeline.bundle
    if is_main(rank):
        print(f"[setup] bundle ready (latent {bundle.spatial_downsample}x ds, "
              f"cap_dim {bundle.config.model.network.params.cap_feat_dim})")

    # ── FA-JVP attention (from kyungmnlee/dmf) and bf16 q_norm/k_norm, so the JVP kernel sees
    # matching primal and tangent dtypes ──
    from lib.fa_jvp.tim_patch import apply_to_tim
    _fa_jvp_state = apply_to_tim(model=bundle.model, dtype=bundle.weight_dtype)
    if is_main(rank):
        print(f"[fa_jvp] applied: flash_attn_func patched, "
              f"{len(_fa_jvp_state.get('patched_norms', []))} q_norm/k_norm LayerNorms → bf16-pure")

    # ── peft LoRA ("default" adapter) + PeftEMA ("ema" adapter) ──────────
    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        bias="none", target_modules=DEFAULT_LORA_TARGETS, task_type=None,
    )
    peft_model = apply_lora_to_tim(pipeline.transformer.inner,
                                   rank=args.lora_rank,
                                   alpha=args.lora_alpha,
                                   adapter_name="default")
    pipeline.transformer.inner = peft_model  # rebind into the adapter
    ema = PeftEMA(peft_model, lora_cfg, adapter_name="ema")
    if args.dual_adapter:
        peft_model.add_adapter("map", lora_cfg)
    def _force_da_grad():
        for _n, _p in peft_model.named_parameters():
            if "lora_" in _n and (".default." in _n or ".map." in _n):
                _p.requires_grad_(True)
    if args.dual_adapter: _force_da_grad()
    peft_model.set_adapter("default")
    if args.dual_adapter: _force_da_grad()

    # ── LoRA parameters in fp32 (as in VGG-Flow), so the Adam state is fp32 ──
    if not args.no_cast_lora_fp32:
        try:
            from peft.utils import cast_training_params
            cast_training_params(peft_model, dtype=torch.float32)
            if is_main(rank):
                # Check a LoRA parameter's dtype.
                _lora_dtypes = set()
                for n, p in peft_model.named_parameters():
                    if p.requires_grad and "lora_" in n:
                        _lora_dtypes.add(str(p.dtype))
                print(f"[lora] cast trainable params to fp32 (verified dtypes: {_lora_dtypes})", flush=True)
        except ImportError:
            # Fallback if peft.utils not available
            for n, p in peft_model.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
            if is_main(rank):
                print(f"[lora] cast trainable params to fp32 (manual fallback)", flush=True)

    wrapped = TiMLossAdapter(peft_model, attn_type=args.attn_type).to(device)
    if is_main(rank):
        print(f"[attn] TiMLossAdapter using attn_type='{args.attn_type}' "
              f"({'FA-JVP shim' if args.attn_type == 'flash_attn' else 'SDPA MATH backend'})", flush=True)
    n_train = num_trainable_params(peft_model)
    if is_main(rank):
        print(f"[lora] trainable {n_train:,} (default+ema adapters)")
    # With dual adapters, each iteration runs several gradient-carrying forwards on the same
    # parameters, so DDP uses static_graph=True.
    if args.dual_adapter:
        wrapped_ddp = DDP(wrapped, device_ids=[local_rank], static_graph=True)
    else:
        wrapped_ddp = DDP(wrapped, device_ids=[local_rank], find_unused_parameters=True)

    # ── Reward + decode helpers ──────────────────────────────────────────
    reward_callable = (make_reward_mix(args.reward_mix, device,
                                      standardize=not args.reward_mix_raw)
                       if args.reward_mix else make_reward(args.reward, device))
    reward_fn = RewardScoreShim(reward_callable)
    def decode_func(z):
        return 2.0 * decode_to_image_with_grad(bundle, z) - 1.0

    # ── Optimizer (only "default"-adapter LoRA params) ───────────────────
    _da_sel = (".default.", ".map.") if args.dual_adapter else (".default.",)
    trainable_params = [p for n, p in peft_model.named_parameters()
                        if "lora_" in n and any(s in n for s in _da_sel)]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr,
                            betas=(0.9, args.adam_beta2),
                            weight_decay=args.weight_decay, eps=1e-8)

    # ── LR schedule: linear warmup over lr_warmup_steps, then optional cosine decay to lr_min_ratio · lr ──
    def _lr_lambda(step):
        # Multiplier on the base LR for the current optimizer step.
        if args.lr_warmup_steps > 0 and step < args.lr_warmup_steps:
            return float(step + 1) / float(args.lr_warmup_steps)
        if args.lr_cosine_decay:
            import math
            progress = (step - args.lr_warmup_steps) / max(1, args.num_iters - args.lr_warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return args.lr_min_ratio + (1 - args.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    loss_fn = FTFMVALUEDMFLoss(
        P_mean_t=args.P_mean_t, P_mean_r=args.P_mean_r,
        cfg_prob=args.cfg_prob, adaptive_p=args.adaptive_p,
        ema_decay=args.loss_ema_decay,
        w_rew_warmup_iters=args.w_rew_warmup_iters,
        target_control_fraction=args.target_control_fraction,
        num_tau_samples=args.num_tau_samples,
        num_t_samples=args.num_t_samples,
        cfg_scale=args.cfg_scale,
        eta_mode=args.eta_mode,
        quantile_clip_pct=args.quantile_clip_pct,
        beta_outer=args.beta_outer,
        w_rew_norm=(not args.no_w_rew_norm),
        fixed_force_ramp_iters=args.fixed_force_ramp_iters,
        xt_chain_steps=args.xt_chain_steps,
        xt_harvest_states=args.xt_harvest_states,
        xt_rollout_mode=args.xt_rollout_mode,
        diagonal_lookahead=args.diagonal_lookahead,
        detach_lookahead_velocity=args.detach_lookahead_velocity,
        hjb_consist_weight=args.hjb_consist_weight,
        hjb_eps_time=args.hjb_eps_time,
        hjb_boundary_weight=args.hjb_boundary_weight,
        hjb_nu_adapter=args.hjb_nu_adapter,
        g_hat_scale=args.g_hat_scale,
        w_rew_cap=args.w_rew_cap,
        stretch_weight=args.stretch_weight,
        stretch_sigma=args.stretch_sigma,
        stretch_floor=args.stretch_floor,
        rollout_steps=args.rollout_steps,
        cost_at_tau=args.cost_at_tau,
        num_cost_mc=args.num_cost_mc,
        jvp_tangent_clip_ratio=args.jvp_tangent_clip_ratio,
        tangent_normalize_eps=args.tangent_normalize_eps,
        distill_method=args.distill_method,
        fd_eps=args.fd_eps,
        distill_weight=args.distill_weight,
        distill_mode=args.distill_mode,
        distill_r0_prob=args.distill_r0_prob,
        distill_time_sigma_d=args.distill_time_sigma_d,
        distill_longjump_alpha=args.distill_longjump_alpha,
        cfg_detach_uncond=args.cfg_detach_uncond,
    )
    loss_fn.div_grep = int(args.div_grep)
    if args.dual_adapter:
        def _route_adapter(name):
            peft_model.set_adapter(name)
            _force_da_grad()  # keep both adapters trainable
        loss_fn.set_adapter_fn = _route_adapter
        loss_fn.dual_diag_weight = float(args.dual_diag_weight)
        if is_main(rank):
            print("[dual-adapter] ON: A=default(dynamics/control), B=map(distillation)", flush=True)
    loss_fn.div_baseline = float(args.div_baseline)
    loss_fn.reward_jitter_pairs = int(args.reward_jitter_pairs)
    loss_fn.reward_jitter_sigma = float(args.reward_jitter_sigma)
    if args.use_imf_oc:
        # Optimal-control variant with iMF-style distillation (--use-imf-oc).
        loss_fn = FTFMOCImFLoss(
            P_mean_t=args.P_mean_t, P_std_t=1.0,
            P_mean_r=args.P_mean_r, P_std_r=1.0,
            path_type='linear',
            cfg_scale=args.cfg_scale,
            cfg_detach_uncond=args.cfg_detach_uncond,
            reward_scale=args.imf_oc_reward_scale,
            lambda_imf=args.imf_oc_lambda,
            anchor_weight=args.imf_oc_anchor_weight,
            diag_prob=args.imf_oc_diag_prob,
            imf_jvp_stopgrad=not args.imf_oc_no_stopgrad,
            adaptive_imf=args.imf_oc_adaptive,
            tangent_normalize_eps=args.imf_oc_tangent_normalize_eps,
            rollout_steps=args.imf_oc_rollout_steps,
            w_rew_norm=args.imf_oc_w_rew_norm,
            target_control_fraction=args.imf_oc_target_control_fraction,
            w_rew_cap=args.imf_oc_w_rew_cap,
            use_original_mf=args.imf_oc_use_original_mf,
        )

    # ── Prompts ──────────────────────────────────────────────────────────
    # Prompts from lib.reward_func.prompts (the VGG-Flow prompt sets), or --prompt-file.
    if args.prompt_file is None:
        from lib.reward_func.prompts import _load_lines, read_hpd, read_csv, ASSETS_PATH
        if args.reward == "aesthetic":
            all_prompts = _load_lines("simple_animals.txt")
        elif args.reward in ("hps", "hpsv2"):
            # HPDv2 photo[10:] + paintings[10:]
            all_prompts = read_hpd("photo") + read_hpd("paintings")
        elif args.reward == "pickscore":
            all_prompts = _load_lines("pick_a_pic.txt")
        elif args.reward == "imagereward":
            all_prompts = read_csv(ASSETS_PATH.joinpath("DrawBench Prompts.csv"))
        else:
            raise ValueError(f"unknown --reward={args.reward}")
    else:
        with open(args.prompt_file) as f:
            all_prompts = [json.loads(line)["prompt"] for line in f if line.strip()]
    rng = random.Random(args.seed + rank)

    # Per-rank, per-set prompt slices.
    per_rank = args.vis_num_prompts
    vis_num_sets = args.vis_num_sets
    vis_nfes = parse_nfes(args.vis_nfes)
    per_set_global = per_rank * world
    vis_total_global = per_set_global * vis_num_sets
    # Fixed-seed shuffle so each (set, rank) slice spans the whole prompt list.
    N = len(all_prompts)
    shuffled_idx = list(range(N))
    random.Random(42).shuffle(shuffled_idx)
    if N >= vis_total_global:
        cycled = [all_prompts[i] for i in shuffled_idx[:vis_total_global]]
    else:
        cycled = (([all_prompts[i] for i in shuffled_idx])
                  * (vis_total_global // max(1, N) + 1))[:vis_total_global]
    vis_set_prompts = []
    for set_idx in range(vis_num_sets):
        set_global = cycled[set_idx * per_set_global : (set_idx + 1) * per_set_global]
        vis_set_prompts.append(set_global[rank * per_rank : (rank + 1) * per_rank])
    _base_cache: dict = {}

    # ── Resume ───────────────────────────────────────────────────────────
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    if is_main(rank):
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4, default=str)
    start_iter = 0
    if args.resume_step > 0:
        ckpt_path = os.path.join(ckpt_dir, f"{args.resume_step:07d}.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        params = dict(peft_model.named_parameters())
        for k, v in ckpt["lora"].items():
            if k in params:
                params[k].data.copy_(v.to(params[k].device))
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        loss_fn.ema_base_flow_norm = ckpt["loss_state"]["B"]
        loss_fn.ema_reward_grad_norm = ckpt["loss_state"]["G"]
        start_iter = ckpt["steps"]
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        for _ in range(start_iter):
            rng.sample(all_prompts, args.batch_size)
        if is_main(rank):
            print(f"[resume] loaded {ckpt_path}; resuming from step {start_iter}")

    if is_main(rank):
        print(f"[train] {args.num_iters} iters total (start={start_iter}); "
              f"LR={args.lr} β2={args.adam_beta2} wd={args.weight_decay} "
              f"tcf={args.target_control_fraction} ema_decay={args.ema_decay} "
              f"batch={args.batch_size}")

    Hl, Wl = (args.height // bundle.spatial_downsample,
              args.width // bundle.spatial_downsample)
    Cl = bundle.config.model.network.params.in_channels
    latent_shape = (Cl, Hl, Wl)

    # ── Training loop ────────────────────────────────────────────────────
    accum = max(1, int(args.grad_accum_steps))
    inner_sgd = max(1, int(args.inner_sgd_steps))
    assert not (accum > 1 and inner_sgd > 1), (
        f"GA={accum} and inner_sgd={inner_sgd} are mutually exclusive. "
        f"Pick one: GA averages 5 grads into 1 update; inner_sgd does 5 sequential updates."
    )

    def model_t_fn(x, t, r, y, **kw):
        with torch.no_grad():
            set_adapter_enabled(peft_model, False)
            out = wrapped(x, t, r, y, **kw)
            set_adapter_enabled(peft_model, True)
        return out

    def frozen_model_fn(x, t, r, y, **kw):
        if args.frozen_model_mode == "ema":
            with ema:
                return wrapped(x, t, r, y, **kw)
        elif args.frozen_model_mode == "frozen_base":
            # frozen_base: the reward rollout uses the base weights (LoRA disabled), with the gradient
            # still flowing through the input x_t.
            set_adapter_enabled(peft_model, False)
            out = wrapped(x, t, r, y, **kw)
            set_adapter_enabled(peft_model, True)
            return out
        else:  # "active": the live LoRA model
            return wrapped(x, t, r, y, **kw)

    def cost_model_fn(x, t, r, y, **kw):
        # In frozen_base mode, the control cost uses the active model so it stays nonzero.
        set_adapter_enabled(peft_model, True)
        return wrapped(x, t, r, y, **kw)
    _cost_model_arg = cost_model_fn if args.frozen_model_mode == "frozen_base" else None
    if _cost_model_arg is not None:
        print("[frozen_base] DECOUPLED cost: reward-rollout=base, cost u_t=active-b "
              "(hybrid estimator; OT regularizer preserved).", flush=True)

    # ── Within-prompt diversity (rank 0, DreamSim and CLIP), loaded on first use ──
    diversity_computer = None
    if is_main(rank):
        try:
            from lib.metrics import DiversityComputer
            diversity_computer = DiversityComputer(device=device)
            print(f"[diversity] DiversityComputer initialized (rank 0 only); "
                  f"will probe within-prompt diversity at every sample event.", flush=True)
        except Exception as e:
            print(f"[diversity] init failed: {e}; diversity logging disabled.", flush=True)
            diversity_computer = None

    _bench_t = time.time()
    for it in range(start_iter, args.num_iters):
        peft_model.set_adapter("default")
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_log: dict = {}

        # Separate control and distillation batches when --batch-size-control / --batch-size-distill
        # differ from --batch-size.
        bs_ctrl = args.batch_size_control if args.batch_size_control is not None else args.batch_size
        bs_dist = args.batch_size_distill if args.batch_size_distill is not None else args.batch_size
        # With num_t_samples > 1, split control and distillation and repeat the control phase per
        # micro-batch with independent (t, τ).
        n_t = max(1, int(args.num_t_samples))
        asymmetric = (bs_ctrl != args.batch_size) or (bs_dist != args.batch_size) or (n_t > 1)

        for micro in range(accum):
            if not asymmetric:
                # Combined backward.
                if args.div_grep > 1:
                    assert args.batch_size % args.div_grep == 0, (
                        f"batch_size {args.batch_size} not divisible by div_grep {args.div_grep}")
                    # Each prompt repeated G times (div_grep).
                    prompts = rng.sample(all_prompts, args.batch_size // args.div_grep) * args.div_grep
                else:
                    prompts = rng.sample(all_prompts, args.batch_size)
                cap_feats, _ = encode_captions(bundle, prompts, truncate_to_max_length_in_batch=False)
                loss, log = loss_fn(
                    model=wrapped_ddp,
                    raw_model=wrapped_ddp,
                    y_text_feats=cap_feats,
                    prompts=prompts,
                    model_t=model_t_fn,
                    frozen_model=frozen_model_fn,
                    cost_model=_cost_model_arg,
                    reward_fn=reward_fn,
                    decode_func=decode_func,
                    latent_shape=latent_shape,
                    y_null=bundle.null_cap_feat[:, :cap_feats.size(1)],
                )
                (loss / accum).backward()
                accum_loss += loss.item()
                for k, v in log.items():
                    v_scalar = v.item() if hasattr(v, "item") else v
                    accum_log[k] = accum_log.get(k, 0.0) + v_scalar
            else:
                # Separate forward and backward passes for L_control and L_distill. Peak memory is the larger
                # of the two phases.

                # ── L_control phase, repeated num_t_samples times with independent (t, τ). Each backward is
                # scaled by 1 / (accum · n_t). ──
                for _n_t_idx in range(n_t):
                    prompts_c = rng.sample(all_prompts, bs_ctrl)
                    cap_feats_c, _ = encode_captions(bundle, prompts_c, truncate_to_max_length_in_batch=False)
                    loss_c, log_c = loss_fn(
                        model=wrapped_ddp,
                        raw_model=wrapped_ddp,
                        y_text_feats=cap_feats_c,
                        prompts=prompts_c,
                        model_t=model_t_fn,
                        frozen_model=frozen_model_fn,
                        cost_model=_cost_model_arg,
                        reward_fn=reward_fn,
                        decode_func=decode_func,
                        latent_shape=latent_shape,
                        y_null=bundle.null_cap_feat[:, :cap_feats_c.size(1)],
                        compute_control=True,
                        compute_distill=False,
                    )
                    (loss_c / (accum * n_t)).backward()
                    accum_loss += loss_c.item() / n_t
                    for k, v in log_c.items():
                        v_scalar = v.item() if hasattr(v, "item") else v
                        accum_log[k] = accum_log.get(k, 0.0) + v_scalar / n_t

                # ── L_distill phase at batch=bs_dist ─────────────────────────
                prompts_d = rng.sample(all_prompts, bs_dist)
                cap_feats_d, _ = encode_captions(bundle, prompts_d, truncate_to_max_length_in_batch=False)
                loss_d, log_d = loss_fn(
                    model=wrapped_ddp,
                    raw_model=wrapped_ddp,
                    y_text_feats=cap_feats_d,
                    prompts=prompts_d,
                    model_t=model_t_fn,
                    frozen_model=frozen_model_fn,
                    cost_model=_cost_model_arg,
                    reward_fn=reward_fn,
                    decode_func=decode_func,
                    latent_shape=latent_shape,
                    y_null=bundle.null_cap_feat[:, :cap_feats_d.size(1)],
                    compute_control=False,
                    compute_distill=True,
                )
                (loss_d / accum).backward()
                accum_loss += loss_d.item()
                # Merge the distillation logs.
                for k in ("mean_flow_loss", "loss_distillation_adaptive"):
                    if k in log_d:
                        v = log_d[k]
                        v_scalar = v.item() if hasattr(v, "item") else v
                        accum_log[k] = accum_log.get(k, 0.0) + v_scalar

        if args.dual_adapter and it <= 3 and is_main(rank):
            # Dual adapter: report per-adapter gradient norms for the first iterations.
            _gn = {"A_default": 0.0, "B_map": 0.0}
            for _n, _p in peft_model.named_parameters():
                if "lora_" not in _n or _p.grad is None: continue
                if ".default." in _n: _gn["A_default"] += float(_p.grad.norm()) ** 2
                elif ".map." in _n:   _gn["B_map"]     += float(_p.grad.norm()) ** 2
            print(f"[dual-adapter it{it}] gradnorm A(default/dynamics)={_gn['A_default']**0.5:.6g} "
                  f"B(map/distill)={_gn['B_map']**0.5:.6g} (dw={args.distill_weight})", flush=True)
        grad_norm_pre_clip = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        accum_log["grad_norm_pre_clip"] = float(grad_norm_pre_clip)
        # Peak GPU memory after backward.
        try:
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats()
            accum_log["peak_mem_gb"] = peak_gb
        except Exception:
            pass
        opt.step()
        scheduler.step()
        ema.update(args.ema_decay)
        accum_loss /= accum
        # grad_norm_pre_clip and peak_mem_gb are set once per step, so they are not divided by accum.
        _no_avg = {"grad_norm_pre_clip", "peak_mem_gb"}
        accum_log = {k: (v if k in _no_avg else v / accum) for k, v in accum_log.items()}

        step = it + 1
        if is_main(rank):
            log_str = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in accum_log.items())
            _bench_now = time.time()
            print(f"[it {step}/{args.num_iters}] loss={accum_loss:.4f} {log_str}")
            _bench_t = _bench_now
            if use_wandb:
                import wandb
                # Main diagnostics go to wandb, the rest to stdout.
                _wandb_keys = {
                    "control_loss", "mean_flow_loss", "rewards", "cost_tau",
                    "base_flow_norm_batch", "reward_grad_norm_batch",
                    "cost_grad_norm_batch", "ema_base_flow_norm",
                    "ema_reward_grad_norm", "w_rew",
                    "g_hat_norm", "u_t_norm", "cos_sim_u_ghat",
                    # variance probes
                    "prompt_norm_cov", "tau_cov", "tau_cos_sim",
                }
                payload = {k: v for k, v in accum_log.items() if k in _wandb_keys}
                payload["loss"] = accum_loss
                wandb.log(payload, step=step)

        # ── Checkpoint save (regular OR emergency on SIGTERM) ──────────
        do_ckpt = (step % args.checkpoint_steps == 0 and step > start_iter) or _SHUTDOWN_REQUESTED
        if do_ckpt:
            if is_main(rank):
                ckpt_path = os.path.join(ckpt_dir, f"{step:07d}.pt")
                lora_default = _trainable_lora_state(peft_model)
                torch.save({
                    "lora": lora_default,
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "loss_state": {
                        # EMA state of either loss class.
                        "B": getattr(loss_fn, "ema_base_flow_norm",
                                     getattr(loss_fn, "_B_ema", None)),
                        "G": getattr(loss_fn, "ema_reward_grad_norm",
                                     getattr(loss_fn, "_G_ema", None)),
                    },
                    "steps": step,
                    "args": vars(args),
                }, ckpt_path)
                tag = "EMERGENCY (SIGTERM)" if _SHUTDOWN_REQUESTED else "ckpt"
                print(f"[{tag}] saved {ckpt_path}")
                if args.gcs_bucket:
                    import subprocess
                    remote = f"{args.gcs_bucket.rstrip('/')}/checkpoints/{step:07d}.pt"
                    # Asynchronous no-clobber upload.
                    subprocess.Popen(
                        ["gsutil", "-q", "cp", "-n", ckpt_path, remote],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"[gcs] queued upload → {remote}", flush=True)
            dist.barrier()
        if _SHUTDOWN_REQUESTED:
            if is_main(rank):
                print(f"[shutdown] exiting cleanly at step {step}; resume with --resume-step {step}")
            break

        # ── Multi-set, multi-NFE qualitative samples ─────────────────────
        if (args.sampling_steps < 99999 and (step == 10 or step == 100 or
                (step % args.sampling_steps == 0 and step > start_iter))):

            def _render_set(prompts_local, nfe, set_idx, kind):
                samp_seed = args.seed + 99991 + rank * 1009 + set_idx * 7919
                caps, _ = encode_captions(bundle, prompts_local)
                if kind == "tuned":
                    peft_model.set_adapter("default")
                else:
                    set_adapter_enabled(peft_model, False)
                lat = sample_with_cfg(
                    wrapped, bundle, caps,
                    num_steps=nfe, cfg_scale=args.cfg_scale,
                    height=args.height, width=args.width,
                    seed=samp_seed, device=device,
                )
                if kind == "base":
                    set_adapter_enabled(peft_model, True)
                return lat.contiguous()

            def _gather_score_and_grid(lat_local, prompts_local):
                imgs = decode_to_image(bundle, lat_local.to(bundle.weight_dtype)).clamp(0, 1)
                rewards_local = reward_callable(imgs, prompts_local).float().contiguous()
                rewards_all = [torch.zeros_like(rewards_local) for _ in range(world)]
                dist.all_gather(rewards_all, rewards_local)
                rewards_full = torch.cat(rewards_all)
                lat_all = [torch.zeros_like(lat_local) for _ in range(world)]
                dist.all_gather(lat_all, lat_local)
                lat_full = torch.cat(lat_all, dim=0) if is_main(rank) else None
                return rewards_full, lat_full

            event_log = {}
            for set_idx in range(vis_num_sets):
                prompts_local = vis_set_prompts[set_idx]
                for nfe in vis_nfes:
                    lat_t = _render_set(prompts_local, nfe, set_idx, "tuned")
                    r_t, lat_t_full = _gather_score_and_grid(lat_t, prompts_local)
                    cache_key = (set_idx, nfe)
                    if cache_key not in _base_cache:
                        with torch.no_grad():
                            lat_b = _render_set(prompts_local, nfe, set_idx, "base")
                            r_b, lat_b_full = _gather_score_and_grid(lat_b, prompts_local)
                        _base_cache[cache_key] = {
                            "rewards_full": r_b.detach().clone(),
                            "lat_full": lat_b_full.detach().clone() if lat_b_full is not None else None,
                        }
                    r_b = _base_cache[cache_key]["rewards_full"]
                    lat_b_full = _base_cache[cache_key]["lat_full"]

                    if is_main(rank):
                        r_t_mean, r_b_mean = r_t.mean(), r_b.mean()
                        delta = (r_t_mean - r_b_mean).item()
                        r_t_sem = (r_t.std(unbiased=False) / math.sqrt(r_t.numel())).item()
                        r_b_sem = (r_b.std(unbiased=False) / math.sqrt(r_b.numel())).item()
                        tag = f"set{set_idx}_nfe{nfe}"
                        if use_wandb:
                            import wandb
                            event_log[f"samples/{tag}_tuned"] = wandb.Image(
                                latents_to_uint8_grid(decode_to_image, bundle, lat_t_full),
                                caption=f"step {step} | tuned | {tag} | n={lat_t_full.size(0)}")
                            if step <= 100 or step == args.sampling_steps + start_iter:
                                event_log[f"samples/{tag}_base"] = wandb.Image(
                                    latents_to_uint8_grid(decode_to_image, bundle, lat_b_full),
                                    caption=f"base | {tag} | n={lat_b_full.size(0)}")
                            # Per-image wandb logging: the first min(8, B) samples with prompt and reward.
                            n_img = min(8, lat_t_full.size(0))
                            with torch.no_grad():
                                imgs_t = decode_to_image(
                                    bundle, lat_t_full[:n_img].to(bundle.weight_dtype)
                                ).clamp(0, 1).float().cpu().numpy()
                            wandb_images_t = []
                            for i in range(n_img):
                                arr = (imgs_t[i].transpose(1, 2, 0) * 255).astype(np.uint8)
                                prompt_str = prompts_local[i % len(prompts_local)]
                                # r_t is (B*world,), index 0..B-1 are rank-0 samples.
                                rew_val = r_t[i].item() if i < r_t.numel() else float("nan")
                                wandb_images_t.append(wandb.Image(
                                    arr, caption=f"{prompt_str} | {rew_val:.2f}"))
                            event_log[f"images/{tag}_tuned"] = wandb_images_t
                            event_log[f"reward/{tag}_tuned"] = r_t_mean.item()
                            event_log[f"reward/{tag}_base"] = r_b_mean.item()
                            event_log[f"reward/{tag}_delta"] = delta
                            event_log[f"reward/{tag}_tuned_sem"] = r_t_sem
                            event_log[f"reward/{tag}_base_sem"] = r_b_sem
                        print(f"[samples] step {step} {tag} | n={r_t.numel()} "
                              f"r_tuned={r_t_mean.item():.4f}±{r_t_sem:.4f} "
                              f"r_base={r_b_mean.item():.4f}±{r_b_sem:.4f} "
                              f"Δ={delta:+.4f}")

            # ── Within-prompt diversity at NFE 20: K = 4 samples per prompt, gathered on rank 0 ──
            try:
                K_div = 16
                L_div = max(1, int(args.vis_num_div_prompts_per_rank))
                # Up to L_div prompts per rank from the first set, cycling if needed.
                rank_prompts = vis_set_prompts[0]
                div_clip_per, div_ds_per = [], []
                for l in range(L_div):
                    div_prompt = rank_prompts[l % len(rank_prompts)]
                    div_caps, _ = encode_captions(bundle, [div_prompt] * K_div,
                                                  truncate_to_max_length_in_batch=False)
                    samp_seed_div = args.seed + 12345 + rank * 1009 + step * 7919 + l * 31337
                    peft_model.set_adapter("default")
                    div_lat = sample_with_cfg(
                        wrapped, bundle, div_caps,
                        num_steps=20, cfg_scale=args.cfg_scale,
                        height=args.height, width=args.width,
                        seed=samp_seed_div, device=device,
                    )
                    div_imgs_local = decode_to_image(
                        bundle, div_lat.to(bundle.weight_dtype)
                    ).clamp(0, 1).contiguous()  # (K_div, 3, H, W)
                    div_imgs_all = [torch.zeros_like(div_imgs_local) for _ in range(world)]
                    dist.all_gather(div_imgs_all, div_imgs_local)
                    if is_main(rank) and diversity_computer is not None:
                        from lib.metrics import _mean_pairwise_l2_squared
                        for r_idx in range(world):
                            rank_imgs = div_imgs_all[r_idx]
                            cf = diversity_computer.embed_clip(rank_imgs)
                            df = diversity_computer.embed_dreamsim(rank_imgs)
                            div_clip_per.append(_mean_pairwise_l2_squared(cf).item())
                            div_ds_per.append(_mean_pairwise_l2_squared(df).item())
                if is_main(rank) and diversity_computer is not None:
                    M_eff = len(div_clip_per)  # = world * L_div
                    div_clip_mean = sum(div_clip_per) / max(1, M_eff)
                    div_ds_mean = sum(div_ds_per) / max(1, M_eff)
                    event_log["diversity_inprompt_clip"] = div_clip_mean
                    event_log["diversity_inprompt_dreamsim"] = div_ds_mean
                    event_log["diversity_inprompt_M"] = M_eff
                    event_log["diversity_inprompt_K"] = K_div
                    print(f"[samples] step {step} diversity | "
                          f"clip={div_clip_mean:.4f} dreamsim={div_ds_mean:.4f} "
                          f"(M={M_eff} prompts × K={K_div} samples)", flush=True)
            except Exception as _e:
                if is_main(rank):
                    print(f"[diversity] probe failed at step {step}: {_e}", flush=True)

            if is_main(rank) and use_wandb:
                deltas = [v for k, v in event_log.items() if k.endswith("_delta")]
                if deltas:
                    event_log["reward/delta_mean"] = sum(deltas) / len(deltas)
                import wandb
                wandb.log(event_log, step=step)

    if is_main(rank):
        print("[train] DONE.")
        if use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
