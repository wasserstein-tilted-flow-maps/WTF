"""WTF objective for the text-to-image lane (TiM-T2I substrate).

`FTFMVALUEDMFLoss` is the WTF objective, adapted to TiM's model-call signature through
`lib/ftfm/wrappers.py:TiMLossAdapter`. `FTFMOCImFLoss` is an alternative variant (`--use-imf-oc`).
"""
from __future__ import annotations

import math
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F


def ln_sampler(z: torch.Tensor, path_type: str = "linear") -> torch.Tensor:
    if path_type == "linear":
        return torch.sigmoid(z)
    if path_type == "cosine":
        return 2 / np.pi * torch.atan(z)
    raise NotImplementedError(path_type)


class FTFMVALUEDMFLoss:
    """WTF loss: diagonal value-gradient regression plus flow-map self-distillation.

    Time runs from noise (t = 1) to data (t = 0). Per sample:
        x_t     = rollout of the frozen reference from noise to t
        u_t     = v_student(x_t, t, t) - v_teacher(x_t, t, t)          (control at the diagonal)
        g_hat   = -(w_rew * grad_x r(x_0(x_t)) - grad_x cost)           (value-gradient target)
        w_rew   = tcf * EMA[|v_teacher|] / EMA[|grad r|]                 (normalized reward scale)
        L       = mean |u_t - g_hat|^2 + distill_weight * L_distill

    Main arguments (the shipped config sets the reported values):
        target_control_fraction  reward scale lambda (tcf); normalized as above unless w_rew_norm=False
        rollout_steps            1: one jump x_t -> x_0; 2: x_t -> x_tau -> x_0 with the cost at tau
        cost_at_tau              rollout_steps 1: running cost at the sampled tau (x_tau by one jump), as in
                                 the paper's estimator, instead of at t
        detach_lookahead_velocity  True: Euclidean reward gradient (identity Jacobian); False: exact
        reward_jitter_pairs / reward_jitter_sigma  antithetic smoothing of grad r (rollout_steps 1)
        quantile_clip_pct        per-sample clip of |grad r| at this batch quantile (0 disables)
        distill_weight, distill_mode, distill_method  weight and form of the self-distillation term
        cfg_scale, cfg_detach_uncond  classifier-free guidance inside the loss
    The remaining arguments are ablations and diagnostics. Their defaults leave them off.

    Token conditioning is off for TiM.
    """
    def __init__(
        self,
        P_mean: float = -0.4, P_std: float = 1.0,
        P_mean_t: float = -0.2, P_std_t: float = 1.0,
        P_mean_r: float = 0.2,  P_std_r: float = 1.0,
        cfg_prob: float = 0.1,
        path_type: str = "linear",
        adaptive_p: float = 1.0,
        ema_decay: float = 0.98,
        w_rew_warmup_iters: int = 0,
        target_control_fraction: float = 1.0,
        cost_at_tau: bool = False,
        rollout_steps: int = 2,
        num_cost_mc: int = 1,
        jvp_tangent_clip_ratio: float = 0.0,
        tangent_normalize_eps: float = 0.0,
        distill_method: str = 'jvp',
        fd_eps: float = 1e-3,
        distill_weight: float = 1.0,
        distill_mode: str = 'dmf',
        distill_r0_prob: float = 0.1,
        distill_time_sigma_d: float = 1.0,
        distill_longjump_alpha: float = 0.0,
        cfg_detach_uncond: bool = False,
        num_tau_samples: int = 1,
        num_t_samples: int = 1,
        cfg_scale: float = 1.0,
        eta_mode: str = 'none',
        quantile_clip_pct: float = 0.0,
        beta_outer: float = 1.0,
        w_rew_norm: bool = True,
        fixed_force_ramp_iters: int = 0,
        xt_chain_steps: int = 1,
        xt_harvest_states: bool = False,
        diagonal_lookahead: bool = False,
        xt_rollout_mode: str = "map",
        detach_lookahead_velocity: bool = False,
        hjb_consist_weight: float = 0.0,
        hjb_eps_time: float = 1e-4,
        hjb_boundary_weight: float = 0.0,
        hjb_nu_adapter: bool = False,
        g_hat_scale: float = 1.0,
        w_rew_cap: float = 0.0,
        stretch_weight: float = 0.0,
        stretch_sigma: float = 0.1,
        stretch_floor: float = 1e-4,
    ):
        self.P_mean, self.P_std = P_mean, P_std
        self.P_mean_t, self.P_std_t = P_mean_t, P_std_t
        self.P_mean_r, self.P_std_r = P_mean_r, P_std_r
        self.cfg_prob = cfg_prob
        self.path_type = path_type
        self.adaptive_p = adaptive_p
        self.ema_decay = ema_decay
        self.target_control_fraction = target_control_fraction
        self.num_tau_samples = max(1, int(num_tau_samples))
        self.num_t_samples = max(1, int(num_t_samples))
        self.cfg_scale = float(cfg_scale)
        self.eta_mode = str(eta_mode)
        self.quantile_clip_pct = float(quantile_clip_pct)
        self.beta_outer = float(beta_outer)
        self.w_rew_norm = bool(w_rew_norm)
        self.fixed_force_ramp_iters = int(fixed_force_ramp_iters)
        self.xt_chain_steps = int(xt_chain_steps)
        self.xt_harvest_states = bool(xt_harvest_states)
        self.xt_rollout_mode = str(xt_rollout_mode)
        self.diagonal_lookahead = bool(diagonal_lookahead)
        self.detach_lookahead_velocity = bool(detach_lookahead_velocity)
        self.hjb_consist_weight = float(hjb_consist_weight)
        self.hjb_eps_time = float(hjb_eps_time)
        self.hjb_boundary_weight = float(hjb_boundary_weight)
        self.hjb_nu_adapter = bool(hjb_nu_adapter)
        self.g_hat_scale = float(g_hat_scale)
        self.w_rew_cap = float(w_rew_cap)
        self.stretch_weight = float(stretch_weight)
        self.stretch_sigma = float(stretch_sigma)
        self.stretch_floor = float(stretch_floor)
        self.rollout_steps = int(rollout_steps)
        self.cost_at_tau = bool(cost_at_tau)
        assert self.rollout_steps in (1, 2), f"rollout_steps must be 1 or 2; got {self.rollout_steps}"
        self.num_cost_mc = max(1, int(num_cost_mc))
        self.jvp_tangent_clip_ratio = float(jvp_tangent_clip_ratio)
        self.tangent_normalize_eps = float(tangent_normalize_eps)
        self.distill_method = str(distill_method)
        assert self.distill_method in ('jvp', 'fd', 'fd_traj', 'semigroup', 'psd_shared'), \
            f"distill_method must be jvp/fd/fd_traj/semigroup/psd_shared, got {self.distill_method}"
        # psd_shared: semigroup distillation that reuses x_t from the value-estimate rollout.
        # With probability 1 - p it anchors to the data endpoint (only the direct jump is new
        # compute); with probability p it uses an interior triplet (t, (t + tau) / 2, tau).
        self.psd_shared_p_interior = 0.5
        self.fd_eps = float(fd_eps)
        self.distill_weight = float(distill_weight)
        self.distill_mode = str(distill_mode)
        assert self.distill_mode in ('dmf', 'tim'), f"distill_mode must be 'dmf' or 'tim'; got {self.distill_mode}"
        self.distill_r0_prob = float(distill_r0_prob)
        self.distill_time_sigma_d = float(distill_time_sigma_d)
        self.distill_longjump_alpha = float(distill_longjump_alpha)
        self.cfg_detach_uncond = bool(cfg_detach_uncond)
        # Dataset-level reward-norm trackers, EMA'd across batches.
        self.ema_base_flow_norm: torch.Tensor | None = None
        self.ema_reward_grad_norm: torch.Tensor | None = None
        self.ema_stretch_grad_norm: torch.Tensor | None = None
        self.w_rew_warmup_iters = int(w_rew_warmup_iters)
        self._ema_update_count = 0   # incremented each _update_ema call (per name shared)

    def _cfg_call(self, model_fn, x, t, r, y_cond, y_null, cond, *, t_endpoint: bool = False):
        """Classifier-free guidance around a model call, as in TiM's `forward_with_cfg`.

        One batched forward of [cond; uncond], blended as v_uncond + cfg_scale (v_cond - v_uncond).
        CFG is skipped at t in {0, 1}. Pass `t_endpoint=True` for anchors at t = 1.
        """
        if self.cfg_scale <= 1.0 or t_endpoint:
            return model_fn(x, t, r, y_cond, **cond)
        x2 = torch.cat([x, x], dim=0)
        t2 = torch.cat([t, t], dim=0)
        r2 = torch.cat([r, r], dim=0)
        y2 = torch.cat([y_cond, y_null.to(y_cond.dtype)], dim=0)
        cond2 = {}
        for k, v in cond.items():
            if torch.is_tensor(v):
                cond2[k] = torch.cat([v, v], dim=0)
            else:
                cond2[k] = v
        v_all = model_fn(x2, t2, r2, y2, **cond2)
        v_cond, v_uncond = v_all.chunk(2, dim=0)
        if self.cfg_detach_uncond:
            # Detach the unconditional branch so the reward gradient flows only through v_cond.
            v_uncond = v_uncond.detach()
        return v_uncond + self.cfg_scale * (v_cond - v_uncond)

    # ── Per-sample sampling of times ────────────────────────────────────────
    def sample_times(self, bsz: int):
        t_val = torch.rand((bsz, 1, 1, 1))
        tau   = torch.rand((bsz, 1, 1, 1)) * t_val
        ln_1 = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std_t + self.P_mean_t, self.path_type)
        ln_2 = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std_r + self.P_mean_r, self.path_type)
        t_mf, r_mf = torch.maximum(ln_1, ln_2), torch.minimum(ln_1, ln_2)
        # A fraction of distillation samples use r = 0 (TiM's consistency_ratio), which trains the
        # jump v(x, t, 0) used at NFE 1 and in the reward lookahead.
        if self.distill_r0_prob > 0:
            n_r0 = max(1, int(self.distill_r0_prob * bsz))
            r_mf[-n_r0:] = 0.0
        return t_val, tau, t_mf, r_mf

    def _sa(self, name):
        """Switch the active LoRA adapter (A = dynamics, B = flow map). No-op without `set_adapter_fn`."""
        _fn = getattr(self, "set_adapter_fn", None)
        if _fn is not None:
            _fn(name)

    @staticmethod
    def _batch_mean_norm(v: torch.Tensor) -> torch.Tensor:
        return v.flatten(1).norm(dim=-1).mean().detach()

    def _update_ema(self, name: str, new_value: torch.Tensor) -> torch.Tensor:
        old = getattr(self, name)
        # Lock the EMAs after `w_rew_warmup_iters` training iterations.
        if (self.w_rew_warmup_iters > 0
                and old is not None
                and self._ema_update_count >= self.w_rew_warmup_iters):
            return old.detach()
        updated = new_value if old is None else self.ema_decay * old + (1 - self.ema_decay) * new_value
        setattr(self, name, updated.detach())
        return updated.detach()

    # ── Main call ───────────────────────────────────────────────────────────
    def __call__(
        self,
        model,                  # peft+DDP-wrapped TiM (TiMLossAdapter)
        raw_model,              # unwrapped (for jvp)
        y_text_feats,           # (B, L, D) Gemma cap_feats
        prompts,                # list[str]
        *,
        model_t,                # callable: (x, t, r, y, **cond) → velocity (frozen teacher)
        frozen_model,           # callable: same signature (EMA snapshot)
        reward_fn,              # object with .score(images, prompts) → (B,)
        decode_func,            # callable: latents → image in [-1, 1]
        latent_shape,           # (C, H, W)
        y_null,                 # (1, L, D) or (B, L, D), TiM null cap-feat for CFG
        compute_control: bool = True,   # if False, skip cost/reward subgraphs and L_control
        compute_distill: bool = True,   # if False, skip the JVP and L_distill
        cost_model=None,        # callable for the u_t control-cost term; None => frozen_model.
                                # In frozen_base rollout mode, pass the active model so the cost stays nonzero.
    ):
        device = y_text_feats.device
        bsz = y_text_feats.shape[0]
        y = y_text_feats
        # Training-iteration counter for the EMA warmup.
        self._ema_update_count += 1
        if y_null.shape[0] == 1:
            y_null = y_null.expand(bsz, -1, -1)
        elif y_null.shape[0] != bsz:
            y_null = y_null[:bsz]

        t_val, tau, t_mf, r_mf = self.sample_times(bsz)
        t_val, tau = t_val.to(device), tau.to(device)
        t_mf, r_mf = t_mf.to(device), r_mf.to(device)

        # Token conditioning is off, so these are constants.
        tcf = torch.full((bsz, 1, 1, 1), float(self.target_control_fraction), device=device)
        t_min = torch.zeros((bsz, 1, 1, 1), device=device)
        t_max = torch.ones((bsz, 1, 1, 1), device=device)
        cond = dict(tcf=tcf.flatten(), t_min=t_min.flatten(), t_max=t_max.flatten())
        teacher_cond = dict(tcf=0.0, t_min=0.0, t_max=1.0)

        C, H, W = latent_shape
        noise = torch.randn(bsz, C, H, W, device=device)
        t_ones = torch.ones((bsz, 1, 1, 1), device=device)
        t_zeros = torch.zeros((bsz, 1, 1, 1), device=device)

        # ── Rollout of the frozen reference to x_t ──
        if compute_control:
            # t_endpoint=True skips CFG at t = 1, as at inference.
            with torch.no_grad():
                _nchain = int(getattr(self, "xt_chain_steps", 1))
                if _nchain <= 1:
                    x_t = noise + (t_val - t_ones) * self._cfg_call(
                        frozen_model, noise, t_ones, t_val, y, y_null, cond, t_endpoint=True)
                else:
                    # Build x_t by chaining K map jumps 1 -> t instead of one jump, so training
                    # states come from an on-trajectory rollout of the frozen policy.
                    x_t = noise
                    _t_prev = t_ones
                    _harvest = bool(getattr(self, "xt_harvest_states", False))
                    _states, _times = [], []
                    for _k in range(1, _nchain + 1):
                        _t_next = t_ones + (t_val - t_ones) * (float(_k) / float(_nchain))
                        if str(getattr(self, "xt_rollout_mode", "map")) == "veuler":
                            # Velocity-Euler step: evaluate the diagonal velocity v(x, t, t) and take an Euler step.
                            _vstep = self._cfg_call(frozen_model, x_t, _t_prev, _t_prev, y, y_null, cond)
                        else:
                            _vstep = self._cfg_call(frozen_model, x_t, _t_prev, _t_next, y, y_null, cond,
                                                    t_endpoint=True)
                        x_t = x_t + (_t_next - _t_prev) * _vstep
                        _t_prev = _t_next
                        if _harvest:
                            _states.append(x_t); _times.append(_t_next)
                    if _harvest and len(_states) > 1:
                        # Draw the training state uniformly along the rollout, which matches its state marginal
                        # at lower cost than using all K states.
                        _pick = torch.randint(0, len(_states), (x_t.shape[0],), device=x_t.device)
                        _S = torch.stack(_states, dim=0)          # (K, B, ...)
                        _T = torch.stack(_times, dim=0)           # (K, B, 1,1,1) or (K,B)
                        _idx = _pick.view(1, -1, *([1] * (_S.dim() - 2))).expand(1, *_S.shape[1:])
                        x_t = torch.gather(_S, 0, _idx).squeeze(0)
                        _ti = _pick.view(1, -1, *([1] * (_T.dim() - 2))).expand(1, *_T.shape[1:])
                        t_val = torch.gather(_T, 0, _ti).squeeze(0)
            x_t = x_t.detach()

            # Teacher velocity at x_t (no grad). The student velocity is computed after the cost and
            # reward subgraphs to save memory.
            with torch.no_grad():
                v_teacher_t = self._cfg_call(model_t, x_t, t_val, t_val, y, y_null, teacher_cond)
        else:
            x_t = None
            v_teacher_t = None

        # ── Reward and cost subgraphs: one τ per step, shared by both, with both gradients taken
        # with respect to the same x_t leaf.
        # rollout_steps 1: one jump x_t → x_0, cost at the sampled τ (cost_at_tau) or at t.
        # rollout_steps 2: x_t → x_τ → x_0, cost at τ.
        # psd_shared reuses the chain endpoints (rollout_steps 2 only).
        _psd_x_tau_target = None
        _psd_x_0_hat_target = None
        grad_stretch = None  # log-stretch injection (set below iff stretch_weight>0)
        _psd_tau_target = None

        if compute_control:
            x_t_leaf = x_t.detach().requires_grad_(True)

            if self.rollout_steps == 2:
                # Shared τ ∈ [0, t_val], reusing the `tau` drawn in sample_times().
                tau_shared = tau.detach()
                # Shared t → τ forward, with gradient through x_t_leaf.
                self._sa("map")  # off-diagonal jump, adapter B
                v_to_tau = self._cfg_call(
                    frozen_model, x_t_leaf, t_val, tau_shared, y, y_null, cond)
                x_tau = x_t_leaf + (tau_shared - t_val) * v_to_tau

                # ── Reward: continue τ→0 ──
                v_to_zero = self._cfg_call(
                    frozen_model, x_tau, tau_shared, t_zeros, y, y_null, cond)
                self._sa("default")
                if bool(getattr(self, "detach_lookahead_velocity", False)):
                    # Euclidean reward gradient: dx_0/dx_t = I through both hops. x_tau keeps its graph for the
                    # running cost.
                    x_0_hat = (x_t_leaf + (tau_shared - t_val) * v_to_tau.detach()
                               + (t_zeros - tau_shared) * v_to_zero.detach())
                else:
                    x_0_hat = x_tau + (t_zeros - tau_shared) * v_to_zero
                x_0_img = (decode_func(x_0_hat) + 1.0) / 2.0
                rewards_raw_log = reward_fn.score(x_0_img, prompts)

                # Detached chain endpoints for psd_shared distillation.
                _psd_x_tau_target = x_tau.detach().clone()
                _psd_x_0_hat_target = x_0_hat.detach().clone()
                _psd_tau_target = tau_shared

                # ── Cost: u(x_τ, τ) ──
                if compute_control:
                    with torch.no_grad():
                        v_teacher_tau = self._cfg_call(
                            model_t, x_tau.detach(), tau_shared, tau_shared,
                            y, y_null, teacher_cond)
                    _cost_model = cost_model if cost_model is not None else frozen_model
                    u_tau = self._cfg_call(
                        _cost_model, x_tau, tau_shared, tau_shared,
                        y, y_null, cond) - v_teacher_tau
                    cost_tau_log = torch.mean(
                        u_tau ** 2, dim=list(range(1, u_tau.ndim)))
                    cost_weighted_sum = (t_val.squeeze() * 0.5 * cost_tau_log).sum()
                else:
                    cost_tau_log = torch.zeros(bsz, device=device, dtype=x_t.dtype)

            else:  # rollout_steps == 1
                # Direct x_t → x_0 (no τ).
                if bool(getattr(self, "diagonal_lookahead", False)):
                    # Diagonal velocity v(x_t, t, t) on adapter A. Needed when distill_weight = 0, since adapter B
                    # then receives no gradient.
                    v_to_zero = self._cfg_call(
                        frozen_model, x_t_leaf, t_val, t_val, y, y_null, cond)
                else:
                    self._sa("map")  # off-diagonal jump v(x, t, 0), adapter B
                    v_to_zero = self._cfg_call(
                        frozen_model, x_t_leaf, t_val, t_zeros, y, y_null, cond)
                    self._sa("default")
                if bool(getattr(self, "detach_lookahead_velocity", False)):
                    # Euclidean reward gradient: with the lookahead velocity detached,
                    # dx0/dxt = I and the gradient is grad r with no Jacobian pullback.
                    v_to_zero = v_to_zero.detach()
                x_0_hat = x_t_leaf + (t_zeros - t_val) * v_to_zero
                x_0_img = (decode_func(x_0_hat) + 1.0) / 2.0
                rewards_raw_log = reward_fn.score(x_0_img, prompts)

                # ── Running cost: at the sampled τ (cost_at_tau) or at t_val ──
                if compute_control:
                    _cost_model = cost_model if cost_model is not None else frozen_model
                    if self.cost_at_tau:
                        # Cost at the sampled τ, at x_τ = X_{t,τ}(x_t) from one map jump, with gradient through the jump.
                        tau_c = tau.detach()
                        self._sa("map")
                        v_to_tau_c = self._cfg_call(
                            frozen_model, x_t_leaf, t_val, tau_c, y, y_null, cond)
                        self._sa("default")
                        x_tau_c = x_t_leaf + (tau_c - t_val) * v_to_tau_c
                        with torch.no_grad():
                            v_teacher_tau_c = self._cfg_call(
                                model_t, x_tau_c.detach(), tau_c, tau_c,
                                y, y_null, teacher_cond)
                        u_t = self._cfg_call(
                            _cost_model, x_tau_c, tau_c, tau_c,
                            y, y_null, cond) - v_teacher_tau_c
                    else:
                        # Cost at t_val, reusing v_teacher_t from the anchor block.
                        u_t = self._cfg_call(
                            _cost_model, x_t_leaf, t_val, t_val,
                            y, y_null, cond) - v_teacher_t.detach()
                    cost_tau_log = torch.mean(
                        u_t ** 2, dim=list(range(1, u_t.ndim)))
                    cost_weighted_sum = (t_val.squeeze() * 0.5 * cost_tau_log).sum()
                else:
                    cost_tau_log = torch.zeros(bsz, device=device, dtype=x_t.dtype)

            assert rewards_raw_log.shape == (bsz,), \
                f"rewards must be shape ({bsz},), got {rewards_raw_log.shape}"

            # Two backward passes through the shared graph (retain on first).
            if compute_control:
                grad_reward = torch.autograd.grad(
                    rewards_raw_log.sum(), x_t_leaf, retain_graph=True)[0].detach()
                grad_cost = torch.autograd.grad(
                    cost_weighted_sum, x_t_leaf)[0].detach()
                # ── Reward-gradient smoothing: average ∇r over antithetic perturbations of x_t at scale
                # sigma · std(x_t). Off when reward_jitter_pairs = 0 (rollout_steps 1 only).
                _jp = int(getattr(self, "reward_jitter_pairs", 0))
                if _jp > 0:
                    _js = float(getattr(self, "reward_jitter_sigma", 0.05))
                    _sig = _js * x_t.detach().std()
                    _acc = None; _cnt = 0
                    for _pi in range(_jp):
                        _eps = torch.randn_like(x_t)
                        for _sgn in (1.0, -1.0):
                            _xp = (x_t.detach() + _sgn * _sig * _eps).requires_grad_(True)
                            # Same adapter routing as the main path (B for the off-diagonal jump).
                            self._sa("map")
                            _vp = self._cfg_call(frozen_model, _xp, t_val, t_zeros, y, y_null, cond)
                            self._sa("default")
                            # Same detach setting as the main path.
                            if bool(getattr(self, "detach_lookahead_velocity", False)):
                                _vp = _vp.detach()
                            _x0p = _xp + (t_zeros - t_val) * _vp
                            _imgp = (decode_func(_x0p) + 1.0) / 2.0
                            _rp = reward_fn.score(_imgp, prompts)
                            _gp = torch.autograd.grad(_rp.sum(), _xp)[0].detach()
                            _acc = _gp if _acc is None else _acc + _gp
                            _cnt += 1
                    grad_reward = _acc / _cnt
            else:
                grad_reward = torch.autograd.grad(
                    rewards_raw_log.sum(), x_t_leaf)[0].detach()
                grad_cost = torch.zeros_like(x_t)

            # ── Log-stretch term: τ · E[log ‖J_T δ‖] via antithetic ±δ perturbations of x_t, maximized.
            # On when stretch_weight > 0 (rollout_steps 1 only).
            if compute_control and self.stretch_weight > 0.0:
                _sig = self.stretch_sigma * x_t_leaf.detach().std()
                _delta = torch.randn_like(x_t_leaf)
                _S = x_t_leaf.new_zeros(())
                for _s in (1.0, -1.0):
                    _xtp = x_t_leaf + (_s * _sig) * _delta
                    self._sa("map")   # off-diagonal jump -> B
                    _x0p = _xtp + (t_zeros - t_val) * self._cfg_call(
                        frozen_model, _xtp, t_val, t_zeros, y, y_null, cond)
                    self._sa("default")
                    _diff = (_x0p - x_0_hat.detach()).flatten(1)
                    _S = _S + torch.log(_diff.norm(dim=1).clamp_min(self.stretch_floor)).mean()
                _S = _S / 2.0
                grad_stretch = torch.autograd.grad(_S, x_t_leaf, retain_graph=False)[0].detach()
            rewards_raw_log = rewards_raw_log.detach()
            grad_reward_samples = [grad_reward]  # length-1 list for diagnostic uniformity
            del x_t_leaf, x_0_hat, x_0_img

            # ── Variance diagnostics: spread of ‖∇r‖ across τ samples (K ≥ 2) and across prompts ──
            with torch.no_grad():
                # Per-prompt-norm spread (always available)
                per_prompt_grad_norm = grad_reward.flatten(1).norm(dim=-1)  # (B,)
                _prompt_norm_mean = per_prompt_grad_norm.mean().item()
                _prompt_norm_std = per_prompt_grad_norm.std(unbiased=False).item()
                _prompt_norm_cov = _prompt_norm_std / max(_prompt_norm_mean, 1e-9)

                # Constant with one τ sample.
                tau_cov = 0.0
                tau_cos_sim = 1.0

                _grad_reward_diag = dict(
                    prompt_norm_mean=_prompt_norm_mean,
                    prompt_norm_std=_prompt_norm_std,
                    prompt_norm_cov=_prompt_norm_cov,
                    tau_cov=tau_cov,
                    tau_cos_sim=tau_cos_sim,
                )
        else:
            cost_tau_log = torch.zeros(bsz, device=device)
            rewards_raw_log = torch.zeros(bsz, device=device)
            grad_cost = None
            grad_reward = None

        # ── Per-sample quantile clipping of ‖∇r‖ (as in VGG-Flow) ──
        if self.quantile_clip_pct > 0.0 and compute_control:
            rgn = grad_reward.flatten(1).norm(dim=-1)                 # per-sample ‖∇R‖
            thr = torch.quantile(rgn.detach(), self.quantile_clip_pct)
            scale = (thr / rgn.clamp(min=1e-8)).clamp(max=1.0)        # ≤1
            grad_reward = grad_reward * scale.view(-1, 1, 1, 1).detach()

        # ── Build g_hat + control loss (only when computing L_control) ─────
        if compute_control:
            base_flow_norm_batch = self._batch_mean_norm(v_teacher_t)
            reward_grad_norm_batch = self._batch_mean_norm(grad_reward)
            cost_grad_norm_batch = self._batch_mean_norm(grad_cost)
            B_ema = self._update_ema("ema_base_flow_norm", base_flow_norm_batch)
            G = self._update_ema("ema_reward_grad_norm", reward_grad_norm_batch)
            # ── Per-prompt control variate: subtract the group-mean reward gradient (div_baseline > 0).
            # reward_grad_norm_batch above uses the gradient before subtraction.
            _dg = getattr(self, "div_grep", 1); _db = getattr(self, "div_baseline", 0.0)
            if _db > 0.0 and _dg > 1:
                assert grad_reward.shape[0] % _dg == 0, (
                    f"batch {grad_reward.shape[0]} not divisible by div_grep {_dg}")
                _Bd = grad_reward.shape[0] // _dg
                _gr = grad_reward.view(_dg, _Bd, *grad_reward.shape[1:])
                grad_reward = (_gr - _db * _gr.mean(dim=0, keepdim=True)).reshape(grad_reward.shape)
            if self.w_rew_norm:
                # Normalized reward scale: w_rew = tcf · ‖v_base‖ / ‖∇R‖.
                w_rew = (tcf * B_ema / (G + 1e-8)).detach()
            else:
                # Unnormalized mode: tcf scales ∇R directly.
                w_rew = tcf.detach()
                # Optional linear ramp of the fixed reward force from 0 to tcf over
                # `fixed_force_ramp_iters` iterations (|grad R| is largest at initialization).
                _ramp_n = int(getattr(self, "fixed_force_ramp_iters", 0))
                if _ramp_n > 0:
                    _frac = min(1.0, float(self._ema_update_count) / float(_ramp_n))
                    w_rew = w_rew * _frac
            if self.w_rew_cap > 0:
                w_rew = w_rew.clamp(max=self.w_rew_cap)
            in_interval = ((t_val >= t_min) & (t_val <= t_max)).to(x_t.dtype)

            # Optional time attenuation η(t) ('none' in the reported recipe).
            if self.eta_mode == 'quad':
                eta = ((1.0 - t_val) ** 2).detach()
            elif self.eta_mode == 'linear':
                eta = (1.0 - t_val).detach()
            elif self.eta_mode == 'constant':
                eta = torch.ones_like(t_val)
            else:
                eta = torch.ones_like(t_val)

            # g_hat = -(η · w_rew · ∇r − ∇cost)  [WTF Eq 47/48]
            if grad_stretch is not None:
                stretch_grad_norm_batch = self._batch_mean_norm(grad_stretch)
                S_ema = self._update_ema("ema_stretch_grad_norm", stretch_grad_norm_batch)
                w_str_eff = (self.stretch_weight * (w_rew.mean() * G) / (S_ema + 1e-8)).detach()
                g_hat_inner = eta * (w_rew * grad_reward + w_str_eff * grad_stretch) - grad_cost
            else:
                g_hat_inner = eta * w_rew * grad_reward - grad_cost
            g_hat_raw = -torch.nan_to_num(g_hat_inner, nan=0.0)
            g_hat = (self.g_hat_scale * in_interval * g_hat_raw).detach()

            # Trainable diagonal control, built after the cost and reward graphs are released to save memory.
            self._sa("default")  # diagonal dynamics, adapter A
            v_student_t = self._cfg_call(model, x_t, t_val, t_val, y, y_null, cond)
            u_t = v_student_t - v_teacher_t

            # HJB ablation: add the fitted correction field to the policy target.
            _tgt = g_hat
            if bool(getattr(self, "hjb_nu_adapter", False)) and getattr(self, "set_adapter_fn", None) is not None:
                try:
                    self._sa('map')
                    _nu_pol = self._cfg_call(model, x_t.detach(), t_val, t_val, y, y_null, cond).detach()
                    self._sa('default')
                    _tgt = g_hat + _nu_pol.to(g_hat.dtype)
                except Exception as _e:
                    if not hasattr(self, "_nupol_warned"):
                        print(f"[nu-policy] DISABLED: {type(_e).__name__}: {_e}", flush=True)
                        self._nupol_warned = True
            loss_control = torch.mean((u_t - _tgt) ** 2)

            # ── HJB consistency ablation (off in the reported recipe): a consistency residual on the
            # correction ν = u - sg(g_hat) and a boundary term at t = 1, as in VGG-Flow (Eq. 12-13).
            if float(getattr(self, "hjb_consist_weight", 0.0)) > 0.0:
                try:
                    # Finite differences in fp32 with a bf16-safe step.
                    _eps = float(getattr(self, "hjb_eps_time", 1e-2))
                    _vb = v_teacher_t.detach()                 # v_base = frozen drift
                    def _u_of(_x, _tt):
        # fp32 for FD stability
                        _vs = self._cfg_call(model, _x, _tt, _tt, y, y_null, cond)
                        _vt = self._cfg_call(frozen_model, _x, _tt, _tt, y, y_null, cond).detach()
                        return _vs - _vt
                    _u0 = u_t                                   # already computed above
                    _t_p = (t_val + _eps).clamp(max=1.0)
                    _du_dt = (_u_of(x_t.detach(), _t_p) - _u0) / _eps          # d u / dt
                    _dx = x_t.detach() + _eps * _vb
                    _adv  = (_u_of(_dx, t_val) - _u0) / _eps                   # [grad u]^T v_base
                    # term 4: [grad v_base]^T u  (VJP of the frozen drift, plain autograd)
                    _xg = x_t.detach().requires_grad_(True)
                    _vb_g = self._cfg_call(frozen_model, _xg, t_val, t_val, y, y_null, cond)
                    _beta = float(self.g_hat_scale)
                    if bool(getattr(self, "hjb_nu_adapter", False)) and getattr(self, "set_adapter_fn", None) is not None:
                        # HJB ablation: nu is a separate learned field on adapter B.
                        self._sa('map')
                        _nu_raw = self._cfg_call(model, x_t.detach(), t_val, t_val, y, y_null, cond)
                        self._sa('default')
                        _nu = _nu_raw.float()
                    else:
                        _nu = _u0.float() - g_hat.detach().float()
                    _jt_nu = torch.autograd.grad(_vb_g, _xg, grad_outputs=_nu.to(_u0.dtype),
                                                 retain_graph=False, create_graph=False)[0].detach().float()
                    _resid = (_du_dt.float() + _adv.float()
                              - _beta * _nu + _jt_nu)
                    loss_hjb = torch.mean(_resid ** 2)
                    # ── Boundary term: pin u(x_1, 1) to the reward gradient ──
                    _bw = float(getattr(self, "hjb_boundary_weight", 0.0))
                    if _bw > 0.0:
                        _t1 = torch.ones_like(t_val)
                        _u1 = _u_of(x_t.detach(), _t1)
                        _tgt1 = g_hat.detach()
                        _loss_hjb_bnd = torch.mean((_u1.float() - _tgt1.float()) ** 2)
                        loss_hjb = loss_hjb + _bw * _loss_hjb_bnd
                except Exception as _e:
                    loss_hjb = torch.zeros((), device=device)
                    if not hasattr(self, "_hjb_warned"):
                        print(f"[hjb] DISABLED this step: {type(_e).__name__}: {_e}", flush=True)
                        self._hjb_warned = True
            else:
                loss_hjb = torch.zeros((), device=device)
        else:
            loss_control = torch.zeros((), device=device)
            base_flow_norm_batch = torch.tensor(0.0, device=device)
            reward_grad_norm_batch = torch.tensor(0.0, device=device)
            cost_grad_norm_batch = torch.tensor(0.0, device=device)
            w_rew = torch.zeros((bsz, 1, 1, 1), device=device)
            g_hat = torch.zeros((bsz,) + latent_shape, device=device)
            B_ema = torch.tensor(0.0, device=device)
            G = torch.tensor(0.0, device=device)

        # ── Mean-flow distillation (3 methods: JVP / FD / semigroup) ─────────
        # Skip when computing only L_control (asymmetric mode).
        if compute_distill:
            unwrapped_model = raw_model.module if hasattr(raw_model, "module") else raw_model

            # ── Common setup for JVP, FD, FD_traj: x_t_mf, v_tgt_mf via frozen_model
            # (Semigroup uses its own (s, u, t) sampling and computes I_s separately)
            if self.distill_method in ('jvp', 'fd', 'fd_traj'):
                with torch.no_grad():
                    self._sa("map")      # noise->t_mf is a JUMP (off-diagonal) -> B
                    x_t_mf = noise + (t_mf - t_ones) * self._cfg_call(
                        frozen_model, noise, t_ones, t_mf, y, y_null, cond, t_endpoint=True)
                    self._sa("default")  # TEACHER v(t_mf,t_mf) is the DIAGONAL -> A (stop-grad)
                    v_tgt_mf = self._cfg_call(frozen_model, x_t_mf.detach(), t_mf, t_mf, y, y_null, cond).detach()
                x_t_mf = x_t_mf.detach()

            # ── Method dispatch ─────────────────────────────────────────────
            if self.distill_method == 'jvp':
                # Eulerian form: F(x, t, r) = v_t - (t - r) dF/dt, with dF/dt from a JVP. The JVP needs fused
                # attention (TiMLossAdapter), since stock flash_attn_func has no forward-mode rule.
                from torch.nn.attention import sdpa_kernel, SDPBackend

                if self.cfg_scale <= 1.0:
                    model_fn = partial(unwrapped_model, y=y, **cond)
                    primals = (x_t_mf, t_mf, r_mf)
                    tangents = (v_tgt_mf, torch.ones_like(t_mf), torch.zeros_like(r_mf))
                    self._sa("map")  # distillation forward, adapter B
                    with sdpa_kernel([SDPBackend.MATH]):
                        v_mf, dv_dt_mf = torch.func.jvp(model_fn, primals, tangents)
                    self._sa("default")
                else:
                    x_t_mf2 = torch.cat([x_t_mf, x_t_mf], dim=0)
                    t_mf2 = torch.cat([t_mf, t_mf], dim=0)
                    r_mf2 = torch.cat([r_mf, r_mf], dim=0)
                    v_tgt_mf2 = torch.cat([v_tgt_mf, v_tgt_mf], dim=0)
                    y2 = torch.cat([y, y_null.to(y.dtype)], dim=0)
                    cond2 = {k: (torch.cat([v, v], dim=0) if torch.is_tensor(v) else v)
                             for k, v in cond.items()}
                    model_fn = partial(unwrapped_model, y=y2, **cond2)
                    primals = (x_t_mf2, t_mf2, r_mf2)
                    tangents = (v_tgt_mf2, torch.ones_like(t_mf2), torch.zeros_like(r_mf2))
                    self._sa("map")  # distillation forward, adapter B
                    with sdpa_kernel([SDPBackend.MATH]):
                        v_mf_all, dv_dt_mf_all = torch.func.jvp(model_fn, primals, tangents)
                    self._sa("default")
                    v_cond, v_uncond = v_mf_all.chunk(2, dim=0)
                    dv_cond, dv_uncond = dv_dt_mf_all.chunk(2, dim=0)
                    v_mf = v_uncond + self.cfg_scale * (v_cond - v_uncond)
                    dv_dt_mf = dv_uncond + self.cfg_scale * (dv_cond - dv_uncond)

                # Optional per-sample clip of the JVP tangent before the (r - t) dv/dt term.
                if self.jvp_tangent_clip_ratio > 0:
                    with torch.no_grad():
                        v_tgt_norm_ = v_tgt_mf.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1.0)
                        dv_norm_ = dv_dt_mf.flatten(1).norm(dim=-1, keepdim=True)
                        max_dv = self.jvp_tangent_clip_ratio * v_tgt_norm_
                        scale = torch.minimum(max_dv / (dv_norm_ + 1e-8), torch.ones_like(dv_norm_))
                    dv_dt_mf = dv_dt_mf * scale.view(-1, *([1] * (dv_dt_mf.ndim - 1)))

                # Optional soft tangent normalization: ‖dv‖ ← ‖dv‖ ‖v_tgt‖ / (eps ‖dv‖ + ‖v_tgt‖).
                if self.tangent_normalize_eps > 0:
                    with torch.no_grad():
                        v_tgt_norm_ = v_tgt_mf.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1.0)
                        dv_norm_    = dv_dt_mf.flatten(1).norm(dim=-1, keepdim=True)
                        scale       = v_tgt_norm_ / (self.tangent_normalize_eps * dv_norm_ + v_tgt_norm_)
                    dv_dt_mf = dv_dt_mf * scale.view(-1, *([1] * (dv_dt_mf.ndim - 1)))

                v_mf_tgt = (v_tgt_mf + (r_mf - t_mf) * dv_dt_mf).detach()
                mf_err = v_mf - v_mf_tgt
                mean_flow_loss = torch.sum(mf_err.flatten(1) ** 2, dim=-1)

            elif self.distill_method == 'fd':
                # Finite-difference directional derivative along (v_tgt, 1, 0) in (x, t, r):
                # F(x + δ v_tgt, t + δ, r) - F(x, t, r) = δ dF/dt + O(δ²), the quantity the JVP returns.
                eps = self.fd_eps
                t_mf_plus = (t_mf + eps).clamp(max=1.0)
                # δ is the actual time step (smaller near t = 1), used for both the x and t perturbations.
                delta_t = (t_mf_plus - t_mf).clamp(min=1e-6)
                v_mf = self._cfg_call(unwrapped_model, x_t_mf, t_mf, r_mf, y, y_null, cond)
                with torch.no_grad():
                    v_mf_plus = self._cfg_call(
                        unwrapped_model, x_t_mf + delta_t * v_tgt_mf, t_mf_plus, r_mf,
                        y, y_null, cond,
                    )
                dv_dt_mf = ((v_mf_plus - v_mf.detach()) / delta_t).detach()

                # Optional soft tangent normalization, as above.
                if self.tangent_normalize_eps > 0:
                    with torch.no_grad():
                        v_tgt_norm_ = v_tgt_mf.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1.0)
                        dv_norm_    = dv_dt_mf.flatten(1).norm(dim=-1, keepdim=True)
                        scale       = v_tgt_norm_ / (self.tangent_normalize_eps * dv_norm_ + v_tgt_norm_)
                    dv_dt_mf = dv_dt_mf * scale.view(-1, *([1] * (dv_dt_mf.ndim - 1)))

                v_mf_tgt = (v_tgt_mf + (r_mf - t_mf) * dv_dt_mf).detach()
                mf_err = v_mf - v_mf_tgt
                mean_flow_loss = torch.sum(mf_err.flatten(1) ** 2, dim=-1)

            elif self.distill_method == 'fd_traj':
                # Finite difference along the frozen trajectory: x at t + ε comes from a frozen flow-map jump
                # from the same noise, so dF/dt = ∂F/∂t + ∂F/∂x dx/dt.
                eps = self.fd_eps
                t_mf_plus = (t_mf + eps).clamp(max=1.0)
                v_mf = self._cfg_call(unwrapped_model, x_t_mf, t_mf, r_mf, y, y_null, cond)
                with torch.no_grad():
                    # Re-roll x at t + ε from the same noise with the frozen model.
                    v_anchor_plus = self._cfg_call(
                        frozen_model, noise, t_ones, t_mf_plus,
                        y, y_null, cond, t_endpoint=True,
                    )
                    x_t_mf_plus = noise + (t_mf_plus - t_ones) * v_anchor_plus
                    v_mf_plus = self._cfg_call(
                        unwrapped_model, x_t_mf_plus, t_mf_plus, r_mf, y, y_null, cond,
                    )
                dt_actual = (t_mf_plus - t_mf).clamp(min=1e-6)
                dv_dt_mf = ((v_mf_plus - v_mf.detach()) / dt_actual).detach()

                # Optional soft tangent normalization, as above.
                if self.tangent_normalize_eps > 0:
                    with torch.no_grad():
                        v_tgt_norm_ = v_tgt_mf.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1.0)
                        dv_norm_    = dv_dt_mf.flatten(1).norm(dim=-1, keepdim=True)
                        scale       = v_tgt_norm_ / (self.tangent_normalize_eps * dv_norm_ + v_tgt_norm_)
                    dv_dt_mf = dv_dt_mf * scale.view(-1, *([1] * (dv_dt_mf.ndim - 1)))

                v_mf_tgt = (v_tgt_mf + (r_mf - t_mf) * dv_dt_mf).detach()
                mf_err = v_mf - v_mf_tgt
                mean_flow_loss = torch.sum(mf_err.flatten(1) ** 2, dim=-1)

            elif self.distill_method == 'semigroup':
                # Compositional consistency (WTF Eq. 15): F(x, t, s) ≈ F(F(x, t, u), u, s) for s ≤ u ≤ t.
                # The model is called with t_start ≥ t_end (noise to data), as in TiM.
                bsz_sg = noise.shape[0]
                with torch.no_grad():
                    uvw = torch.sort(torch.rand(bsz_sg, 3, device=device), dim=-1).values
                    # Largest = start (noise side), smallest = end (data side).
                    s_d = uvw[:, 0:1].view(-1, 1, 1, 1)  # end (data side)
                    u_d = uvw[:, 1:2].view(-1, 1, 1, 1)  # mid
                    t_d = uvw[:, 2:3].view(-1, 1, 1, 1)  # start (noise side)
                    ones_d = torch.ones_like(t_d)
                    # I_t: rollout of the frozen model from noise to the largest time t.
                    noise_sg = torch.randn_like(noise)
                    v_anchor = self._cfg_call(frozen_model, noise_sg, ones_d, t_d, y, y_null, cond, t_endpoint=True)
                    I_t = (noise_sg + (t_d - ones_d) * v_anchor).detach()

                # Differentiable direct jump I_t → s.
                v_big = self._cfg_call(unwrapped_model, I_t, t_d, s_d, y, y_null, cond)
                X_big = I_t + (s_d - t_d) * v_big  # (s-t) < 0, displacement toward data

                # Two small jumps (stop-grad): I_t -> X_tu at u, then X_tu -> X_us at s
                with torch.no_grad():
                    v_tu = self._cfg_call(unwrapped_model, I_t, t_d, u_d, y, y_null, cond)
                    X_tu = I_t + (u_d - t_d) * v_tu  # (u-t) < 0
                    v_us = self._cfg_call(unwrapped_model, X_tu, u_d, s_d, y, y_null, cond)
                    X_us = X_tu + (s_d - u_d) * v_us  # (s-u) < 0

                sg_err = X_big - X_us.detach()
                mean_flow_loss = torch.sum(sg_err.flatten(1) ** 2, dim=-1)

                # Common tensors for the shared logging code.
                v_mf = X_big
                v_mf_tgt = X_us.detach()
                dv_dt_mf = v_big.detach()
                # Use (t_d, s_d) as (t_mf, r_mf) for the TiM time weighting.
                t_mf = t_d
                r_mf = s_d
                v_tgt_mf = X_us.detach()  # for diagnostic v_tgt_mf_norm logging

            elif self.distill_method == 'psd_shared':
                # Semigroup distillation reusing the rollout, split 50/50 per sample.
                # Anchor: F(x_t, t, 0) against sg(x_0_hat) from the reward path, with no extra forwards.
                # Interior: F(x_t, t, τ) against the chain F(x_t, t, u), F(x_u, u, τ), with u = (t + τ)/2.
                if x_t is None or _psd_x_0_hat_target is None:
                    raise RuntimeError(
                        "psd_shared distill requires rollout_steps=2 and "
                        "compute_control=True (needs x_τ + x_0_hat reused from "
                        "the reward path). Use another distill method otherwise.")

                bsz_sg = x_t.shape[0]
                t_d = t_val.view(-1, 1, 1, 1) if t_val.ndim == 1 else t_val
                t_d = t_d.detach()
                zero_d = torch.zeros_like(t_d)

                # Reuse the τ from the reward path.
                tau_d = _psd_tau_target.detach()

                # Per-sample bernoulli: 0.5 anchor / 0.5 interior.
                use_interior = (torch.rand(bsz_sg, device=device) < self.psd_shared_p_interior)
                ui_b = use_interior.view(-1, 1, 1, 1).to(t_d.dtype)

                end_d = ui_b * tau_d + (1 - ui_b) * zero_d
                mid_d_interior = (t_d + tau_d) / 2.0  # only used in interior case

                # Direct jump x_t → end with gradient (end = 0 for anchor, τ for interior).
                v_direct = self._cfg_call(unwrapped_model, x_t, t_d, end_d, y, y_null, cond)
                X_direct = x_t + (end_d - t_d) * v_direct

                # RHS: blend reused x_0_hat (anchor) with fresh chain (interior).
                with torch.no_grad():
                    # Interior chain, computed for the whole batch and masked for anchor samples.
                    v_to_mid = self._cfg_call(unwrapped_model, x_t, t_d, mid_d_interior, y, y_null, cond)
                    X_mid = x_t + (mid_d_interior - t_d) * v_to_mid
                    v_to_end_int = self._cfg_call(unwrapped_model, X_mid, mid_d_interior, tau_d, y, y_null, cond)
                    X_chain_interior = X_mid + (tau_d - mid_d_interior) * v_to_end_int

                # Per-sample target: anchor → x_0_hat (reused), interior → X_chain_interior.
                X_chain = ui_b * X_chain_interior + (1 - ui_b) * _psd_x_0_hat_target

                psd_err = X_direct - X_chain.detach()
                mean_flow_loss = torch.sum(psd_err.flatten(1) ** 2, dim=-1)

                # Common-interface tensors for downstream logging.
                v_mf = X_direct
                v_mf_tgt = X_chain.detach()
                dv_dt_mf = v_direct.detach()
                t_mf = t_d
                r_mf = end_d
                v_tgt_mf = X_chain.detach()

            # TiM distillation loss: add 1 - cos(F_pred, F_target) to the L2 term (TiM transition.py).
            if self.distill_mode == 'tim':
                v_mf_flat = v_mf.flatten(1)
                v_mf_tgt_flat = v_mf_tgt.flatten(1)
                cos_sim_mf = F.cosine_similarity(v_mf_flat, v_mf_tgt_flat, dim=-1)
                directional_loss = 1.0 - cos_sim_mf
                directional_loss = torch.nan_to_num(directional_loss, nan=0.0, posinf=1e5, neginf=-1e5)
                mean_flow_loss = mean_flow_loss + directional_loss

            mf_weights = 1.0 / (mean_flow_loss.detach() + 1e-3).pow(self.adaptive_p)

            # TiM time weighting 1/sqrt(t - r + σ_d) (weight_time_type='sqrt').
            if self.distill_mode == 'tim':
                _delta_t = (t_mf - r_mf).flatten()  # (B,), t_mf >= r_mf by construction
                _time_w = 1.0 / torch.sqrt(_delta_t + self.distill_time_sigma_d)
                mf_weights = mf_weights * _time_w.detach()

            # Optional long-jump weight (1 + α|t - r|), off at α = 0.
            if self.distill_longjump_alpha > 0:
                _delta_t_lj = (t_mf - r_mf).flatten().detach()
                mf_weights = mf_weights * (1.0 + self.distill_longjump_alpha * _delta_t_lj)

            # ── Dual-adapter ablation: regress B's diagonal onto A's detached diagonal so the two adapters
            # agree at r = t ──
            _dd_w = float(getattr(self, 'dual_diag_weight', 0.0))
            loss_diag_b = torch.zeros((), device=device)
            if (_dd_w > 0.0 and getattr(self, 'set_adapter_fn', None) is not None
                    and self.distill_method in ('jvp', 'fd', 'fd_traj')):
                self._sa('map')
                v_diag_b = self._cfg_call(model, x_t_mf, t_mf, t_mf, y, y_null, cond)
                self._sa('default')
                loss_diag_b = torch.mean((v_diag_b - v_tgt_mf) ** 2)
            loss_distillation = (mf_weights * mean_flow_loss).mean()
            # ── Diagnostics: JVP tangent magnitude ──
            with torch.no_grad():
                _dv_norm = dv_dt_mf.float().flatten(1).norm(dim=-1).mean().item()
                _v_tgt_norm = v_tgt_mf.float().flatten(1).norm(dim=-1).mean().item()
                _v_mf_norm = v_mf.float().flatten(1).norm(dim=-1).mean().item()
                _ratio = _dv_norm / max(_v_tgt_norm, 1e-9)
        else:
            loss_distillation = torch.zeros((), device=device)
            loss_diag_b = torch.zeros((), device=device)
            _dd_w = 0.0
            mean_flow_loss = torch.zeros(bsz, device=device)

        # Final loss: each component is zero when its compute_* flag is False.
        _dd_w_g = _dd_w; loss_diag_b_g = loss_diag_b
        _hjb_w = float(getattr(self, 'hjb_consist_weight', 0.0))
        if _hjb_w > 0.0:
            # Adaptive HJB weight: normalize by an EMA of the residual's magnitude so the term is a fixed
            # fraction of the control loss.
            _h = loss_hjb.detach()
            _prev = getattr(self, "_hjb_ema", None)
            _hjb_ema = _h if _prev is None else (0.98 * _prev + 0.02 * _h)
            self._hjb_ema = _hjb_ema.detach()
            _scale = (loss_control.detach() / (_hjb_ema + 1e-8)).clamp(max=1e4)
            _hjb_term = _hjb_w * _scale * loss_hjb
        else:
            _hjb_term = torch.zeros((), device=device)
        loss = self.beta_outer * loss_control + self.distill_weight * loss_distillation + _dd_w_g * loss_diag_b_g + _hjb_term

        # ── Diagnostics ────────────────────────────────────────────────────
        with torch.no_grad():
            if compute_control:
                g_hat_flat = g_hat.flatten(1)
                u_t_flat = u_t.flatten(1)
                g_hat_norm = g_hat_flat.norm(dim=-1).mean()
                u_t_norm = u_t_flat.norm(dim=-1).mean()
                cos_sim_u_ghat = F.cosine_similarity(u_t_flat, g_hat_flat, dim=-1).mean()
                x_t_norm = x_t.flatten(1).norm(dim=-1).mean()
            else:
                g_hat_norm = torch.tensor(0.0, device=device)
                u_t_norm = torch.tensor(0.0, device=device)
                cos_sim_u_ghat = torch.tensor(0.0, device=device)
                x_t_norm = torch.tensor(0.0, device=device)
            rgrad_mean = reward_grad_norm_batch
            effective_beta = w_rew.mean().detach()

        loss_dict = {

            'loss_hjb': float(loss_hjb.detach()),
            # ── Loss components ───────────────────────────────────────────────
            "control_loss": loss_control.detach(),  # control regression term
            "mean_flow_loss": mean_flow_loss.mean().detach(),
            "loss_diag_b": loss_diag_b_g.detach(),
            "loss_distillation_adaptive": loss_distillation.detach(),
            "dv_dt_mf_norm": torch.tensor(_dv_norm if compute_distill else 0.0, device=device),
            "v_tgt_mf_norm": torch.tensor(_v_tgt_norm if compute_distill else 0.0, device=device),
            "v_mf_norm": torch.tensor(_v_mf_norm if compute_distill else 0.0, device=device),
            "dv_over_v_tgt": torch.tensor(_ratio if compute_distill else 0.0, device=device),
            # ── Reward signal ────────────────────────────────────────────────
            "rewards": rewards_raw_log.mean().detach(),
            "cost_tau": cost_tau_log.mean().detach(),
            # ── Training statistics ──
            "u_t_norm": u_t_norm,  # ‖u_t‖
            "g_hat_norm": g_hat_norm,  # ‖g_hat‖
            "cos_sim_u_ghat": cos_sim_u_ghat,
            "rgrad_mean": rgrad_mean,  # ‖∇R‖
            "effective_beta": effective_beta,
            "x_t_norm": x_t_norm,
            # ── Normalization statistics ──
            "base_flow_norm_batch": base_flow_norm_batch,  # ‖v_teacher‖, used to normalize w_rew
            "reward_grad_norm_batch": reward_grad_norm_batch,  # same as rgrad_mean
            "cost_grad_norm_batch": cost_grad_norm_batch,  # ‖∇cost‖
            "ema_base_flow_norm": B_ema,                          # EMA tracker
            "ema_reward_grad_norm": G,                            # EMA tracker
            "w_rew": w_rew.mean().detach(),  # same as effective_beta
            # ── Variance attribution (≥0 only when compute_control) ──────────
            "prompt_norm_cov": torch.tensor(
                _grad_reward_diag["prompt_norm_cov"] if compute_control else 0.0, device=device
            ),
            "tau_cov": torch.tensor(
                _grad_reward_diag["tau_cov"] if compute_control else 0.0, device=device
            ),
            "tau_cos_sim": torch.tensor(
                _grad_reward_diag["tau_cos_sim"] if compute_control else 1.0, device=device
            ),
        }
        return loss, loss_dict


# =============================================================================
# FTFMOCImFLoss, iMF-style OC loss for TiM-T2I
# =============================================================================
class FTFMOCImFLoss:
    """
    Optimal-control variant with iMF-style mean-flow distillation (`--use-imf-oc`). It uses the TiM
    model interface with CFG (`_cfg_call`) and rollout_steps in {1, 3} for the reward path.

    The distillation target comes from a frozen flow-map lookahead:

        e_imf       ~ N(0, I)
        x_target    = e_imf + (0-1)·F_frozen(e_imf, t=1, r=0)  # full rollout
        z_t         = (1-t_mf)·x_target + t_mf·e_imf           # linear interpolant
        v_target    = (e_imf − x_target).detach()              # FM regression target

        v_diag      = u_θ(z_t, t_mf, t_mf)                     # boundary cond
        (u, du_dt)  = jvp(model_fn, (z_t, t_mf, r_mf), (v_diag, 1, 0))
        V_θ         = u + (t_mf - r_mf)·du_dt.detach()         # iMF Eq 12, stop-grad
        L_iMF       = ‖V_θ - v_target‖²

    Total loss:
        L = L_reg(teacher anchor) − reward_scale·rewards + λ_imf · L_iMF
    """

    def __init__(
        self,
        # Time sampling (mirror FTFMVALUEDMFLoss signature)
        P_mean_t:        float = -0.2,
        P_std_t:         float = 1.0,
        P_mean_r:        float = 0.2,
        P_std_r:         float = 1.0,
        path_type:       str   = "linear",
        # Substrate
        cfg_scale:       float = 1.0,
        cfg_detach_uncond: bool = False,
        # Loss weights
        reward_scale:    float = 0.1,        # weight on -reward term
        lambda_imf:      float = 1.0,        # weight on L_iMF
        anchor_weight:   float = 0.5,        # weight on teacher-anchor regularization
        diag_prob:       float = 0.5,        # P(use diagonal r_reg=t_mf for anchor)
        # iMF flags
        imf_jvp_stopgrad: bool = True,       # iMF default, stop-grad on JVP outcome
        adaptive_imf:    bool  = False,      # apply adaptive 1/loss to L_iMF
        tangent_normalize_eps: float = 0.0,  # AYF-style soft cap on du_dt (0 = off)
        # ── Reward-gradient normalization (as w_rew in FTFMVALUEDMFLoss): EMA-normalize ∂r/∂x_0 and
        # rescale to target_control_fraction × ‖v_diag‖ ──
        w_rew_norm:               bool  = False,
        # 1.0 = reward force at ~‖v_diag‖. Higher = stronger reward pull.
        target_control_fraction:  float = 1.0,
        ema_decay:                float = 0.98,
        w_rew_cap:                float = 0.0,  # 0 = uncapped; >0 hard-caps w_rew
        # OC rollout
        rollout_steps:   int   = 3,  # 1 = noise→x0, 3 = noise→xt→xr→x0
        # Ablation: JVP tangent from original MeanFlow (Eq 9 of the iMF paper, tangent e - x) instead of
        # iMF (Eq 12, tangent u_θ(z, t, t)). False uses iMF.
        use_original_mf: bool  = False,
        **kwargs,
    ):
        self.P_mean_t = P_mean_t
        self.P_std_t  = P_std_t
        self.P_mean_r = P_mean_r
        self.P_std_r  = P_std_r
        self.path_type = path_type
        self.cfg_scale = float(cfg_scale)
        self.cfg_detach_uncond = bool(cfg_detach_uncond)
        self.reward_scale  = float(reward_scale)
        self.lambda_imf    = float(lambda_imf)
        self.anchor_weight = float(anchor_weight)
        self.diag_prob     = float(diag_prob)
        self.imf_jvp_stopgrad = bool(imf_jvp_stopgrad)
        self.adaptive_imf  = bool(adaptive_imf)
        self.tangent_normalize_eps = float(tangent_normalize_eps)
        self.w_rew_norm = bool(w_rew_norm)
        self.target_control_fraction = float(target_control_fraction)
        self.ema_decay = float(ema_decay)
        self.w_rew_cap = float(w_rew_cap)
        # EMA state, initialized lazily on first call.
        self._G_ema = None  # ‖∂r/∂x_0‖ EMA
        self._B_ema = None  # ‖v_diag‖   EMA
        assert rollout_steps in (1, 2, 3), f"rollout_steps must be 1, 2, or 3, got {rollout_steps}"
        self.rollout_steps = int(rollout_steps)
        self.use_original_mf = bool(use_original_mf)

    # Mirrors FTFMVALUEDMFLoss._cfg_call
    def _cfg_call(self, model_fn, x, t, r, y_cond, y_null, cond, *, t_endpoint: bool = False):
        if self.cfg_scale <= 1.0 or t_endpoint:
            return model_fn(x, t, r, y_cond, **cond)
        x2 = torch.cat([x, x], dim=0)
        t2 = torch.cat([t, t], dim=0)
        r2 = torch.cat([r, r], dim=0)
        y2 = torch.cat([y_cond, y_null.to(y_cond.dtype)], dim=0)
        cond2 = {k: (torch.cat([v, v], dim=0) if torch.is_tensor(v) else v)
                 for k, v in cond.items()}
        v_all = model_fn(x2, t2, r2, y2, **cond2)
        v_cond, v_uncond = v_all.chunk(2, dim=0)
        if self.cfg_detach_uncond:
            v_uncond = v_uncond.detach()
        return v_uncond + self.cfg_scale * (v_cond - v_uncond)

    def sample_times(self, bsz: int):
        ln_1 = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std_t + self.P_mean_t, self.path_type)
        ln_2 = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std_r + self.P_mean_r, self.path_type)
        t_mf, r_mf = torch.maximum(ln_1, ln_2), torch.minimum(ln_1, ln_2)
        return t_mf, r_mf

    def __call__(
        self,
        model,
        raw_model,
        y_text_feats,
        prompts,
        *,
        model_t,                 # frozen pretrained teacher (never updated)
        frozen_model,            # frozen anchor, EMA copy or model_t
        reward_fn,
        decode_func,
        latent_shape,
        y_null,
    ):
        device = y_text_feats.device
        bsz = y_text_feats.shape[0]
        y = y_text_feats
        if y_null.shape[0] == 1:
            y_null = y_null.expand(bsz, -1, -1)
        elif y_null.shape[0] != bsz:
            y_null = y_null[:bsz]

        t_mf, r_mf = self.sample_times(bsz)
        t_mf, r_mf = t_mf.to(device), r_mf.to(device)

        # Token conditioning is off, so cond uses scalar defaults.
        cond = dict(tcf=0.0, t_min=0.0, t_max=1.0)
        teacher_cond = dict(tcf=0.0, t_min=0.0, t_max=1.0)

        C, H, W = latent_shape
        noise = torch.randn(bsz, C, H, W, device=device, dtype=y_text_feats.dtype)
        t_ones  = torch.ones((bsz, 1, 1, 1), device=device)
        t_zeros = torch.zeros((bsz, 1, 1, 1), device=device)

        # ── Reward rollout ───────────────────────────────────────────────────
        # Rollout times (t = 1 noise, t = 0 data): rollout_steps 1 is 1 → 0, 2 is 1 → t_mf → 0,
        # 3 is 1 → t_mf → r_mf → 0. The reward gradient flows through every model call.
        if self.rollout_steps == 3:
            xt_est = noise + (t_mf - t_ones) * self._cfg_call(
                model, noise, t_ones, t_mf, y, y_null, cond, t_endpoint=True)
            xr_est = xt_est + (r_mf - t_mf) * self._cfg_call(
                model, xt_est, t_mf, r_mf, y, y_null, cond)
            x0_est = xr_est + (t_zeros - r_mf) * self._cfg_call(
                model, xr_est, r_mf, t_zeros, y, y_null, cond)
        elif self.rollout_steps == 2:
            # 1 → t_mf → 0
            xt_est = noise + (t_mf - t_ones) * self._cfg_call(
                model, noise, t_ones, t_mf, y, y_null, cond, t_endpoint=True)
            x0_est = xt_est + (t_zeros - t_mf) * self._cfg_call(
                model, xt_est, t_mf, t_zeros, y, y_null, cond)
        else:  # rollout_steps == 1
            x0_est = noise + (t_zeros - t_ones) * self._cfg_call(
                model, noise, t_ones, t_zeros, y, y_null, cond, t_endpoint=True)
            xt_est = None  # not used in 1-step

        # ── Reward ───────────────────────────────────────────────────────────
        x0_decoded = (decode_func(x0_est) + 1.0) / 2.0
        rewards = reward_fn.score(x0_decoded, prompts).float()
        rewards_raw = rewards.clone()

        # ── Reward-gradient normalization: surrogate whose gradient on x_0 is -w_rew ∂r/∂x_0 ──
        if self.w_rew_norm and rewards.requires_grad:
            # Latent reward gradient, keeping the graph for the surrogate.
            with torch.enable_grad():
                grad_r = torch.autograd.grad(
                    rewards.sum(), x0_est, retain_graph=True, create_graph=False
                )[0]
            g_norm = grad_r.flatten(1).norm(dim=-1).mean().detach()
            # EMA update for ‖∂r/∂x_0‖, lazy init.
            self._G_ema = g_norm if self._G_ema is None else (
                self.ema_decay * self._G_ema + (1 - self.ema_decay) * g_norm)
            self._G_ema = self._G_ema.detach()
            # B_ema is updated after v_diag below. On the first step, use the current gradient scale.
            if self._B_ema is None:
                w_rew = torch.tensor(1.0, device=device)
            else:
                w_rew = self.target_control_fraction * self._B_ema / (self._G_ema + 1e-6)
            if self.w_rew_cap > 0:
                w_rew = w_rew.clamp(max=self.w_rew_cap)
            # Surrogate reward term: gradient on x_0 = -w_rew · ∂r/∂x_0.
            # Equivalent loss = -w_rew · ⟨∂r/∂x_0 (detached), x_0⟩ summed per sample.
            reward_term_loss = -(w_rew.detach() * grad_r.detach() * x0_est)
            reward_term_loss = reward_term_loss.flatten(1).sum(dim=-1).mean()
            # The surrogate replaces the explicit -reward_scale · rewards term.
            use_surrogate_reward = True
        else:
            w_rew = torch.tensor(1.0, device=device)
            reward_term_loss = -rewards.mean()  # fallback uses raw rewards
            use_surrogate_reward = False

        # ── Teacher-anchor regularization at the rollout's intermediate state ──
        anchor_state = xt_est if (xt_est is not None) else noise
        anchor_t = t_mf if (xt_est is not None) else t_ones
        diag_mask = (torch.rand(bsz, device=device) < self.diag_prob).view(-1, 1, 1, 1)
        r_reg = torch.where(diag_mask, anchor_t, r_mf)
        v_student_reg = self._cfg_call(model, anchor_state, anchor_t, r_reg, y, y_null, cond)
        with torch.no_grad():
            v_teacher_reg = self._cfg_call(model_t, anchor_state, anchor_t, r_reg, y, y_null, teacher_cond)
        reg_err = v_student_reg - v_teacher_reg.detach()
        # Mean over (C, H, W) per sample.
        reg_loss = torch.mean(reg_err ** 2, dim=list(range(1, len(reg_err.shape)))) * 0.5

        # ── iMF mean-flow loss, using the reward rollout's endpoint x0_est as the target ──
        x_target = x0_est.detach()
        # Linear interpolant at t_mf along the (noise, x_target) line.
        # Convention: t=1 noise (z_1 = noise), t=0 data (z_0 = x_target).
        z_t = ((t_ones - t_mf) * x_target + t_mf * noise).detach()
        v_target = (noise - x_target).detach()    # FM-style regression target

        # Boundary condition: v_θ(z, t) ≡ u_θ(z, t, t), same network at t=r.
        unwrapped_model = raw_model.module if hasattr(raw_model, "module") else raw_model

        with torch.no_grad():
            v_diag = self._cfg_call(unwrapped_model, z_t, t_mf, t_mf, y, y_null, cond)

        # Update B_ema (‖v_diag‖) for w_rew normalization on next iter.
        if self.w_rew_norm:
            v_diag_norm_b = v_diag.flatten(1).norm(dim=-1).mean().detach()
            self._B_ema = v_diag_norm_b if self._B_ema is None else (
                self.ema_decay * self._B_ema + (1 - self.ema_decay) * v_diag_norm_b)
            self._B_ema = self._B_ema.detach()

        # iMF JVP through attention with the SDPA math backend.
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from functools import partial

        # JVP tangent: v_diag for iMF (Eq 12), v_target = e - x for original MeanFlow (Eq 9).
        state_tangent = v_target if self.use_original_mf else v_diag

        if self.cfg_scale <= 1.0:
            model_fn = partial(unwrapped_model, y=y, **cond)
            primals  = (z_t, t_mf, r_mf)
            tangents = (state_tangent, torch.ones_like(t_mf), torch.zeros_like(r_mf))
            with sdpa_kernel([SDPBackend.MATH]):
                u_jvp, du_dt = torch.func.jvp(model_fn, primals, tangents)
        else:
            # CFG path: duplicate batch [cond; uncond], JVP, then blend.
            z_t2 = torch.cat([z_t, z_t], dim=0)
            t_mf2 = torch.cat([t_mf, t_mf], dim=0)
            r_mf2 = torch.cat([r_mf, r_mf], dim=0)
            state_tangent2 = torch.cat([state_tangent, state_tangent], dim=0)
            y2 = torch.cat([y, y_null.to(y.dtype)], dim=0)
            cond2 = {k: (torch.cat([v, v], dim=0) if torch.is_tensor(v) else v)
                     for k, v in cond.items()}
            model_fn = partial(unwrapped_model, y=y2, **cond2)
            primals  = (z_t2, t_mf2, r_mf2)
            tangents = (state_tangent2, torch.ones_like(t_mf2), torch.zeros_like(r_mf2))
            with sdpa_kernel([SDPBackend.MATH]):
                u_all, du_all = torch.func.jvp(model_fn, primals, tangents)
            u_cond, u_uncond = u_all.chunk(2, dim=0)
            du_cond, du_uncond = du_all.chunk(2, dim=0)
            u_jvp = u_uncond + self.cfg_scale * (u_cond - u_uncond)
            du_dt = du_uncond + self.cfg_scale * (du_cond - du_uncond)

        # Optional soft tangent normalization: ‖dv‖ ← ‖dv‖ ‖v_diag‖ / (eps ‖dv‖ + ‖v_diag‖).
        if self.tangent_normalize_eps > 0:
            with torch.no_grad():
                v_diag_norm_ = v_diag.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1.0)
                du_norm_     = du_dt.flatten(1).norm(dim=-1, keepdim=True)
                scale = v_diag_norm_ / (self.tangent_normalize_eps * du_norm_ + v_diag_norm_)
            du_dt = du_dt * scale.view(-1, *([1] * (du_dt.ndim - 1)))

        # iMF Eq 12: V_θ = u + (t - r) · sg(JVP), as in iMF §4.1.
        if self.imf_jvp_stopgrad:
            V_theta = u_jvp + (t_mf - r_mf) * du_dt.detach()
        else:
            V_theta = u_jvp + (t_mf - r_mf) * du_dt

        imf_err = V_theta - v_target
        # MEAN over (C, H, W) per sample.
        imf_loss = torch.mean(imf_err ** 2, dim=list(range(1, len(imf_err.shape))))
        imf_loss_for_log = imf_loss.detach().mean()

        if self.adaptive_imf:
            mf_weights = 1.0 / (imf_loss.detach() + 1e-3)
            imf_loss_eff = (mf_weights * imf_loss).mean()
        else:
            imf_loss_eff = imf_loss.mean()

        # With w_rew_norm, reward_term_loss is the surrogate. Otherwise it is -rewards.mean().
        loss = (self.anchor_weight * reg_loss.mean()
                + self.reward_scale * reward_term_loss
                + self.lambda_imf * imf_loss_eff)

        loss_dict = {
            "loss":            loss.detach(),
            "reg_loss":        reg_loss.mean().detach(),
            "rewards":         rewards_raw.mean().detach(),
            "imf_loss":        imf_loss_for_log,
            "imf_loss_eff":    imf_loss_eff.detach(),
            "v_target_norm":   v_target.flatten(1).norm(dim=-1).mean().detach(),
            "v_diag_norm":     v_diag.flatten(1).norm(dim=-1).mean().detach(),
            "du_dt_norm":      du_dt.flatten(1).norm(dim=-1).mean().detach(),
            "V_theta_norm":    V_theta.flatten(1).norm(dim=-1).mean().detach(),
            "w_rew":           w_rew.detach() if torch.is_tensor(w_rew) else torch.tensor(w_rew, device=device),
            "G_ema":           self._G_ema.detach() if self._G_ema is not None else torch.tensor(0.0, device=device),
            "B_ema":           self._B_ema.detach() if self._B_ema is not None else torch.tensor(0.0, device=device),
        }
        return loss, loss_dict
