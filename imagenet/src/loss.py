import math
from functools import partial

import torch
import numpy as np
from timm.data import ImageNetInfo


def interpolant(x, t, path_type="linear"):
    eps = torch.randn_like(x)
    if path_type == "linear":
        alpha_t = 1 - t
        sigma_t = t
        d_alpha_t = -1
        d_sigma_t =  1
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = (-np.pi / 2) * torch.sin(t * np.pi / 2)
        d_sigma_t = (np.pi / 2) * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError()
    x_t = alpha_t * x + sigma_t * eps
    v_t = d_alpha_t * x + d_sigma_t * eps
    return x_t, v_t


def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def log_lv_loss(x, y, lv, eps=1.0):
    err = ((x - y) ** 2)
    mse_loss = torch.sum(err, dim=list(range(1, len(x.shape))))
    mean_loss = torch.mean(err, dim=list(range(1, len(x.shape))))
    log_loss = torch.log((1 / lv.exp()) * mean_loss + eps) + 0.5 * lv
    return mse_loss, log_loss


def mse_loss(x, y):
    err = (x - y)**2
    mse_loss = torch.sum(err, dim=list(range(1, len(x.size()))))
    loss = torch.mean(err, dim=list(range(1, len(x.size()))))
    return mse_loss, loss

def l2_loss(x, y):
    err = (x - y)**2
    loss = torch.sum(err, dim=list(range(1, len(x.size()))))
    l2_loss = torch.sum(err, dim=list(range(1, len(x.size()))))
    return l2_loss, loss

def adaptive_loss(x, y, p=1.0, c=0.001):
    err = (x - y)**2
    l2_loss = torch.sum(err, dim=list(range(1, len(x.size()))))
    norm_calc = torch.sum(err, dim=list(range(1, len(x.size()))))
    weights = 1.0 / (norm_calc + c) ** p
    weight = weights.detach()
    loss = l2_loss * weight
    return l2_loss, loss

def adaptive(loss, p=1.0, c=0.001):
    weights = 1.0 / (loss + c) ** p
    weight = weights.detach()
    loss = loss * weight
    return loss

def repa_loss(zs, zs_tilde):
    proj_loss = 0.
    bsz = zs[0].shape[0]
    for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
        for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
            z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
            z_j = torch.nn.functional.normalize(z_j, dim=-1) 
            proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
    proj_loss /= (len(zs) * bsz)
    return proj_loss


def compute_guidance(model, x, t, y, v_t, drop_ids, omega=0.5, g_min=0.0, g_max=1.0, g_type="default"):
    if g_type == "default":
        return v_t
    elif g_type == "mg":
        drop_ids = drop_ids.reshape(-1, 1, 1, 1)
        cfg_ids = (t >= g_min) & (t <= g_max) & (~drop_ids)
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
        with torch.no_grad():
            x_2 = torch.cat([x, x], dim=0)
            t_2 = torch.cat([t, t], dim=0)
            y_2 = torch.cat([y, y_null], dim=0)
            v = model(x_2, t_2, t_2, y_2).to(torch.float32)
            v_cond, v_uncond = v.chunk(2, dim=0)
        v_tgt = v_t + omega * (v_cond - v_uncond)
        v_tgt = torch.where(cfg_ids, v_tgt, v_t)
        return v_tgt
    elif g_type == "distill":
        cfg_ids = (t >= g_min) & (t <= g_max)
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
        with torch.no_grad():
            x_2 = torch.cat([x, x], dim=0)
            t_2 = torch.cat([t, t], dim=0)
            y_2 = torch.cat([y, y_null], dim=0)
            v = model(x_2, t_2, y_2).to(torch.float32)
            v_cond, v_uncond = v.chunk(2, dim=0)
        v_tgt = v_uncond + omega * (v_cond - v_uncond)
        v_tgt = torch.where(cfg_ids, v_tgt, v_cond)  # no class dropout for distillation
        return v_tgt
    else:
        raise NotImplementedError(f"Unknown guidance type: {g_type}")


def ln_sampler(z, path_type="linear"):
    if path_type == "linear":
        return torch.sigmoid(z)
    elif path_type == "cosine":
        return 2 / np.pi * torch.atan(z)
    else:
        raise NotImplementedError(f"Unknown path_type: {path_type}")


class FTFMLoss:
    def __init__(
        self,
        P_mean:         float = -0.4,
        P_std:          float = 1.0,
        P_mean_t:       float = -0.2,
        P_std_t:        float = 1.0,
        P_mean_r:       float = 0.2,
        P_std_r:        float = 1.0,
        cfg_prob:       float = 0.1,
        omega:          float = 0.0,
        g_min:          float = 0.0,
        g_max:          float = 1.0,
        path_type:      str   = "linear",
        g_type:         str   = "default",
        diag_prob:      float = 0.5,
        **kwargs,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.P_mean_t = P_mean_t
        self.P_std_t = P_std_t
        self.P_mean_r = P_mean_r
        self.P_std_r = P_std_r
        self.cfg_prob = cfg_prob
        self.path_type = path_type
        self.diag_prob = diag_prob
        self.guidance_kwargs = dict(
            omega=omega,
            g_min=g_min,
            g_max=g_max,
            g_type=g_type,
        )

    def cond_drop(self, y):
        bsz = y.shape[0]
        drop_ids = torch.rand(bsz, device=y.device) < self.cfg_prob
        y = torch.where(drop_ids, 1000, y)
        return y, drop_ids

    def sample_times(self, bsz):
        t_fm = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std + self.P_mean, self.path_type)
        ln_1 = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std_t + self.P_mean_t, self.path_type)
        ln_2 = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std_r + self.P_mean_r, self.path_type)
        t_mf, r_mf = torch.maximum(ln_1, ln_2), torch.minimum(ln_1, ln_2)
        return t_fm, t_mf, r_mf

    def __call__(self, model, raw_model, x, y, model_t=None, reward_fn=None, decode_func=None, **kwargs):

        assert model_t is not None, "model_t must be provided for FTFMLoss"
        y, drop_ids = y, None
        
        t_fm, t_mf, r_mf = self.sample_times(x.shape[0])
        t_fm, t_mf, r_mf = t_fm.to(x.device), t_mf.to(x.device), r_mf.to(x.device)

        # No token conditioning.
        cond_kwargs = dict(tcf=0.0, t_min=0.0, t_max=1.0)

        class_description = ImageNetInfo()
        base_prompt = "A high-resolution, high-quality photograph of a "
        prompts = []
        for i in range(y.shape[0]):
            class_id = y[i].item()
            assert class_id < 1000, "no class dropout for reward computation"
            class_name = class_description.index_to_description(class_id)
            prompt = base_prompt + class_name
            prompts.append(prompt)

        noise = torch.randn_like(x)

        t_ones = torch.ones((x.shape[0], 1, 1, 1), device=x.device)
        t_zeros = torch.zeros((x.shape[0], 1, 1, 1), device=x.device)

        # Three-stage rollout
        xt_estimate = noise + (t_mf - t_ones) * model(noise, t_ones, t_mf, y, **cond_kwargs)
        xr_estimate = xt_estimate + (r_mf - t_mf) * model(xt_estimate, t_mf, r_mf, y, **cond_kwargs)
        x0_estimate = xr_estimate + (t_zeros - r_mf) * model(xr_estimate, r_mf, t_zeros, y, **cond_kwargs)

        x0_estimate = (decode_func(x0_estimate) + 1.0) / 2.0

        rewards = reward_fn.score(x0_estimate, prompts)
        rewards_raw = rewards.clone()
        assert rewards.shape == (x.shape[0],), f"Expected rewards shape {(x.shape[0],)}, but got {rewards.shape}"

        # Regularization: per-sample mix of diagonal (t,t) and off-diagonal (t,r)
        diag_mask = (torch.rand(x.shape[0], device=x.device) < self.diag_prob).view(-1, 1, 1, 1)
        r_reg = torch.where(diag_mask, t_mf, r_mf)
        model_reg = model(xt_estimate, t_mf, r_reg, y, **cond_kwargs) - model_t(xt_estimate, t_mf, r_reg, y, **cond_kwargs)
        reg_loss_mean = torch.mean(model_reg ** 2, dim=list(range(1, len(model_reg.shape)))) * 0.5


        # MeanFlow Loss on data-interpolant states
        x_t_mf, v_t_mf = interpolant(x, t_mf, self.path_type)
        v_tgt_mf = v_t_mf.detach()

        unwrapped_model = raw_model.module if hasattr(raw_model, "module") else raw_model
        model_fn = partial(unwrapped_model, y=y, return_logvar=False, **cond_kwargs)
        primals = (x_t_mf, t_mf, r_mf)
        tangents = (v_tgt_mf, torch.ones_like(t_mf), torch.zeros_like(r_mf))
        u, du_dt = torch.func.jvp(model_fn, primals, tangents)
        u_tgt = v_tgt_mf + (r_mf - t_mf) * du_dt
        u_tgt = u_tgt.detach()
        _, mean_flow_loss = mse_loss(u, u_tgt)

        loss_distillation_adaptive = adaptive(mean_flow_loss)

        lamda = 1.0
        reward_scale = 0.1
        loss = reg_loss_mean - reward_scale * rewards + loss_distillation_adaptive * lamda
        loss_dict = {
            "reg_loss": reg_loss_mean,
            "rewards": rewards_raw,
            "loss_distillation": loss_distillation_adaptive,
            "mean_flow_loss": mean_flow_loss,
        }
        return loss, loss_dict



class FTFMOCImFLoss:
    """
    Optimal-control reward fine-tuning with iMF-style mean-flow distillation, using a frozen
    flow-map lookahead as the regression target.

    iMF (Geng et al. 2025, arXiv:2512.02012) trains the instantaneous velocity through the
    boundary condition v_θ(z, t) = u_θ(z, t, t):
        V_θ(z_t, t, r) = u_θ(z_t, t, r) + (t - r) · sg(JVP(u_θ; v_θ, 0, 1))
        L_iMF = ‖V_θ - (e - x)‖²
    Without data, x is the frozen model's rollout from noise:
        x_target = e + F_frozen(e, t=1, r=0),  z_t = (1 - t) x_target + t e
    Total loss: L = L_reg - reward_scale · rewards + λ_imf · L_iMF
    """

    def __init__(
        self,
        P_mean:           float = -0.4,
        P_std:            float = 1.0,
        P_mean_t:         float = -0.2,
        P_std_t:          float = 1.0,
        P_mean_r:         float = 0.2,
        P_std_r:          float = 1.0,
        cfg_prob:         float = 0.1,
        omega:            float = 0.0,
        g_min:            float = 0.0,
        g_max:            float = 1.0,
        path_type:        str   = "linear",
        g_type:           str   = "default",
        diag_prob:        float = 0.5,
        reward_scale:     float = 0.1,    # weight on reward term (matches FTFMLoss default)
        lambda_imf:       float = 1.0,    # weight on the iMF mean-flow loss
        imf_jvp_stopgrad: bool  = True,   # iMF default, stop-grad on JVP outcome
        adaptive_imf:     bool  = False,  # apply adaptive 1/loss to L_iMF (off by default)
        **kwargs,
    ):
        self.P_mean         = P_mean
        self.P_std          = P_std
        self.P_mean_t       = P_mean_t
        self.P_std_t        = P_std_t
        self.P_mean_r       = P_mean_r
        self.P_std_r        = P_std_r
        self.cfg_prob       = cfg_prob
        self.path_type      = path_type
        self.diag_prob      = diag_prob
        self.reward_scale   = reward_scale
        self.lambda_imf     = lambda_imf
        self.imf_jvp_stopgrad = imf_jvp_stopgrad
        self.adaptive_imf   = adaptive_imf
        self.guidance_kwargs = dict(omega=omega, g_min=g_min, g_max=g_max, g_type=g_type)

    def cond_drop(self, y):
        bsz = y.shape[0]
        drop_ids = torch.rand(bsz, device=y.device) < self.cfg_prob
        y = torch.where(drop_ids, 1000, y)
        return y, drop_ids

    def sample_times(self, bsz):
        # t_fm is unused here.
        # t_mf > r_mf, used for both the reward 3-stage rollout AND the iMF time pair.
        t_fm = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std + self.P_mean, self.path_type)
        ln_1 = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std_t + self.P_mean_t, self.path_type)
        ln_2 = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std_r + self.P_mean_r, self.path_type)
        t_mf, r_mf = torch.maximum(ln_1, ln_2), torch.minimum(ln_1, ln_2)
        return t_fm, t_mf, r_mf

    def __call__(self, model, raw_model, x, y,
                 model_t=None, reward_fn=None, decode_func=None,
                 frozen_model=None, **kwargs):
        assert model_t is not None,    "model_t required"
        assert reward_fn is not None,  "reward_fn required"
        assert decode_func is not None, "decode_func required"
        # Lookahead model: frozen_model if given (e.g. an EMA copy), otherwise model_t.

        bsz = x.shape[0]
        device = x.device
        latent_shape = x.shape[1:]
        anchor = frozen_model if frozen_model is not None else model_t

        t_fm, t_mf, r_mf = self.sample_times(bsz)
        t_fm = t_fm.to(device); t_mf = t_mf.to(device); r_mf = r_mf.to(device)
        t_ones  = torch.ones((bsz, 1, 1, 1), device=device)
        t_zeros = torch.zeros((bsz, 1, 1, 1), device=device)

        # ImageNet-class prompts for reward computation (no class-dropout).
        cls_info = ImageNetInfo()
        base_prompt = "A high-resolution, high-quality photograph of a "
        prompts = [base_prompt + cls_info.index_to_description(int(c)) for c in y]

        # No token conditioning.
        cond_kwargs = dict(tcf=0.0, t_min=0.0, t_max=1.0)

        # ── Three-stage rollout for the reward path, with the reward gradient through all three calls ──
        noise = torch.randn_like(x)
        xt_est = noise  + (t_mf  - t_ones)  * model(noise, t_ones, t_mf, y, **cond_kwargs)
        xr_est = xt_est + (r_mf  - t_mf)    * model(xt_est, t_mf, r_mf, y, **cond_kwargs)
        x0_est = xr_est + (t_zeros - r_mf)  * model(xr_est, r_mf, t_zeros, y, **cond_kwargs)
        x0_decoded = (decode_func(x0_est) + 1.0) / 2.0
        rewards = reward_fn.score(x0_decoded, prompts)
        rewards_raw = rewards.clone()
        assert rewards.shape == (bsz,)

        # ── Teacher-anchor regularization ──
        diag_mask = (torch.rand(bsz, device=device) < self.diag_prob).view(-1, 1, 1, 1)
        r_reg = torch.where(diag_mask, t_mf, r_mf)
        model_reg  = model(xt_est, t_mf, r_reg, y, **cond_kwargs)  \
                   - model_t(xt_est, t_mf, r_reg, y, **cond_kwargs)
        reg_loss_mean = torch.mean(model_reg ** 2,
                                   dim=list(range(1, len(model_reg.shape)))) * 0.5

        # ── iMF mean-flow loss: (e, x_target) from a frozen rollout of the anchor model ──
        e_imf = torch.randn_like(x)
        with torch.no_grad():
            # Frozen full rollout: anchor(e, t=1, r=0) yields the avg velocity
            # over the full interval. The displacement (0-1)·v is x_target - e.
            x_target = e_imf + (t_zeros - t_ones) * anchor(
                e_imf, t_ones, t_zeros, y, **cond_kwargs)

        # iMF interpolant: linear path between paired (e, x_target) at time t_mf.
        # Convention: t=0 is data (so x_target lives at t=0), t=1 is noise (e).
        # z_t = (1 − t_mf)·x_target + t_mf·e_imf.
        z_t = ((t_ones - t_mf) * x_target + t_mf * e_imf).detach()

        # FM regression target: v_c = e − x at t=1 → t=0; the pairing makes
        # this the conditional velocity along the linear interpolant.
        v_target = (e_imf - x_target).detach()

        # iMF construction. Boundary condition v_θ(z_t, t) ≡ u_θ(z_t, t, t):
        # we just call the same network with t=r=t_mf for the diagonal.
        unwrapped_model = raw_model.module if hasattr(raw_model, "module") else raw_model
        model_fn = partial(unwrapped_model, y=y, return_logvar=False, **cond_kwargs)

        # Diagonal v_θ via boundary condition (no_grad if we want to use it
        # purely as a tangent direction with no second-order grad. Here we
        # don't detach because we want gradient to flow through v_diag if
        # imf_jvp_stopgrad=False).
        with torch.no_grad():
            v_diag = unwrapped_model(z_t, t_mf, t_mf, y, **cond_kwargs)

        # JVP: (u, du_dt) = jvp(model_fn, (z_t, t_mf, r_mf), (v_diag, 1, 0))
        # → u  = model output at off-diagonal (z_t, t_mf, r_mf)
        # → du = ∂_z u·v_diag + ∂_t u (total derivative along trajectory)
        primals  = (z_t, t_mf, r_mf)
        tangents = (v_diag, torch.ones_like(t_mf), torch.zeros_like(r_mf))
        u_jvp, du_dt = torch.func.jvp(model_fn, primals, tangents)

        # iMF compound function V_θ. Stop-grad on JVP outcome (paper §4.1
        # "About stop-gradient", keeps optimization easier by avoiding
        # high-order gradients in θ; v_θ still gets gradient via shared
        # network params, just not through this specific JVP backward path).
        if self.imf_jvp_stopgrad:
            V_theta = u_jvp + (t_mf - r_mf) * du_dt.detach()
        else:
            V_theta = u_jvp + (t_mf - r_mf) * du_dt

        imf_err = V_theta - v_target
        imf_loss = torch.sum(imf_err.flatten(1) ** 2, dim=-1)
        imf_loss_for_log = imf_loss.detach().mean()

        if self.adaptive_imf:
            imf_loss_eff = adaptive(imf_loss)
        else:
            imf_loss_eff = imf_loss
        imf_loss_eff = imf_loss_eff.mean()

        # ── Total loss
        loss = (reg_loss_mean.mean()
                - self.reward_scale * rewards.mean()
                + self.lambda_imf * imf_loss_eff)

        loss_dict = {
            "reg_loss":        reg_loss_mean.mean().detach(),
            "rewards":         rewards_raw.mean().detach(),
            "imf_loss":        imf_loss_for_log,
            "imf_loss_eff":    imf_loss_eff.detach(),
            "v_target_norm":   v_target.flatten(1).norm(dim=-1).mean().detach(),
            "v_diag_norm":     v_diag.flatten(1).norm(dim=-1).mean().detach(),
            "du_dt_norm":      du_dt.flatten(1).norm(dim=-1).mean().detach(),
            "V_theta_norm":    V_theta.flatten(1).norm(dim=-1).mean().detach(),
        }
        return loss, loss_dict


class FTFMVALUELoss:
    def __init__(
        self,
        P_mean:         float = -0.4,
        P_std:          float = 1.0,
        P_mean_t:       float = -0.2,
        P_std_t:        float = 1.0,
        P_mean_r:       float = 0.2,
        P_std_r:        float = 1.0,
        cfg_prob:       float = 0.1,
        omega:          float = 0.0,
        g_min:          float = 0.0,
        g_max:          float = 1.0,
        path_type:      str   = "linear",
        g_type:         str   = "default",
        **kwargs,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.P_mean_t = P_mean_t
        self.P_std_t = P_std_t
        self.P_mean_r = P_mean_r
        self.P_std_r = P_std_r
        self.cfg_prob = cfg_prob
        self.path_type = path_type
        self.guidance_kwargs = dict(
            omega=omega,
            g_min=g_min,
            g_max=g_max,
            g_type=g_type,
        )

    def cond_drop(self, y):
        bsz = y.shape[0]
        drop_ids = torch.rand(bsz, device=y.device) < self.cfg_prob
        y = torch.where(drop_ids, 1000, y)
        return y, drop_ids

    def sample_times(self, bsz):
        t_rew = torch.rand((bsz, 1, 1, 1))
        tau_rew = torch.rand((bsz, 1, 1, 1)) * t_rew
        ln_1 = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std_t + self.P_mean_t, self.path_type)
        ln_2 = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std_r + self.P_mean_r, self.path_type)
        t_mf, r_mf = torch.maximum(ln_1, ln_2), torch.minimum(ln_1, ln_2)
        return t_rew, tau_rew, t_mf, r_mf

    def __call__(self, model, raw_model, x, y, model_t=None, reward_fn=None, decode_func=None, frozen_model=None, **kwargs):

        assert model_t is not None, "model_t must be provided for FTFMLoss"
        y, drop_ids = y, None
        
        t_rew, tau_rew, t_mf, r_mf = self.sample_times(x.shape[0])
        t_rew, tau_rew, t_mf, r_mf = t_rew.to(x.device), tau_rew.to(x.device), t_mf.to(x.device), r_mf.to(x.device)

        # No token conditioning.
        cond_kwargs = dict(tcf=0.0, t_min=0.0, t_max=1.0)
        reward_scale = 500.0  # fixed gain

        class_description = ImageNetInfo()
        base_prompt = "A high-resolution, high-quality photograph of a "
        prompts = []
        for i in range(y.shape[0]):
            class_id = y[i].item()
            assert class_id < 1000, "no class dropout for reward computation"
            class_name = class_description.index_to_description(class_id)
            prompt = base_prompt + class_name
            prompts.append(prompt)

        noise = torch.randn_like(x)

        t_ones = torch.ones((x.shape[0], 1, 1, 1), device=x.device)
        t_zeros = torch.zeros((x.shape[0], 1, 1, 1), device=x.device)
        with torch.no_grad():
            xt_rew_estimate = noise + (t_rew - t_ones) * frozen_model(noise, t_ones, t_rew, y, **cond_kwargs)
        xt_rew_estimate = xt_rew_estimate.detach().requires_grad_(True)

        loss1 = model(xt_rew_estimate, t_rew, t_rew, y, **cond_kwargs) - model_t(xt_rew_estimate, t_rew, t_rew, y, **cond_kwargs)


        xtau_rew_estimate = xt_rew_estimate + (tau_rew - t_rew) * frozen_model(xt_rew_estimate, t_rew, tau_rew, y, **cond_kwargs)
        x0_estimate = xtau_rew_estimate + (t_zeros - tau_rew) * frozen_model(xtau_rew_estimate, tau_rew, t_zeros, y, **cond_kwargs)

        x0_estimate = (decode_func(x0_estimate) + 1.0) / 2.0

        rewards = reward_fn.score(x0_estimate, prompts)
        rewards_raw = rewards.clone()
        rewards = rewards * reward_scale
        assert rewards.shape == (x.shape[0],), f"Expected rewards shape {(x.shape[0],)}, but got {rewards.shape}"

        model_tautau = frozen_model(xtau_rew_estimate, tau_rew, tau_rew, y, **cond_kwargs) - model_t(xtau_rew_estimate, tau_rew, tau_rew, y, **cond_kwargs)
        tautau_loss_mean = torch.mean(model_tautau ** 2, dim=list(range(1, len(model_tautau.shape))))

        loss2 = rewards - t_rew.squeeze() * tautau_loss_mean * 0.5


        g_hat = torch.autograd.grad(loss2.sum(), xt_rew_estimate, create_graph=False)[0]

        loss_control = torch.mean((loss1 - g_hat.detach()) ** 2)




        x_t_mf, v_t_mf = interpolant(x, t_mf, self.path_type)
        v_tgt_mf = v_t_mf.detach()

        # MeanFlow Loss
        unwrapped_model = raw_model.module if hasattr(raw_model, "module") else raw_model
        model_fn = partial(unwrapped_model, y=y, return_logvar=False, **cond_kwargs)
        primals = (x_t_mf, t_mf, r_mf)
        tangents = (v_tgt_mf, torch.ones_like(t_mf), torch.zeros_like(r_mf))
        u, du_dt = torch.func.jvp(model_fn, primals, tangents)
        u_tgt = v_tgt_mf + (r_mf - t_mf) * du_dt
        u_tgt = u_tgt.detach()
        _, mean_flow_loss = mse_loss(u, u_tgt)

        loss_distillation = mean_flow_loss
        loss_distillation = adaptive(loss_distillation)

        lamda = 1.0
        loss = loss_control + loss_distillation * lamda
        loss_dict = {
            "control_loss": loss_control,
            "rewards": rewards_raw,
            "rewards_scaled": rewards,
            "diag_loss": loss1,
            "loss_distillation": loss_distillation,
            "mean_flow_loss": mean_flow_loss,
        }
        return loss, loss_dict



class FTFMVALUEDMFLoss:
    """
    WTF loss for ImageNet: diagonal value-gradient regression with a normalized reward scale.

    u(x, s) = v_student(x, s, s) - v_teacher(x, s, s) is the control. Time runs from noise (t = 1)
    to data (t = 0), so the optimal control is u* = -∇_x V.

    Single-τ Monte Carlo target at x_t, with τ ~ U[0, t]:
        g_hat = -w_rew ∇_x R(x_0(x_t)) + (t/2) ∇_x ‖u(x_τ, τ)‖²
    where w_rew = target_control_fraction · B / G, and B, G are EMAs of ‖v_teacher‖ and ‖∇_x R‖.

    Loss: L_control = E ‖u(x_t, t) - sg(g_hat)‖² plus adaptive-weighted mean-flow self-distillation.
    """

    def __init__(
        self,
                 detach_lookahead: str = 'none',
        P_mean:                  float = -0.4,
        P_std:                   float = 1.0,
        P_mean_t:                float = -0.2,
        P_std_t:                 float = 1.0,
        P_mean_r:                float = 0.2,
        P_std_r:                 float = 1.0,
        cfg_prob:                float = 0.1,
        omega:                   float = 0.0,
        g_min:                   float = 0.0,
        g_max:                   float = 1.0,
        path_type:               str   = "linear",
        g_type:                  str   = "default",
        adaptive_p:              float = 1.0,
        ema_decay:               float = 0.98,
        target_control_fraction: float = 1.0,
        use_token_cond:          bool  = False,
        tcf_min:                 float = 0.1,
        tcf_max:                 float = 25.0,
        tcf_drop_prob:           float = 0.1,
        tcf_log_uniform:         bool  = False,
        tmin_max:                float = 0.5,
        tmax_min:                float = 0.5,
        num_tau_samples:         int   = 1,
        anchor_weight:           float = 0.0,
        rollout_steps:           int   = 2,
        quantile_clip_pct:       float = 0.0,
        distill_method:          str   = 'jvp',  # jvp | fd | semigroup
        fd_eps:                  float = 1e-3,
        jvp_tangent_clip_ratio:  float = 0.0,
        **kwargs,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.P_mean_t = P_mean_t
        self.P_std_t = P_std_t
        self.P_mean_r = P_mean_r
        self.P_std_r = P_std_r
        self.cfg_prob = cfg_prob
        self.path_type = path_type
        self.adaptive_p = adaptive_p
        self.ema_decay = ema_decay
        self.target_control_fraction = target_control_fraction
        # Token conditioning (off in the reported recipe): per-sample tcf drawn log-uniformly on
        # [tcf_min, tcf_max] and a reward interval [t_min, t_max]. Outside the interval g_hat is zero.
        self.use_token_cond = use_token_cond
        self.tcf_min = tcf_min
        self.tcf_max = tcf_max
        self.tcf_drop_prob = tcf_drop_prob
        self.tcf_log_uniform = tcf_log_uniform
        self.tmin_max = tmin_max
        self.tmax_min = tmax_min
        # Number of τ samples averaged per step (1 in the reported recipe).
        self.num_tau_samples = max(1, int(num_tau_samples))
        # Reward rollout: 1 = one jump x_t → x_0, 2 = x_t → x_τ → x_0 with the cost at τ.
        assert rollout_steps in (1, 2), f"rollout_steps must be 1 or 2, got {rollout_steps}"
        self.rollout_steps = int(rollout_steps)
        assert detach_lookahead in ('none','last','all'), detach_lookahead
        self.detach_lookahead = detach_lookahead
        # Per-sample quantile clipping of ‖∇R‖ at this batch quantile, as in VGG-Flow (0 disables).
        assert 0.0 <= quantile_clip_pct < 1.0, f"quantile_clip_pct in [0,1), got {quantile_clip_pct}"
        self.quantile_clip_pct = float(quantile_clip_pct)
        # Distillation method: jvp (Eulerian) | fd (Lagrangian finite-difference)
        # | semigroup (PSD compositional consistency, WTF Eq. 15).
        assert distill_method in ('jvp', 'fd', 'semigroup'), \
            f"distill_method must be jvp/fd/semigroup, got {distill_method}"
        self.distill_method = str(distill_method)
        self.fd_eps = float(fd_eps)
        self.jvp_tangent_clip_ratio = float(jvp_tangent_clip_ratio)
        # Anchor loss weight: at tcf = 0, match the frozen teacher's off-diagonal.
        self.anchor_weight = float(anchor_weight)
        # Dataset-level norm statistics, tracked as EMAs.
        self.ema_base_flow_norm = None    # B := EMA[ ‖v_teacher(x_t, t, t)‖ ]
        self.ema_reward_grad_norm = None  # G := EMA[ ‖∇_x R‖ ]
        self.guidance_kwargs = dict(
            omega=omega,
            g_min=g_min,
            g_max=g_max,
            g_type=g_type,
        )

    def cond_drop(self, y):
        bsz = y.shape[0]
        drop_ids = torch.rand(bsz, device=y.device) < self.cfg_prob
        y = torch.where(drop_ids, 1000, y)
        return y, drop_ids

    def sample_times(self, bsz):
        # t_val  ∈ [0, 1], value-function anchor time (where u is trained)
        # tau    ∈ [0, t_val], sub-sample time for the running-cost term
        # t_mf > r_mf in [0,1], mean-flow endpoints (reward-independent)
        t_val = torch.rand((bsz, 1, 1, 1))
        tau   = torch.rand((bsz, 1, 1, 1)) * t_val
        ln_1 = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std_t + self.P_mean_t, self.path_type)
        ln_2 = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std_r + self.P_mean_r, self.path_type)
        t_mf, r_mf = torch.maximum(ln_1, ln_2), torch.minimum(ln_1, ln_2)
        return t_val, tau, t_mf, r_mf

    def sample_token_cond(self, bsz, device):
        """Per-sample (tcf, t_min, t_max), each of shape (B, 1, 1, 1). Without token conditioning,
        every sample gets target_control_fraction and the interval [0, 1].
        """
        if not self.use_token_cond:
            tcf = torch.full((bsz, 1, 1, 1), float(self.target_control_fraction), device=device)
            t_min = torch.zeros((bsz, 1, 1, 1), device=device)
            t_max = torch.ones((bsz, 1, 1, 1), device=device)
            return tcf, t_min, t_max

        u = torch.rand((bsz, 1, 1, 1), device=device)
        if self.tcf_log_uniform:
            # Log-uniform sampling.
            log_lo, log_hi = math.log(self.tcf_min), math.log(self.tcf_max)
            tcf = torch.exp(log_lo + u * (log_hi - log_lo))
        else:
            # Uniform sampling.
            tcf = self.tcf_min + u * (self.tcf_max - self.tcf_min)

        # tcf = 0 with probability tcf_drop_prob.
        drop = (torch.rand((bsz, 1, 1, 1), device=device) < self.tcf_drop_prob).float()
        tcf = tcf * (1.0 - drop)

        t_min = torch.rand((bsz, 1, 1, 1), device=device) * self.tmin_max
        t_max = self.tmax_min + torch.rand((bsz, 1, 1, 1), device=device) * (1.0 - self.tmax_min)
        return tcf, t_min, t_max

    @staticmethod
    def _batch_mean_norm(v):
        """Per-sample L2 norm, then batch mean. Returns a detached scalar."""
        return v.flatten(1).norm(dim=-1).mean().detach()

    def _update_ema(self, name, new_value):
        old = getattr(self, name)
        updated = new_value if old is None else self.ema_decay * old + (1 - self.ema_decay) * new_value
        setattr(self, name, updated.detach())
        return updated.detach()

    def __call__(self, model, raw_model, x, y, model_t=None, reward_fn=None,
                 decode_func=None, frozen_model=None, **kwargs):
        assert model_t is not None, "model_t must be provided"
        assert frozen_model is not None, "frozen_model (EMA rollout) must be provided"

        device = x.device
        bsz = x.shape[0]

        t_val, tau, t_mf, r_mf = self.sample_times(bsz)
        t_val, tau = t_val.to(device), tau.to(device)
        t_mf, r_mf = t_mf.to(device), r_mf.to(device)

        # Per-sample conditioning passed to every model call.
        tcf, t_min, t_max = self.sample_token_cond(bsz, device)
        cond = dict(tcf=tcf.flatten(), t_min=t_min.flatten(), t_max=t_max.flatten())

        # Per-class text prompts for the reward model.
        class_description = ImageNetInfo()
        base_prompt = "A high-resolution, high-quality photograph of a "
        prompts = []
        for i in range(bsz):
            class_id = y[i].item()
            assert class_id < 1000, "no class dropout for reward computation"
            prompts.append(base_prompt + class_description.index_to_description(class_id))

        noise   = torch.randn_like(x)
        t_ones  = torch.ones((bsz, 1, 1, 1), device=device)
        t_zeros = torch.zeros((bsz, 1, 1, 1), device=device)

        # ── Rollout to the value-function anchor ─────────────────────────────
        #   x_t = noise + (t_val − 1) · v_frozen(noise, 1, t_val)
        with torch.no_grad():
            x_t = noise + (t_val - t_ones) * frozen_model(noise, t_ones, t_val, y, **cond)
        x_t = x_t.detach()

        # ── Control residual at the anchor, u(x_t, t_val) = v_student - v_teacher ──
        # The teacher is evaluated unconditionally, so u is the deviation from the base flow.
        teacher_cond = dict(tcf=0.0, t_min=0.0, t_max=1.0)
        with torch.no_grad():
            v_teacher_t = model_t(x_t, t_val, t_val, y, **teacher_cond)
        v_student_t = model(x_t, t_val, t_val, y, **cond)
        u_t = v_student_t - v_teacher_t

        # ── With an active rollout, disable parameter gradients during the rollout so DDP sees each
        # parameter once (only input gradients are needed) ──
        active_mode = (frozen_model is raw_model)
        if active_mode:
            _saved_rg = []
            for p in raw_model.parameters():
                if p.requires_grad:
                    _saved_rg.append(p)
                    p.requires_grad_(False)

        # ── Reward and cost subgraphs: one τ per step, shared by both, with both gradients taken with
        # respect to the same x_t leaf (as in t2i/src/lib/ftfm/loss.py).
        # rollout_steps 1: one jump x_t → x_0, cost at t. rollout_steps 2: x_t → x_τ → x_0, cost at τ.
        x_t_leaf = x_t.detach().requires_grad_(True)

        if self.rollout_steps == 2:
            # Shared τ ∈ [0, t_val] from sample_times().
            tau_shared = tau.detach()
            # Shared t → τ forward, with gradient through x_t_leaf.
            v_to_tau = frozen_model(x_t_leaf, t_val, tau_shared, y, **cond)
            x_tau = x_t_leaf + (tau_shared - t_val) * v_to_tau

            # ── Reward: continue τ→0 ──
            v_to_zero = frozen_model(x_tau, tau_shared, t_zeros, y, **cond)
            # ── Reward-gradient detach setting (reward path only, x_tau keeps its graph for the cost) ──
            _dl = getattr(self, 'detach_lookahead', 'none')
            if _dl == 'all':
                # dx_0/dx_t = I  (both hops cut). Same VALUE as the default path.
                x_0_hat = (x_t_leaf
                           + (tau_shared - t_val) * v_to_tau.detach()
                           + (t_zeros - tau_shared) * v_to_zero.detach())
            elif _dl == 'last':
                # dx_0/dx_t = I + (tau - t) J_1  (only the final hop cut)
                x_0_hat = x_tau + (t_zeros - tau_shared) * v_to_zero.detach()
            else:
                x_0_hat = x_tau + (t_zeros - tau_shared) * v_to_zero
            x_0_img = (decode_func(x_0_hat) + 1.0) / 2.0
            rewards_raw_log = reward_fn.score(x_0_img, prompts)

            # ── Cost: u(x_τ, τ), reuses x_τ from shared forward ──
            with torch.no_grad():
                v_teacher_tau = model_t(x_tau.detach(), tau_shared, tau_shared, y, **teacher_cond)
            u_tau = frozen_model(x_tau, tau_shared, tau_shared, y, **cond) - v_teacher_tau
            cost_tau_log = torch.mean(u_tau ** 2, dim=list(range(1, u_tau.ndim)))
            cost_weighted_sum = (t_val.squeeze() * 0.5 * cost_tau_log).sum()

        else:  # rollout_steps == 1
            # Direct x_t → x_0 (no τ).
            v_to_zero = frozen_model(x_t_leaf, t_val, t_zeros, y, **cond)
            if getattr(self, 'detach_lookahead', 'none') != 'none':
                v_to_zero = v_to_zero.detach()  # dx_0/dx_t = I, as in text-to-image
            x_0_hat = x_t_leaf + (t_zeros - t_val) * v_to_zero
            x_0_img = (decode_func(x_0_hat) + 1.0) / 2.0
            rewards_raw_log = reward_fn.score(x_0_img, prompts)

            # ── Cost at t_val, reusing v_teacher_t ──
            u_t_cost = frozen_model(x_t_leaf, t_val, t_val, y, **cond) - v_teacher_t.detach()
            cost_tau_log = torch.mean(u_t_cost ** 2, dim=list(range(1, u_t_cost.ndim)))
            cost_weighted_sum = (t_val.squeeze() * 0.5 * cost_tau_log).sum()

        assert rewards_raw_log.shape == (bsz,), \
            f"rewards must be shape ({bsz},), got {rewards_raw_log.shape}"

        # Two backward passes through the shared graph (retain on first).
        grad_reward = torch.autograd.grad(
            rewards_raw_log.sum(), x_t_leaf, retain_graph=True)[0].detach()
        grad_cost = torch.autograd.grad(
            cost_weighted_sum, x_t_leaf)[0].detach()

        rewards_raw_log = rewards_raw_log.detach()
        cost_tau_log = cost_tau_log.detach()
        # Constant with one τ sample.
        rewards_k_std = torch.tensor(0.0, device=device)
        del x_t_leaf, x_0_hat, x_0_img

        # ── Restore parameter gradients before the mean-flow path ──
        if active_mode:
            for p in _saved_rg:
                p.requires_grad_(True)

        # ── Per-sample quantile clipping of ‖∇_x R‖, as in VGG-Flow ──
        if self.quantile_clip_pct > 0:
            with torch.no_grad():
                norms = grad_reward.flatten(1).norm(dim=-1)  # (B,)
                q = torch.quantile(norms.float(),
                                   torch.tensor(self.quantile_clip_pct, device=norms.device)).to(norms.dtype)
                scale = torch.minimum(q / (norms + 1e-8), torch.ones_like(norms))
                grad_reward = grad_reward * scale.view(-1, *([1] * (grad_reward.ndim - 1)))

        # ── Dataset-normalized reward weight ─────────────────────────────────
        # Per-sample:  w_rew = tcf · EMA[‖v_teacher‖] / EMA[‖∇_x R‖].
        # The EMAs use the ungated batch statistics.
        base_flow_norm_batch   = self._batch_mean_norm(v_teacher_t)
        reward_grad_norm_batch = self._batch_mean_norm(grad_reward)
        cost_grad_norm_batch   = self._batch_mean_norm(grad_cost)

        B = self._update_ema("ema_base_flow_norm",   base_flow_norm_batch)
        G = self._update_ema("ema_reward_grad_norm", reward_grad_norm_batch)
        w_rew = (tcf * B / (G + 1e-8)).detach()  # per-sample (B,1,1,1)

        # Interval gate: the reward applies only when the anchor time is in [t_min, t_max].
        in_interval = ((t_val >= t_min) & (t_val <= t_max)).to(x_t.dtype)

        # ── Optimal-control target and control loss ──────────────────────────
        #   g_hat = in_interval · [ − w_rew · ∇_x R  +  (t_val/2) · ∇_x ‖u(x_τ,τ)‖² ]
        # The reward term is negative because sampling runs with dt < 0.
        g_hat_raw = -torch.nan_to_num(w_rew * grad_reward - grad_cost, nan=0.0)
        g_hat = (in_interval * g_hat_raw).detach()
        loss_control = torch.mean((u_t - g_hat) ** 2)

        # ── Mean-flow distillation loss ──────────────────────────────────────
        # Same per-sample conditioning as the control loss.
        unwrapped_model = raw_model.module if hasattr(raw_model, "module") else raw_model

        # Common setup for JVP and FD (semigroup samples its own times below).
        if self.distill_method in ('jvp', 'fd'):
            with torch.no_grad():
                x_t_mf = noise + (t_mf - t_ones) * frozen_model(noise, t_ones, t_mf, y, **cond)
                v_tgt_mf = frozen_model(x_t_mf.detach(), t_mf, t_mf, y, **cond).detach()
            x_t_mf = x_t_mf.detach()

        if self.distill_method == 'jvp':
            # Eulerian: F(x, t, r) = v_t - (t - r) dF/dt, with dF/dt from a JVP.
            model_fn = partial(unwrapped_model, y=y, return_logvar=False, **cond)
            primals  = (x_t_mf, t_mf, r_mf)
            tangents = (v_tgt_mf, torch.ones_like(t_mf), torch.zeros_like(r_mf))
            v_mf, dv_dt_mf = torch.func.jvp(model_fn, primals, tangents)
            # Optional JVP tangent clip.
            if self.jvp_tangent_clip_ratio > 0:
                with torch.no_grad():
                    v_tgt_norm_ = v_tgt_mf.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1.0)
                    dv_norm_ = dv_dt_mf.flatten(1).norm(dim=-1, keepdim=True)
                    max_dv = self.jvp_tangent_clip_ratio * v_tgt_norm_
                    scale = torch.minimum(max_dv / (dv_norm_ + 1e-8), torch.ones_like(dv_norm_))
                dv_dt_mf = dv_dt_mf * scale.view(-1, *([1] * (dv_dt_mf.ndim - 1)))
            v_mf_tgt = (v_tgt_mf + (r_mf - t_mf) * dv_dt_mf).detach()
            mf_err = v_mf - v_mf_tgt
            mean_flow_loss = torch.sum(mf_err.flatten(1) ** 2, dim=-1)

        elif self.distill_method == 'fd':
            # Lagrangian: dF/dt ≈ (F(x, t + ε, r) - F(x, t, r)) / ε, with gradient only through v_mf.
            eps = self.fd_eps
            t_mf_plus = (t_mf + eps).clamp(max=1.0)
            v_mf = unwrapped_model(x_t_mf, t_mf, r_mf, y, return_logvar=False, **cond)
            with torch.no_grad():
                v_mf_plus = unwrapped_model(x_t_mf, t_mf_plus, r_mf, y, return_logvar=False, **cond)
            dt_actual = (t_mf_plus - t_mf).clamp(min=1e-6)
            dv_dt_mf = ((v_mf_plus - v_mf.detach()) / dt_actual).detach()
            v_mf_tgt = (v_tgt_mf + (r_mf - t_mf) * dv_dt_mf).detach()
            mf_err = v_mf - v_mf_tgt
            mean_flow_loss = torch.sum(mf_err.flatten(1) ** 2, dim=-1)

        else:  # semigroup
            # Compositional consistency (WTF Eq. 15): F(x, t, s) ≈ F(F(x, t, u), u, s) for s ≤ u ≤ t.
            # The model is called with t_start ≥ t_end (noise to data).
            with torch.no_grad():
                uvw = torch.sort(torch.rand(bsz, 3, device=device), dim=-1).values
                s_d = uvw[:, 0:1].view(-1, 1, 1, 1)  # end (data side)
                u_d = uvw[:, 1:2].view(-1, 1, 1, 1)  # mid
                t_d = uvw[:, 2:3].view(-1, 1, 1, 1)  # start (noise side)
                ones_d = torch.ones_like(t_d)
                # I_t: frozen rollout from noise to the largest time t.
                noise_sg = torch.randn_like(noise)
                v_anchor = frozen_model(noise_sg, ones_d, t_d, y, **cond)
                I_t = (noise_sg + (t_d - ones_d) * v_anchor).detach()
            # Differentiable direct jump I_t → s.
            v_big = unwrapped_model(I_t, t_d, s_d, y, return_logvar=False, **cond)
            X_big = I_t + (s_d - t_d) * v_big  # (s-t) < 0, displacement toward data
            # Two stop-gradient jumps: I_t → u, then u → s.
            with torch.no_grad():
                v_tu = unwrapped_model(I_t, t_d, u_d, y, return_logvar=False, **cond)
                X_tu = I_t + (u_d - t_d) * v_tu  # (u-t) < 0
                v_us = unwrapped_model(X_tu, u_d, s_d, y, return_logvar=False, **cond)
                X_us = X_tu + (s_d - u_d) * v_us  # (s-u) < 0
            sg_err = X_big - X_us.detach()
            mean_flow_loss = torch.sum(sg_err.flatten(1) ** 2, dim=-1)
            # Common tensors for the anchor loss and the shared logging code.
            v_mf = X_big
            v_mf_tgt = X_us.detach()
            dv_dt_mf = v_big.detach()

        mf_weights = 1.0 / (mean_flow_loss.detach() + 1e-3).pow(self.adaptive_p)
        loss_distillation = mf_weights * mean_flow_loss

        # ── Anchor loss: at tcf = 0, the student off-diagonal matches the teacher ──
        tcf_zero_mask = (tcf.flatten() == 0.0).to(v_mf.dtype)  # (B,)
        anchor_count = tcf_zero_mask.sum()
        if self.anchor_weight > 0.0 and anchor_count > 0:
            with torch.no_grad():
                v_teacher_mf = model_t(x_t_mf, t_mf, r_mf, y, **teacher_cond).detach()
            anchor_err = (v_mf - v_teacher_mf).flatten(1)
            anchor_per_sample = (anchor_err ** 2).sum(dim=-1)  # (B,)
            anchor_loss = (anchor_per_sample * tcf_zero_mask).sum() / anchor_count
        else:
            anchor_loss = torch.zeros((), device=device, dtype=v_mf.dtype)

        # Per-component terms for the split-backward option (one backward per forward graph under DDP).
        loss_control_scalar      = loss_control                       # already scalar
        # Anchor and distillation share the v_mf graph, so they are backpropagated together.
        loss_distillation_scalar = loss_distillation.mean() + self.anchor_weight * anchor_loss
        loss_anchor_scalar       = self.anchor_weight * anchor_loss  # logged only
        loss = loss_control + loss_distillation + self.anchor_weight * anchor_loss

        # ── Diagnostics ──────────────────────────────────────────────────────
        with torch.no_grad():
            g_hat_flat = g_hat.flatten(1)
            u_t_flat   = u_t.flatten(1)
            g_hat_norm = g_hat_flat.norm(dim=-1).mean()
            u_t_norm   = u_t_flat.norm(dim=-1).mean()
            cos_sim_u_ghat = torch.nn.functional.cosine_similarity(
                u_t_flat, g_hat_flat, dim=-1).mean()

        loss_dict = {
            "control_loss":             loss_control,
            "mean_flow_loss":           mean_flow_loss,
            "rewards":                  rewards_raw_log,
            "cost_tau":                 cost_tau_log,
            # Norm diagnostics
            "base_flow_norm_batch":     base_flow_norm_batch,
            "reward_grad_norm_batch":   reward_grad_norm_batch,
            "cost_grad_norm_batch":     cost_grad_norm_batch,
            "ema_base_flow_norm":       B,
            "ema_reward_grad_norm":     G,
            "w_rew":                    w_rew.mean(),
            # Conditioning diagnostics
            "tcf":                      tcf.mean(),
            "t_min":                    t_min.mean(),
            "t_max":                    t_max.mean(),
            "in_interval":              in_interval.mean(),
            # Control-target alignment
            "g_hat_norm":               g_hat_norm,
            "u_t_norm":                 u_t_norm,
            "cos_sim_u_ghat":           cos_sim_u_ghat,
            # Multi-τ MC diagnostic
            "rewards_k_std":            rewards_k_std,
            # Anchor diagnostic
            "anchor_loss":              anchor_loss,
            "anchor_frac":              (anchor_count / bsz).detach(),
            # Split-backward terms (--split-backward).
            "_term_control":            loss_control_scalar,
            "_term_distill":            loss_distillation_scalar,
            "_term_anchor":             loss_anchor_scalar,
        }
        return loss, loss_dict





class DMFLoss:
    def __init__(
        self,
        P_mean:         float = -0.4,
        P_std:          float = 1.0,
        P_mean_t:       float = -0.2,
        P_std_t:        float = 1.0,
        P_mean_r:       float = 0.2,
        P_std_r:        float = 1.0,
        cfg_prob:       float = 0.1,
        omega:          float = 0.0,
        g_min:          float = 0.0,
        g_max:          float = 1.0,
        path_type:      str   = "linear",
        g_type:         str   = "default"
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.P_mean_t = P_mean_t
        self.P_std_t = P_std_t
        self.P_mean_r = P_mean_r
        self.P_std_r = P_std_r
        self.cfg_prob = cfg_prob
        self.path_type = path_type
        self.guidance_kwargs = dict(
            omega=omega,
            g_min=g_min,
            g_max=g_max,
            g_type=g_type,
        )

    def cond_drop(self, y):
        bsz = y.shape[0]
        drop_ids = torch.rand(bsz, device=y.device) < self.cfg_prob
        y = torch.where(drop_ids, 1000, y)
        return y, drop_ids

    def sample_times(self, bsz):
        t_fm = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std + self.P_mean, self.path_type)
        ln_1 = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std_t + self.P_mean_t, self.path_type)
        ln_2 = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std_r + self.P_mean_r, self.path_type)
        t_mf, r_mf = torch.maximum(ln_1, ln_2), torch.minimum(ln_1, ln_2)
        return t_fm, t_mf, r_mf

    def __call__(self, model, raw_model, x, y, model_t=None, **kwargs):
        if model_t is None:
            model_t = model
        # class dropout
        y, drop_ids = self.cond_drop(y)
        # sample timesteps
        t_fm, t_mf, r_mf = self.sample_times(x.shape[0])
        t_fm, t_mf, r_mf = t_fm.to(x.device), t_mf.to(x.device), r_mf.to(x.device)

        # flow matching loss
        x_t_fm, v_t_fm = interpolant(x, t_fm, self.path_type)
        v_fm = model(x_t_fm, t_fm, t_fm, y, return_logvar=False)
        v_tgt_fm = compute_guidance(
            model_t, x_t_fm, t_fm, y, v_t_fm, drop_ids, **self.guidance_kwargs
        ).detach()
        fm_l2_loss, fm_l2_mean_loss = adaptive_loss(v_fm, v_tgt_fm)

        # MeanFlow loss
        x_t_mf, v_t_mf = interpolant(x, t_mf, self.path_type)
        v_tgt_mf = compute_guidance(
            model_t, x_t_mf, t_mf, y, v_t_mf, drop_ids, **self.guidance_kwargs
        ).detach()
        
        unwrapped_model = raw_model.module if hasattr(raw_model, "module") else raw_model
        model_fn = partial(unwrapped_model, y=y, return_logvar=False)
        primals = (x_t_mf, t_mf, r_mf)
        tangents = (v_tgt_mf, torch.ones_like(t_mf), torch.zeros_like(r_mf))
        u, du_dt = torch.func.jvp(model_fn, primals, tangents)
        u_tgt = v_tgt_mf + (r_mf - t_mf) * du_dt
        u_tgt = u_tgt.detach()
        mf_l2_loss, mf_adaptive_loss = adaptive_loss(u, u_tgt)
        loss = fm_l2_mean_loss + mf_adaptive_loss
        loss_dict = {
            "fm_l2_loss": fm_l2_loss,
            "fm_l2_mean_loss": fm_l2_mean_loss,
            "mf_l2_loss": mf_l2_loss,
            "mf_adaptive_loss": mf_adaptive_loss,
        }
        return loss, loss_dict

