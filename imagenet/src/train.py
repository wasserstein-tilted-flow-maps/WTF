import argparse
import copy
import logging
import os
import json
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.dmft import DMFT_models
from models.sit import SiT_models
from loss import FTFMLoss, FTFMVALUELoss, FTFMVALUEDMFLoss, FTFMOCImFLoss
from samplers import euler_sampler, diagonal_euler_sampler, ftfm_euler_sampler

from diffusers.models import AutoencoderKL
import wandb
import math
from torchvision.utils import make_grid

import numpy as np
from PIL import Image
from torchvision import transforms
import webdataset as wds

from imscore.hps.model import HPSv2
from imscore.imreward.model import ImageReward
from imscore.pickscore.model import PickScorer
from rewards import RednessReward
import functools
import gc


torch.set_num_threads(4)
torch.set_num_interop_threads(1)
logger = get_logger(__name__)


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


def progression_grid(samples_by_cond, n_cols: int = 8):
    """Stack one grid per conditioning value as rows, sharing noise across columns.
    Input: dict[str -> (B, C, H, W)]. Output: (H_total, W_total, 3) uint8.
    """
    keys = list(samples_by_cond.keys())
    rows = []
    for k in keys:
        v = samples_by_cond[k][:n_cols]
        rows.append(v)
    x = torch.cat(rows, dim=0)  # (rows·cols, C, H, W)
    grid = make_grid(x.clamp(0, 1), nrow=n_cols, value_range=(0, 1), padding=2)
    return grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    """Sample from latent VAE"""
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias)
    return z

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])



@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    decay: float = 0.9999,
    only_trainable: bool = True,   # skip params with requires_grad=False
    update_buffers: bool = True    # keep buffers (e.g., running stats) in sync
):
    """
    Exponential moving average update.
    - Parameters with requires_grad=False are skipped when only_trainable=True.
    - Buffers are copied directly (no decay) if update_buffers=True.
    """
    # Map names consistently (DDP may prepend "module.")
    def _strip(name: str) -> str:
        for pre in ("module.", "orig_mod.", "_orig_mod."):
            if name.startswith(pre):
                return name[len(pre):]
        return name

    # Build quick-lookup sets & dicts
    trainable = {_strip(n) for n, p in model.named_parameters() if p.requires_grad}
    ema_params = { _strip(n): p for n, p in ema_model.named_parameters() }
    model_params = { _strip(n): p for n, p in model.named_parameters() }

    # EMA on parameters
    for name, p_model in model_params.items():
        if only_trainable and name not in trainable:
            continue
        p_ema = ema_params.get(name, None)
        if p_ema is None:
            continue
        # Guard: only EMA float tensors
        if p_ema.is_floating_point():
            p_ema.mul_(decay).add_(p_model.data, alpha=1.0 - decay)
        else:
            p_ema.copy_(p_model.data)

    # Keep buffers in sync (no EMA decay)
    if update_buffers:
        ema_buffers = { _strip(n): b for n, b in ema_model.named_buffers() }
        for name, b_model in model.named_buffers():
            name = _strip(name)
            b_ema = ema_buffers.get(name, None)
            if b_ema is not None:
                b_ema.copy_(b_model)


def create_logger(logging_dir):
    """Create a logger that writes to a log file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


def vae_decode_func(vae, latents_scale, latents_bias, latents):
    latents = (latents - latents_bias) / latents_scale
    images = vae.decode(latents).sample
    return images


_SYNSET_TO_IDX = None

def _get_synset_to_idx():
    """Build and cache a mapping from ImageNet synset IDs to class indices."""
    global _SYNSET_TO_IDX
    if _SYNSET_TO_IDX is None:
        from timm.data import ImageNetInfo
        info = ImageNetInfo()
        _SYNSET_TO_IDX = {s: i for i, s in enumerate(info.label_names())}
    return _SYNSET_TO_IDX


def _cls_handler(key, data):
    """Custom WDS decoder: return .cls as a raw string instead of int."""
    if not key.endswith(".cls"):
        return None
    return data.decode("utf-8").strip() if isinstance(data, bytes) else str(data).strip()


def _decode_class(c):
    """Map a synset ID (e.g. 'n02127052') or raw int string to a class index."""
    if isinstance(c, bytes):
        c = c.decode("utf-8")
    c = str(c).strip()
    if c.isdigit():
        return int(c)
    return _get_synset_to_idx()[c]


def main(args):
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    # static_graph=True: with an active rollout and the mean-flow JVP, the same parameters appear in
    # two forward graphs per iteration.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, static_graph=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    block_kwargs = {"attn_func": args.attn_func, "qk_norm": args.qk_norm}
    model = DMFT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg_embedding=(args.cfg_prob > 0.0),
        dmf_depth=args.dmf_depth,
        use_logvar=args.use_logvar,
        use_token_cond=args.use_token_cond,
        num_tcf_tokens=args.num_tcf_tokens,
        num_interval_tokens=args.num_interval_tokens,
        **block_kwargs
    )
    if not args.pretrained:
        raise SystemExit("--pretrained is required: the DMF output layer is zero-initialized, so without "
                         "the pretrained weights the model outputs zero and receives no training signal. "
                         "Download dmf_xl_2_256.pt from huggingface.co/kyungmnlee/DMF (see docs/SETUP.md).")
    if args.pretrained:
        if accelerator.is_main_process:
            logger.info(f"Loading pretrained model from {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        # Accept a bare state_dict or a training checkpoint, preferring the EMA weights.
        if isinstance(ckpt, dict) and ("ema" in ckpt or "model" in ckpt):
            state = ckpt.get("ema", ckpt.get("model"))
            if accelerator.is_main_process:
                logger.info(f"Pretrained is a training checkpoint; using key "
                            f"{'ema' if 'ema' in ckpt else 'model'} "
                            f"(step {ckpt.get('steps', '?')}).")
        else:
            state = ckpt
        # Reconcile shape mismatches. A learn_sigma checkpoint's final layer has 2C output channels, of
        # which the first C are kept. Other mismatched tensors keep the model's initialization.
        cur_sd = model.state_dict()
        filtered, sliced, dropped = {}, [], []
        for k, v in state.items():
            if k not in cur_sd:
                continue
            want = cur_sd[k].shape
            if v.shape == want:
                filtered[k] = v
            elif (v.ndim == want.__len__()
                  and v.shape[0] == 2 * want[0]
                  and v.shape[1:] == want[1:]):
                # 2·C→C along dim 0: keep the first half (velocity part).
                filtered[k] = v[:want[0]]
                sliced.append((k, tuple(v.shape), tuple(want)))
            else:
                dropped.append((k, tuple(v.shape), tuple(want)))
        msg = model.load_state_dict(filtered, strict=False)
        if accelerator.is_main_process:
            logger.info(msg)
            if sliced:
                logger.info(f"Sliced {len(sliced)} key(s) from 2·C→C (learn_sigma → velocity-only):")
                for k, got, want in sliced:
                    logger.info(f"  {k}: ckpt={got}  →  model={want}")
            if dropped:
                logger.info(f"Dropped {len(dropped)} mismatched key(s):")
                for k, got, want in dropped:
                    logger.info(f"  {k}: ckpt={got}  model={want}")
        del ckpt, state, filtered
        gc.collect()

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    vae_name = "stabilityai/sd-vae-ft-mse"
    vae_cache_dir = "/tmp/.hf_cache/hub"
    try:
        vae = AutoencoderKL.from_pretrained(vae_name, cache_dir=vae_cache_dir, local_files_only=True).to(device).eval()
    except Exception:
        if accelerator.is_main_process:
            logger.info(f"VAE not cached locally — downloading {vae_name}")
        vae = AutoencoderKL.from_pretrained(vae_name, cache_dir=vae_cache_dir).to(device).eval()
    requires_grad(ema, False)

    # Create the fixed, unconditional velocity generator
    model_t = deepcopy(model).to(device)
    requires_grad(model_t, False)
    model_t.eval()

    
    if args.reward_model == "hps":
        reward_fn = HPSv2.from_pretrained("RE-N-Y/hpsv21")
    elif args.reward_model == "imreward":
        reward_fn = ImageReward.from_pretrained("RE-N-Y/imreward")
    elif args.reward_model == "pickscore":
        reward_fn = PickScorer.from_pretrained("RE-N-Y/pickscore")
    elif args.reward_model == "redness":
        reward_fn = RednessReward()
    reward_fn = reward_fn.to(device)
    reward_fn.eval()

    # HPSv2 for validation, reusing the training reward when it is HPS.
    hps_eval_fn = None
    if args.eval_hps:
        if args.reward_model == "hps":
            hps_eval_fn = reward_fn
        else:
            try:
                hps_eval_fn = HPSv2.from_pretrained("RE-N-Y/hpsv21").to(device).eval()
            except Exception as e:
                if accelerator.is_main_process:
                    logger.warning(f"Could not load HPSv2 for eval ({type(e).__name__}: {e}). "
                                   "Proceeding without HPS validation metrics.")
                hps_eval_fn = None

    gc.collect()
    torch.cuda.empty_cache()

    latents_scale = torch.tensor([0.18215] * 4).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor([0.0] * 4).view(1, 4, 1, 1).to(device)
    decode_func = functools.partial(vae_decode_func, vae, latents_scale, latents_bias)

    # create loss function
    loss_config = dict(
        P_mean=args.P_mean,
        P_std=args.P_std,
        P_mean_t=args.P_mean_t,
        P_std_t=args.P_std_t,
        P_mean_r=args.P_mean_r,
        P_std_r=args.P_std_r,
        cfg_prob=args.cfg_prob,
        omega=args.omega,
        g_min=args.g_min,
        g_max=args.g_max,
        path_type=args.path_type,
        g_type=args.g_type,
        diag_prob=args.diag_prob,
        target_control_fraction=args.target_control_fraction,
        use_token_cond=args.use_token_cond,
        tcf_min=args.tcf_min,
        tcf_max=args.tcf_max,
        tcf_drop_prob=args.tcf_drop_prob,
        tcf_log_uniform=args.tcf_log_uniform,
        tmin_max=args.tmin_max,
        tmax_min=args.tmax_min,
        num_tau_samples=args.num_tau_samples,
        anchor_weight=args.anchor_weight,
        rollout_steps=args.rollout_steps,
    )
    if args.use_imf_oc:
        # Optimal-control variant with iMF-style distillation (--use-imf-oc). Takes precedence over
        # --value-approach and --use-logvar.
        loss_fn = FTFMOCImFLoss(
            P_mean=args.P_mean, P_std=args.P_std,
            P_mean_t=args.P_mean_t, P_std_t=args.P_std_t,
            P_mean_r=args.P_mean_r, P_std_r=args.P_std_r,
            cfg_prob=args.cfg_prob, omega=args.omega,
            g_min=args.g_min, g_max=args.g_max,
            path_type=args.path_type, g_type=args.g_type,
            diag_prob=args.diag_prob,
            reward_scale=args.imf_oc_reward_scale,
            lambda_imf=args.imf_oc_lambda,
            imf_jvp_stopgrad=not args.imf_oc_no_stopgrad,
            adaptive_imf=args.imf_oc_adaptive,
        )
    elif args.value_approach:
        loss_fn = FTFMVALUELoss(**loss_config)
    else:
        loss_fn = FTFMLoss(**loss_config)
    if args.use_logvar and not args.use_imf_oc:
        # Only FTFMVALUEDMFLoss exposes distill-method controls.
        loss_fn = FTFMVALUEDMFLoss(
            **loss_config,
            distill_method=args.distill_method,
            fd_eps=args.fd_eps,
            jvp_tangent_clip_ratio=args.jvp_tangent_clip_ratio,
            detach_lookahead=args.detach_lookahead,
        )

    if accelerator.is_main_process:
        logger.info(f"Training Parameters: {sum(p.numel() for p in model.parameters()):,}")
    # Setup optimizer
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
        fused=True,
    )

    # ── Data: the WTF losses roll out from noise and never read x, so --synthetic-data skips the
    # WebDataset and VAE encode. Only x's shape and the class labels are used. ──
    if args.synthetic_data:
        # Latent shape: 4 channels, resolution/8 per side (sd-vae-ft-mse).
        LATENT_C = 4
        latent_hw = args.resolution // 8

        class _SyntheticLatentDataset(torch.utils.data.IterableDataset):
            """Yields (zeros((4, h, h)), random class) forever."""
            def __iter__(self):
                while True:
                    yield (
                        torch.zeros(LATENT_C, latent_hw, latent_hw),
                        torch.randint(0, 1000, (1,)).item(),
                    )

        local_batch_size = int(args.batch_size // accelerator.num_processes)
        train_dataloader = DataLoader(
            _SyntheticLatentDataset(),
            batch_size=local_batch_size,
            num_workers=0,
            drop_last=True,
        )
        if accelerator.is_main_process:
            logger.info(
                f"Using synthetic data (no ImageNet, no VAE encode) — "
                f"latent shape ({LATENT_C}, {latent_hw}, {latent_hw}) "
                f"batch={local_batch_size}/proc")
    else:
        # ── WebDataset path (only when --no-synthetic-data) ──
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        cache_dir = "/tmp/imagenet"
        if not args.data_dir:
            raise SystemExit("--data-dir is required: the ImageNet webdataset shard prefix "
                             "(set substrate.data_dir / $IMAGENET_WDS; see docs/SETUP.md)")
        if args.data_dir.startswith("gs://"):
            os.makedirs(cache_dir, exist_ok=True)
            urls = []
            for i in range(args.num_shards):
                shard_name = f"train-{i:06d}.tar"
                local_path = os.path.join(cache_dir, shard_name)
                if os.path.exists(local_path):
                    urls.append(local_path)
                else:
                    urls.append(
                        f"pipe:bash -c '"
                        f"LOCAL={local_path}.tmp.$$ ; "
                        f"for try in 1 2 3 4 5; do "
                        f"gcloud storage cat {args.data_dir}-{i:06d}.tar > \"$LOCAL\" && "
                        f"mv \"$LOCAL\" {local_path} && break; "
                        f"sleep $((try*2)); done; "
                        f"cat {local_path}'"
                    )
        else:
            urls = [f"{args.data_dir}-{i:06d}.tar" for i in range(args.num_shards)]

        local_batch_size = int(args.batch_size // accelerator.num_processes)
        wds_dataset = (
            wds.WebDataset(urls, nodesplitter=wds.split_by_node, shardshuffle=True,
                           handler=wds.warn_and_continue)
            .shuffle(5000)
            .decode(_cls_handler, "pil", handler=wds.warn_and_continue)
            .to_tuple("jpg;jpeg;png;JPEG", "cls", handler=wds.warn_and_continue)
            .map_tuple(transform, _decode_class)
        )
        train_dataloader = DataLoader(
            wds_dataset,
            batch_size=local_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4 if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0,
        )
        if accelerator.is_main_process:
            logger.info(f"WebDataset with {args.num_shards} shards from {args.data_dir}")

    # Prepare models for training:
    update_ema(ema, model, decay=0.0, only_trainable=False, update_buffers=True)
    model.train()  # enables embedding dropout for CFG
    ema.eval()  # EMA model should always be in eval mode
    model_t.eval()  # model_t is a fixed teacher, so always eval mode

    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            weights_only=False
        )
        model_state = {k.replace('module.', '', 1): v for k, v in ckpt['model'].items()}
        msg = model.load_state_dict(model_state)
        if accelerator.is_main_process:
            logger.info(msg)
        ema_state = {k.replace('module.', '', 1): v for k, v in ckpt['ema'].items()}
        ema.load_state_dict(ema_state)
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']
        if accelerator.is_main_process:
            logger.info(f"Resume from {global_step}")

    model, optimizer = accelerator.prepare(model, optimizer)

    if accelerator.is_main_process:
        wandb_base = args.wandb_dir
        os.makedirs(wandb_base, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_base
        os.environ["WANDB_CACHE_DIR"] = os.path.join(wandb_base, ".cache")
        os.environ["WANDB_DATA_DIR"] = os.path.join(wandb_base, ".data")
        os.environ["WANDB_CONFIG_DIR"] = os.path.join(wandb_base, ".config")

        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name=args.project_name,
            config=tracker_config,
            init_kwargs={"wandb": {
                "name": f"{args.exp_name}",
                "dir": wandb_base,
                "settings": wandb.Settings(save_code=False),
            }},
        )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Create sampling noise:
    sample_batch_size = 64 // accelerator.num_processes
    ys = torch.randint(0, 1000, (sample_batch_size,), device=device)
    x1 = torch.randn((sample_batch_size, 4, latent_size, latent_size), device=device)

    # Fixed text prompts for reward/ImageReward scoring of sampled images.
    from timm.data import ImageNetInfo as _ImageNetInfo
    _imnet_info = _ImageNetInfo()
    _prompt_prefix = "A high-resolution, high-quality photograph of a "
    sample_prompts = [_prompt_prefix + _imnet_info.index_to_description(int(c)) for c in ys.tolist()]
    sampling_kwargs = dict(
        num_steps=4,                  # NFE=4, used by TCF + interval sweeps.
        cfg_scale=1.0,
        g_min=0.0,
        g_max=1.0,
        shift=1.0,
        path_type=args.path_type,
    )
    sampling_kwargs_onestep = sampling_kwargs.copy()
    sampling_kwargs_onestep['num_steps'] = 1
    # Additional qualitative samples at NFE 2, 8 and 30 (NFE 4 and 1 are logged as
    # `samples_uncond` and `samples_uncond_onestep`).
    SAMPLING_KWARGS_EXTRA = {
        nfe: {**sampling_kwargs, 'num_steps': nfe} for nfe in (2, 8, 30)
    }

    grad_norm = torch.tensor(0.0) # init as zero
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # Synthetic data is already latent. Real images are VAE-encoded.
            if not args.synthetic_data:
                with torch.no_grad():
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    unwrapped_model = accelerator.unwrap_model(model)
                    # frozen_model_mode: ema routes rollouts through the EMA copy, active through the live model.
                    # The loss only requests gradients with respect to x_t, so rollouts add no parameter gradients.
                    rollout_ref = unwrapped_model if args.frozen_model_mode == "active" else ema
                    loss, loss_dict = loss_fn(model, unwrapped_model, x, y, model_t=model_t, reward_fn=reward_fn, decode_func=decode_func, frozen_model=rollout_ref)
                loss = loss.mean()

                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model, decay=args.ema_decay, only_trainable=True, update_buffers=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (global_step == 10 or global_step==100 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                # Conditioning sweeps with fixed noise. tcf_sweep varies the reward strength from 0 to 25, and
                # interval_sweep varies the time window where the reward applies (early = noisy half).
                tcf_sweep = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0]
                interval_sweep = [
                    ("full",  0.0, 1.0),
                    ("mid",   0.3, 0.7),
                    ("early", 0.5, 1.0),  # reward only in the noisy half (t near 1)
                    ("late",  0.0, 0.5),  # reward only in the near-data half
                ]
                tcf_mid = args.tcf_eval
                with torch.no_grad():
                    with accelerator.autocast():
                        samples_uncond         = euler_sampler(model_t, x1, ys, **sampling_kwargs).to(torch.float32)
                        samples_uncond_onestep = euler_sampler(model_t, x1, ys, **sampling_kwargs_onestep).to(torch.float32)
                        # TCF sweep at 4 steps.
                        tcf_samples = {}
                        for _tcf in tcf_sweep:
                            key = f"tcf{_tcf:g}".replace(".", "p")
                            tcf_samples[key] = euler_sampler(
                                ema, x1, ys, tcf=_tcf, t_min=0.0, t_max=1.0,
                                **sampling_kwargs,
                            ).to(torch.float32)
                        # One-step variant at a smaller sweep for contrast.
                        tcf_samples_onestep = {}
                        for _tcf in [0.0, tcf_mid, tcf_sweep[-1]]:
                            key = f"tcf{_tcf:g}".replace(".", "p")
                            tcf_samples_onestep[key] = euler_sampler(
                                ema, x1, ys, tcf=_tcf, t_min=0.0, t_max=1.0,
                                **sampling_kwargs_onestep,
                            ).to(torch.float32)
                        # Interval sweep at a single tcf.
                        interval_samples = {}
                        for name, tmin, tmax in interval_sweep:
                            interval_samples[name] = euler_sampler(
                                ema, x1, ys, tcf=tcf_mid, t_min=tmin, t_max=tmax,
                                **sampling_kwargs,
                            ).to(torch.float32)
                        # Diagonal (velocity ODE) sampling, unconditional.
                        samples_diag = diagonal_euler_sampler(
                            ema, x1, ys, num_steps=50,
                            **{k: v for k, v in sampling_kwargs.items() if k != 'num_steps'},
                        ).to(torch.float32)

                    def _vae_decode(z):
                        return ((vae.decode((z - latents_bias) / latents_scale).sample) + 1) / 2.

                    samples_uncond         = _vae_decode(samples_uncond)
                    samples_uncond_onestep = _vae_decode(samples_uncond_onestep)
                    tcf_samples         = {k: _vae_decode(v) for k, v in tcf_samples.items()}
                    tcf_samples_onestep = {k: _vae_decode(v) for k, v in tcf_samples_onestep.items()}
                    interval_samples    = {k: _vae_decode(v) for k, v in interval_samples.items()}
                    samples_diag = _vae_decode(samples_diag)

                    # Unconditional samples at NFE 2, 8 and 30, alongside NFE 4 and NFE 1.
                    samples_uncond_nfe_extra = {}
                    with torch.no_grad():
                        with accelerator.autocast():
                            for _nfe, _kw in SAMPLING_KWARGS_EXTRA.items():
                                _z = euler_sampler(model_t, x1, ys, **_kw).to(torch.float32)
                                samples_uncond_nfe_extra[_nfe] = _vae_decode(_z)

                # Build the progression grid on local tensors, then gather it so rank 0 logs one image.
                n_cols = min(8, samples_uncond.shape[0])
                tcf_prog_local = progression_grid(tcf_samples, n_cols=n_cols)  # rank-local image
                interval_prog_local = progression_grid(interval_samples, n_cols=n_cols)

                out_samples_uncond         = accelerator.gather(samples_uncond.to(torch.float32))
                out_samples_uncond_onestep = accelerator.gather(samples_uncond_onestep.to(torch.float32))
                out_samples_diag           = accelerator.gather(samples_diag.to(torch.float32))
                out_samples_uncond_nfe_extra = {
                    nfe: accelerator.gather(v.to(torch.float32))
                    for nfe, v in samples_uncond_nfe_extra.items()
                }
                out_tcf         = {k: accelerator.gather(v.to(torch.float32)) for k, v in tcf_samples.items()}
                out_tcf_onestep = {k: accelerator.gather(v.to(torch.float32)) for k, v in tcf_samples_onestep.items()}
                out_interval    = {k: accelerator.gather(v.to(torch.float32)) for k, v in interval_samples.items()}

                log_payload = {
                    "samples_uncond":         wandb.Image(array2grid(out_samples_uncond)),       # NFE=4
                    "samples_uncond_onestep": wandb.Image(array2grid(out_samples_uncond_onestep)), # NFE=1
                    "samples_diag_50step":    wandb.Image(array2grid(out_samples_diag)),
                    # Progression grids with fixed noise: rows sweep the setting, columns share seeds.
                    "tcf_progression":        wandb.Image(
                        tcf_prog_local,
                        caption=f"rows (top→bottom): tcf = {tcf_sweep}",
                    ),
                    "interval_progression":   wandb.Image(
                        interval_prog_local,
                        caption=f"rows (top→bottom) at tcf={tcf_mid}: "
                                f"{[name for name,_,_ in interval_sweep]}",
                    ),
                }
                for k, v in out_tcf.items():
                    log_payload[f"samples/{k}"] = wandb.Image(array2grid(v))
                for k, v in out_tcf_onestep.items():
                    log_payload[f"samples_onestep/{k}"] = wandb.Image(array2grid(v))
                for k, v in out_interval.items():
                    log_payload[f"samples_interval_tcf{tcf_mid:g}/{k}".replace(".", "p")] = wandb.Image(array2grid(v))
                # Multi-NFE qualitative samples (uncond) at NFE ∈ {2, 8, 30}.
                for nfe, v in out_samples_uncond_nfe_extra.items():
                    log_payload[f"samples_uncond_nfe{nfe}"] = wandb.Image(array2grid(v))

                # Score each local variant with the training reward (and optionally HPS), averaged across ranks.
                with torch.no_grad():
                    variants = {"samples_uncond": samples_uncond, "samples_diag_50step": samples_diag}
                    for k, v in tcf_samples.items():
                        variants[f"samples/{k}"] = v
                    for k, v in tcf_samples_onestep.items():
                        variants[f"samples_onestep/{k}"] = v
                    for k, v in interval_samples.items():
                        variants[f"samples_interval/{k}"] = v
                    for name, imgs in variants.items():
                        imgs_clamped = imgs.clamp(0.0, 1.0)
                        r_train = reward_fn.score(imgs_clamped, sample_prompts).float().mean()
                        log_payload[f"reward/{name}"] = accelerator.gather(
                            r_train.unsqueeze(0)).mean().item()
                        if hps_eval_fn is not None:
                            r_hps = hps_eval_fn.score(imgs_clamped, sample_prompts).float().mean()
                            log_payload[f"hps/{name}"] = accelerator.gather(
                                r_hps.unsqueeze(0)).mean().item()

                # tcf vs. reward curve for wandb.
                for _tcf in tcf_sweep:
                    key = f"tcf{_tcf:g}".replace(".", "p")
                    r_key = f"reward/samples/{key}"
                    if r_key in log_payload:
                        log_payload[f"reward_vs_tcf/{key}"] = log_payload[r_key]

                accelerator.log(log_payload)
                logging.info("Generating EMA samples done.")

            grad_norm = grad_norm.to(accelerator.device)
            logs = {
                "loss": accelerator.gather(loss).mean().detach().item(),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            for key, value in loss_dict.items():
                logs[key] = accelerator.gather(value).mean().detach().item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="./exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--project-name", type=str, required=True)
    parser.add_argument("--wandb-dir", type=str, default="./wandb/")

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--dmf-depth", type=int, default=20)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)


    # Fine-tuning
    parser.add_argument("--pretrained", type=str, default=None)

    # training loss
    parser.add_argument("--g-type", type=str, default="default", choices=["default", "mg", "distill"])
    parser.add_argument("--P-mean", type=float, default=0.0, help="Mean of the P distribution.")
    parser.add_argument("--P-std", type=float, default=1.0, help="Standard deviation of the P distribution.")
    parser.add_argument("--P-mean-t", type=float, default=0.4, help="Mean of the P_t distribution.")
    parser.add_argument("--P-std-t", type=float, default=1.0, help="Standard deviation of the P_t distribution.")
    parser.add_argument("--P-mean-r", type=float, default=-1.2, help="Mean of the P_r distribution.")
    parser.add_argument("--P-std-r", type=float, default=1.0, help="Standard deviation of the P_r distribution.")
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--attn-func", type=str, default="base", choices=["base", "torch_sdpa", "fa2", "fa3"])
    parser.add_argument("--omega", type=float, default=0.5, help="Scale for the mean flow loss.")
    parser.add_argument("--g-min", type=float, default=0.0, help="Minimum value for the cfg index.")
    parser.add_argument("--g-max", type=float, default=1.0, help="Maximum value for the cfg index.")
    parser.add_argument("--diag-prob", type=float, default=0.5,
                        help="Probability of diagonal (t,t) vs off-diagonal (t,r) regularization per sample.")

    # dataset (WebDataset tar shards)
    parser.add_argument("--synthetic-data", action=argparse.BooleanOptionalAction, default=True,
                        help="On by default: WTF rolls every training sample out from noise and "
                             "uses the batch only for its shape and (random) class labels, so no "
                             "ImageNet images are read. --no-synthetic-data streams real ImageNet "
                             "shards from --data-dir instead (only their labels affect the loss).")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Tar shard prefix, e.g. gs://bucket/path/train (shards: train-000000.tar … train-NNNNNN.tar)")
    parser.add_argument("--num-shards", type=int, default=1281, help="Number of WebDataset tar shards")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=10000)
    parser.add_argument("--sampling-steps", type=int, default=500)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    # DiT uses adam-beta2 = 0.95
    parser.add_argument("--adam-beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="Exponential moving average decay.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--reward-model", type=str, default="hps",
                        choices=["hps", "imreward", "pickscore", "redness"])
    parser.add_argument("--eval-hps", action="store_true",
                        help="Score sampled images with HPSv2 at each validation step.")
    parser.add_argument("--target-control-fraction", type=float, default=25.0,
                        help="FTFMVALUEDMFLoss: reward-pull magnitude as a fraction of ‖v_teacher‖. "
                             "When --use-token-cond is off, this is a fixed scalar used for all samples. "
                             "When on, the per-sample tcf is drawn log-uniformly on [--tcf-min, --tcf-max] "
                             "instead, and this flag only governs the default/eval tcf.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--value-approach", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-logvar", action=argparse.BooleanOptionalAction, default=False)

    # ── Optimal-control variant with iMF-style distillation (FTFMOCImFLoss, --use-imf-oc) ──
    parser.add_argument("--use-imf-oc", action=argparse.BooleanOptionalAction, default=False,
                        help="Use FTFMOCImFLoss (iMF-style OC, no data interpolant). "
                             "Takes precedence over --value-approach and --use-logvar.")
    parser.add_argument("--imf-oc-reward-scale", type=float, default=0.1,
                        help="Weight on reward term in iMF-OC loss. (FTFMLoss default 0.1.)")
    parser.add_argument("--imf-oc-lambda", type=float, default=1.0,
                        help="Weight λ_imf on the iMF mean-flow loss term.")
    parser.add_argument("--imf-oc-no-stopgrad", action=argparse.BooleanOptionalAction, default=False,
                        help="If set, do NOT stop-grad on JVP outcome inside V_θ. "
                             "iMF default is stopgrad ON; turning OFF allows second-order "
                             "gradient flow through JVP backward (per iMF §4.1, optimization "
                             "becomes harder but v_θ gets explicit JVP-tangent training signal).")
    parser.add_argument("--imf-oc-adaptive", action=argparse.BooleanOptionalAction, default=False,
                        help="Apply adaptive 1/loss weighting to L_iMF. iMF claims its loss "
                             "has stable variance, so adaptive weighting is unnecessary; "
                             "off by default. Enable to mimic legacy FTFMVALUEDMFLoss behavior.")

    # Token conditioning: (tcf, t_min, t_max) enter as prefix tokens, so inference can set the reward
    # strength and the times at which it applies.
    parser.add_argument("--use-token-cond", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable prefix-token conditioning on (tcf, t_min, t_max) in DMFT.")
    parser.add_argument("--num-tcf-tokens", type=int, default=4,
                        help="Number of learnable tokens representing tcf in the prefix.")
    parser.add_argument("--num-interval-tokens", type=int, default=2,
                        help="Number of learnable tokens representing each of t_min / t_max.")
    parser.add_argument("--tcf-min", type=float, default=0.1,
                        help="Log-uniform lower bound on per-sample tcf during training.")
    parser.add_argument("--tcf-max", type=float, default=25.0,
                        help="Log-uniform upper bound on per-sample tcf during training.")
    parser.add_argument("--tcf-drop-prob", type=float, default=0.1,
                        help="Probability of forcing tcf=0 (unconditional anchor) per sample.")
    parser.add_argument("--tcf-log-uniform", action=argparse.BooleanOptionalAction, default=False,
                        help="Sample tcf log-uniformly on [tcf_min, tcf_max]. When False, "
                             "samples uniformly (more samples at high tcf, stronger reward signal).")
    parser.add_argument("--tmin-max", type=float, default=0.5,
                        help="Upper bound of U[0, tmin_max] from which per-sample t_min is drawn.")
    parser.add_argument("--tmax-min", type=float, default=0.5,
                        help="Lower bound of U[tmax_min, 1] from which per-sample t_max is drawn.")
    parser.add_argument("--tcf-eval", type=float, default=5.0,
                        help="tcf value used when sweeping intervals in the training-time eval grid.")
    parser.add_argument("--num-tau-samples", type=int, default=1,
                        help="Multi-τ MC averaging for the reward gradient sub-graph. "
                             "K samples reduces variance ∝ 1/K, raising cos_sim(u, g_hat) and "
                             "therefore the tcf-conditioning incentive ∝ γ². K=1 is the baseline. "
                             "Cost: sub-graph B is K× slower (sequential implementation).")
    parser.add_argument("--rollout-steps", type=int, default=2, choices=[1, 2],
                        help="Reward rollout for the value-gradient loss. "
                             "2 = two flow-map jumps x_t→x_τ→x_0 with τ ~ U[0, t_val] (the reported recipe); "
                             "1 = one jump x_t→x_0.")
    parser.add_argument("--frozen-model-mode", type=str, default="ema",
                        choices=["ema", "active"],
                        help="Reference model used for the rollout forwards. "
                             "ema = EMA copy (the reported recipe). "
                             "active = live training model (vggflow-style; "
                             "implicit coupling, no EMA lag).")
    parser.add_argument("--anchor-weight", type=float, default=0.0,
                        help="MSE anchor loss strength on tcf=0 samples (forces student "
                             "off-diagonal at tcf=0 to match the frozen teacher). 0 disables. "
                             "Requires --tcf-drop-prob > 0 to actually have tcf=0 samples.")
    parser.add_argument("--distill-method", type=str, default="jvp",
                        choices=["jvp", "fd", "semigroup"],
                        help="Mean-flow self-distill objective. "
                             "jvp = Eulerian (torch.func.jvp, exact derivative — default). "
                             "fd = Lagrangian finite-difference (x fixed, ∂F/∂t only). "
                             "semigroup = PSD compositional consistency (no derivative).")
    parser.add_argument("--fd-eps", type=float, default=1e-3,
                        help="Time perturbation for fd distill method.")
    parser.add_argument("--jvp-tangent-clip-ratio", type=float, default=0.0,
                        help="Cap dv_dt_mf at k·||v_tgt_mf||. 0 disables; ~5 stabilizes "
                             "active mode at high LR. Only applies to distill-method=jvp.")
    parser.add_argument("--detach-lookahead", type=str, default="none",
                        choices=["none", "last", "all"],
                        help="DIVERSITY LEVER, ported from the T2I flag detach-lookahead-velocity. Cuts the Jacobian pullback in the REWARD lookahead ONLY; the cost term is untouched. With rollout_steps=2: none keeps both hops, last cuts the final hop, all cuts both so dx0/dxt = I which is the T2I-equivalent object. Default none leaves behaviour unchanged.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
