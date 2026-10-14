# Configuration

Training is launched from a YAML config through `run.py`, which maps each key onto the training CLI of the
benchmark (`t2i/src/train.py` or `imagenet/src/train.py`) and starts it under `torchrun` (text-to-image) or
`accelerate` (ImageNet).

```bash
python run.py t2i/configs/wtf_t2i.yaml                        # launch
python run.py t2i/configs/wtf_t2i.yaml --dry-run              # print the command only
python run.py t2i/configs/wtf_t2i.yaml --set objective.distill_weight=1.0 --set optim.seed=5
python run.py t2i/configs/wtf_t2i.yaml --gpus 4
```

- The benchmark is the top-level folder of the config (`t2i/` or `imagenet/`). A config stored elsewhere
  can name it with a top-level `lane: t2i` or `lane: imagenet` key.
- A config can inherit from another with `defaults: <relative path>` and override some keys, as the ablations under `*/configs/ablations/` do.
- `${VAR}` in a value is expanded from the environment. `run.py` refuses to launch when a substrate path
  is unset (`TIM_CKPT` and `TIM_CONFIG`, or `DMF_CKPT`).
- `run.py` lists unknown keys as `ignored keys` at launch.
- Every run writes its full argument set to `args.json` in its output folder. Compare these files, and the
  references in `reference/`, before comparing results.

## Config groups

| group | text-to-image | ImageNet |
|---|---|---|
| `substrate` | `ckpt`, `config` (TiM), `height`, `width`, `attn_type`, `reward` | `pretrained` (DMF), `model`, `dmf_depth`, `resolution` |
| `adaptation` | `lora_rank`, `lora_alpha`, `dual_adapter` | (full fine-tuning) |
| `objective` | `target_control_fraction`, `distill_weight`, `distill_mode`, `distill_method`, `eta_mode` | `target_control_fraction`, `diag_prob`, `anchor_weight` |
| `estimator` | `detach`, `rollout_steps`, `cost_at_tau`, `quantile_clip_pct`, `reward_jitter_pairs`, `reward_jitter_sigma`, `cfg_scale`, `no_w_rew_norm`, ... | `detach`, `num_tau_samples`, `use_logvar`, `use_token_cond` |
| `guidance` | | substrate guidance settings (`g_type`, `omega`, `g_min`, `g_max`, `cfg_prob`) |
| `optim` | `lr`, `batch_size` (per GPU), `grad_accum`, `num_iters`, `checkpoint_steps`, `seed`, `gpus`, ... | `lr`, `batch_size` (global), `max_train_steps`, `checkpointing_steps`, `ema_decay`, `seed`, `exp_name`, ... |
| `hjb` | optional HJB terms, all zero in the shipped recipe | |
| `runtime` | | `attn_func`, `num_workers` |

The shipped configs, `t2i/configs/wtf_t2i.yaml` and `imagenet/configs/wtf_imagenet.yaml`, are the recipes
of the reported runs. Their comments describe each value.

## The hyperparameters that matter

**Reward scale λ (`objective.target_control_fraction`).** The coefficient of the reward in the value
estimate. Both shipped recipes normalize it by running averages of the base velocity and reward-gradient
norms,
$w_{\mathrm{rew}} = \lambda \cdot \mathrm{EMA}\lVert b\rVert / \mathrm{EMA}\lVert\nabla r\rVert$,
so one value transfers across rewards of different scales. The reported values are λ = 30 (text-to-image)
and λ = 15 (ImageNet). Text-to-image can apply λ directly instead with `estimator.no_w_rew_norm: true`.

**Reward gradient (`estimator.detach`).** How much of the flow map Jacobian enters the value gradient.

| value | gradient | benchmarks |
|---|---|---|
| `all` | Euclidean, $\nabla r(\bar x_1)$ (identity Jacobian); the reported recipe | both |
| `none` | exact, pulled back through the rollout | both |
| `last` | pulled back through the first rollout hop only | ImageNet (two-hop rollout) |

The text-to-image ablation `t2i/configs/ablations/exact_gradient.yaml` and the ImageNet ablations
`detach_none.yaml` and `detach_last.yaml` switch this setting.

**Distillation weight β (`objective.distill_weight`, text-to-image).** The weight of the flow map
self-distillation term (0.5 in the reported recipe). It trades diversity against reward. The ImageNet loss weights its
distillation term adaptively and has no β.

**Rollout (`estimator.rollout_steps`, `estimator.cost_at_tau`).** Text-to-image reaches the endpoint with one
flow map jump from $x_t$ and, with `cost_at_tau: true`, evaluates the running cost at a uniformly sampled
$\tau$, at $x_\tau$ reached by one jump from $x_t$ (`false` evaluates it at $t$). ImageNet uses two jumps
through the sampled $\tau$, with the cost at $\tau$. With one $\tau$ sample per step, `num_tau_samples` has no
effect in either loss.

**Reward-gradient smoothing (text-to-image).** `reward_jitter_pairs: 1` and `reward_jitter_sigma: 0.2`
average $\nabla r$ over one antithetic pair of perturbed states at scale 0.2·std($x_t$), and
`quantile_clip_pct: 0.8` clips each per-sample gradient norm at the 0.8 quantile of the batch.

**Training length.** The reported checkpoints are step 1,460 (text-to-image) and step 12,500 (ImageNet).


## Settings kept only for argument parity

Some keys in the shipped ImageNet config have no effect on the WTF objective but are kept so that a new run's
`args.json` matches the reported runs: `objective.diag_prob`, `estimator.num_tau_samples` and the `guidance`
group. `estimator.use_logvar: true` must stay on, since it selects the WTF objective in `imagenet/src/loss.py`.
