# Reproducing the paper

A step-by-step checklist for every WTF number in the paper, with what to run, what it writes and what to expect.
It assumes the environment and substrates of [`docs/SETUP.md`](docs/SETUP.md).

## 0. Before spending GPU time

```bash
export TIM_REPO=/path/to/TiM PYTHONPATH=/path/to/TiM            # TiM at commit d01ef74
export TIM_CKPT=$TIM_REPO/t2i_model.bin TIM_CONFIG=$TIM_REPO/configs/t2i/tim_xl_p1_t2i.yaml
export DMF_CKPT=/path/to/dmf_xl_2_256.pt

python run.py t2i/configs/wtf_t2i.yaml --dry-run             # must print no "ignored keys" or "warning" line
python run.py imagenet/configs/wtf_imagenet.yaml --dry-run    # likewise
```

Every training run writes its arguments to `args.json`. They must equal the saved arguments of the reported
runs in `reference/` on every key except paths, output and logging locations, `seed` and `exp_name` (ImageNet),
and `num_iters`, `checkpoint_steps` and `cost_at_tau` (text-to-image). The text-to-image reference run trained for 1,800
iterations with a checkpoint every 10, and the learning rate is constant after warmup, so steps 1 to 1,460 are
identical. For `cost_at_tau`, see `t2i/configs/wtf_t2i.yaml`.

## 1. Evaluate the released checkpoints

With the [released checkpoints](README.md#checkpoints), every reported WTF number takes one evaluation.
Text-to-image evaluations draw each sample's noise from its GPU rank, so use the GPU counts below to
reproduce the numbers exactly ([details](docs/EVALUATION.md)).

```bash
# text-to-image table, WTF rows (1 GPU)
for n in 1 4 8 50; do
  python eval.py t2i/configs/wtf_t2i.yaml --ckpt checkpoints/wtf_t2i_step1460.pt --nfe $n --gpus 1
done
# text-to-image table, base model (1 GPU)
for n in 1 4 8 50; do python eval.py t2i/configs/wtf_t2i.yaml --base --nfe $n --gpus 1; done

# ImageNet table (one GPU each)
for s in 0 1 2; do for n in 1 250; do
  python eval.py imagenet/configs/wtf_imagenet.yaml --ckpt checkpoints/wtf_imagenet_seed${s}_step12500.pt --nfe $n
done; done
```

## 2. Text-to-image table

Train on 8×H100 for about 6 hours. Checkpoints are saved every 10 steps, and the reported one is the last, step 1,460.

```bash
python run.py t2i/configs/wtf_t2i.yaml                        # -> exps/wtf_t2i/checkpoints/0001460.pt
for n in 1 4 8 50; do
  python eval.py t2i/configs/wtf_t2i.yaml --ckpt exps/wtf_t2i/checkpoints/0001460.pt --nfe $n --gpus 1
done
```

Expected (`eval_results_nfe<n>.json`: `quality.<reward>.mean`, `diversity.<metric>_pairwise_l2sq.mean`):

| NFE | HPSv2 | PickScore | ImageReward | DreamSim div. | CLIP div. |
|---|---|---|---|---|---|
| 1 | 0.347 | 22.31 | 1.069 | 0.325 | 0.289 |
| 4 | 0.401 | 22.97 | 1.383 | 0.250 | 0.220 |
| 8 | 0.400 | 22.93 | 1.394 | 0.233 | 0.212 |
| 50 | 0.398 | 22.87 | 1.416 | 0.217 | 0.205 |

The base model scores HPSv2 0.274 at NFE 50 under the same protocol.

## 3. Text-to-image reward against compute

```bash
python t2i/scripts/reward_curve.py t2i/configs/wtf_t2i.yaml --run exps/wtf_t2i
```

Evaluates every saved checkpoint at the curve protocol (NFE 50, 200 samples, HPSv2 only) into
`exps/wtf_t2i/reward_curve.jsonl` and plots `reward_curve.png`, with GPU-hours from the measured 15.087 s per
step on 8 GPUs. The Adjoint Matching and Flow-GRPO curves come from their own training runs.

## 4. ImageNet table

Three training runs that differ only in the seed (8×H100, about 3 hours each):

```bash
for s in 0 1 2; do
  python run.py imagenet/configs/wtf_imagenet.yaml --set optim.seed=$s --set optim.exp_name=wtf_imagenet_s$s
done                                                          # -> exps/wtf_imagenet_s<s>/checkpoints/0012500.pt
for s in 0 1 2; do for n in 1 250; do
  python eval.py imagenet/configs/wtf_imagenet.yaml --ckpt exps/wtf_imagenet_s$s/checkpoints/0012500.pt --nfe $n
done; done                                                    # -> eval_runs/wtf_imagenet_s<s>_0012500_nfe<n>/pc.json
```

Expected, as reported in the paper (`pc.json`: `quality.<reward>_on_div_classes.mean`,
`diversity.<metric>_pairwise_l2sq.mean`):

| NFE | HPSv2 | PickScore | ImageReward | DreamSim div. | CLIP div. |
|---|---|---|---|---|---|
| 1 | 0.319 | 20.24 | 0.612 | 0.482 | 0.201 |
| 250 | 0.330 | 20.45 | 0.718 | 0.460 | 0.177 |

Reference values of the three released checkpoints, to check a single run:

| seed | NFE | HPSv2 | PickScore | ImageReward | DreamSim div. | CLIP div. |
|---|---|---|---|---|---|---|
| 0 | 1 | 0.3186 | 20.216 | 0.623 | 0.4826 | 0.2098 |
| 1 | 1 | 0.3194 | 20.274 | 0.620 | 0.4832 | 0.1992 |
| 2 | 1 | 0.3203 | 20.235 | 0.594 | 0.4797 | 0.1948 |
| 0 | 250 | 0.3302 | 20.446 | 0.718 | 0.4603 | 0.1809 |
| 1 | 250 | 0.3296 | 20.461 | 0.685 | 0.4625 | 0.1781 |
| 2 | 250 | 0.3313 | 20.460 | 0.686 | 0.4575 | 0.1714 |

## 5. Checking results

Evaluating a released checkpoint with the GPU counts above reproduces the reported numbers. For a retrained
run, compare its `args.json` against `reference/` and the `args` in the evaluation JSON against
[`docs/EVALUATION.md`](docs/EVALUATION.md).
