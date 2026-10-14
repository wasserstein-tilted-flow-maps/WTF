# Evaluation

`eval.py` at the repository root runs the evaluation harness of each benchmark with the protocol of the
paper fixed. Each run writes every metric and its full argument set to a JSON file.

```bash
python eval.py t2i/configs/wtf_t2i.yaml --ckpt <checkpoint.pt> --nfe 4         # text-to-image
python eval.py t2i/configs/wtf_t2i.yaml --base --nfe 50                       # base model, no LoRA
python eval.py imagenet/configs/wtf_imagenet.yaml --ckpt <checkpoint.pt> --nfe 1
```

The default output folder is `eval_runs/<run>_<checkpoint>_nfe<n>/`, and `--output-dir` overrides it.

## Text-to-image protocol

| | |
|---|---|
| prompts | `t2i/src/lib/assets/HPDv2/sample100_photo50_paint50.txt` (100 HPDv2 photo and painting prompts) |
| quality | 1,000 samples, prompts drawn with replacement from the 100 (seed 42); HPSv2, PickScore, ImageReward |
| diversity | the first 32 prompts × 16 samples each |
| sampling | 512×512, CFG 2.5, NFE 1, 4, 8 or 50 |
| output | `eval_results_nfe<n>.json`: `quality.<reward>.mean`, `diversity.<metric>_pairwise_l2sq.mean` |

Diversity is the mean pairwise squared L2 distance between the unit-normalized CLIP ViT-L/14 or DreamSim
embeddings of the 16 samples of a prompt, averaged over the 32 prompts.

**GPU count.** The harness runs under `torchrun`, and the noise of each sample depends on its GPU rank. A
result is reproduced exactly only with the same number of GPUs as the original evaluation. The reported WTF, baseline and
ablation rows were evaluated on 1 GPU, the default, and `--gpus` changes it:

```bash
python eval.py t2i/configs/wtf_t2i.yaml --ckpt wtf_t2i_step1460.pt --nfe 4 --gpus 1
```

**Baselines.** The Adjoint Matching and Flow-GRPO rows use the same compute budget as WTF and this harness. Their checkpoints are not released. To evaluate a LoRA checkpoint of another rank, pass `--lora-rank`. The harness refuses checkpoints whose LoRA shapes do not match the rank.

## ImageNet-256 protocol

| | |
|---|---|
| classes | the first 32 ImageNet classes, 16 class-conditional samples each (512 images) |
| sampling | 256×256, CFG 1.0, NFE 1 or 250 (Euler flow map sampler), seed 42, one GPU |
| quality | HPSv2, PickScore, ImageReward on the same 512 images, with the prompt "A high-resolution, high-quality photograph of a \<class name\>" |
| diversity | within-class mean pairwise squared L2 of CLIP ViT-L/14 and DreamSim embeddings, averaged over the 32 classes |
| weights | the EMA weights stored in the checkpoint |
| output | `pc.json`: `quality.{hps,pickscore,imagereward}_on_div_classes.mean`, `diversity.<metric>_pairwise_l2sq.mean`, and per-class values under `per_class` |

The noise is drawn class by class from one generator seeded with 42, so every checkpoint and NFE sees the same initial latents.

**Base model.** Evaluate the pre-trained DMF XL/2 network with the same protocol by passing it as the checkpoint:

```bash
python eval.py imagenet/configs/wtf_imagenet.yaml --ckpt $DMF_CKPT --nfe 1
```

**Baselines.** The Adjoint Matching, MFM and VFM rows of the ImageNet table are the values reported by those methods.

## Comparing runs

Before comparing two results, compare the `args` recorded in their JSON files: the prompt file, the number of
quality samples, the diversity sizes, the seed, the CFG scale and, for text-to-image, `world_size`.
