# Setup

## Environment

One environment serves both benchmarks: Python 3.11, CUDA 12.1, PyTorch 2.5.1.

```bash
conda create -n wtf python=3.11 -y && conda activate wtf
pip install -r requirements.txt
pip install --no-deps -r requirements-nodeps.txt
# hpsv2 1.2.0 ships without its tokenizer vocabulary; copy it from open_clip
python -c "import os, shutil, hpsv2, open_clip; shutil.copy(os.path.join(os.path.dirname(open_clip.__file__), 'bpe_simple_vocab_16e6.txt.gz'), os.path.join(os.path.dirname(hpsv2.__file__), 'src', 'open_clip'))"
```

`requirements.txt` installs PyTorch first and then a prebuilt `flash-attn` 2.7.3 wheel built against it. If
your platform needs a different wheel, install `flash-attn` after PyTorch from the
[FlashAttention releases](https://github.com/Dao-AILab/flash-attention/releases).

`hpsv2` 1.2.0 and `image-reward` 1.5 pin old versions of `protobuf`, `pytest` and `timm` that conflict with the
rest of the stack, so `requirements-nodeps.txt` installs them (and `fairscale`) without their dependencies. The runtime dependencies they need are in `requirements.txt`. The last command restores a tokenizer file that the
`hpsv2` 1.2.0 wheel is missing.

The reward and metric packages `hpsv2`, `image-reward`, `open_clip_torch`, `dreamsim` and `imscore` are pinned because they define the numbers in the paper. Check the installed versions with

```bash
pip show hpsv2 image-reward open_clip_torch dreamsim imscore | grep -E "^(Name|Version)"
```

## Text-to-image substrate: TiM-T2I

The text-to-image experiments fine-tune [TiM-T2I](https://github.com/WZDTHU/TiM), a 865M-parameter
text-to-image flow map. It needs the following components:

| component | source | how it is found |
|---|---|---|
| TiM code, commit `d01ef74` | `git clone https://github.com/WZDTHU/TiM` | `PYTHONPATH` |
| TiM-T2I weights | [`GoodEnough/TiM-T2I/t2i_model.bin`](https://huggingface.co/GoodEnough/TiM-T2I) | `TIM_CKPT` |
| text encoder | [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) (gated) | TiM config, then the Hub |
| autoencoder | [`mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers`](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers) | TiM config, then the Hub |

```bash
git clone https://github.com/WZDTHU/TiM && git -C TiM checkout d01ef74   # the TiM version of the paper
wget -c https://huggingface.co/GoodEnough/TiM-T2I/resolve/main/t2i_model.bin -O TiM/t2i_model.bin

export TIM_REPO=$PWD/TiM PYTHONPATH=$PWD/TiM
export TIM_CKPT=$TIM_REPO/t2i_model.bin
export TIM_CONFIG=$TIM_REPO/configs/t2i/tim_xl_p1_t2i.yaml
```

Use commit `d01ef74`.

TiM's config names the text encoder and autoencoder by local paths (`checkpoints/gemma-3-1b-it`,
`checkpoints/dc-ae-f32c32-sana-1.1-diffusers`). If those folders exist in the working directory or under
`$TIM_REPO`, they are used, and otherwise both are loaded from the Hugging Face Hub. Gemma-3 is a gated model, so accept its license on its Hugging Face page and run `huggingface-cli login` once.

The text-to-image lane must use `substrate.attn_type: fused_attn` (the shipped setting).

## ImageNet substrate: Decoupled MeanFlow XL/2

The ImageNet experiments fully fine-tune the [Decoupled MeanFlow](https://github.com/kyungmnlee/dmf)
XL/2 flow map at 256 resolution, decoded with the SD-VAE.

```bash
huggingface-cli download kyungmnlee/DMF dmf_xl_2_256.pt --local-dir checkpoints
export DMF_CKPT=$PWD/checkpoints/dmf_xl_2_256.pt
```

`imagenet/configs/wtf_imagenet.yaml` reads `substrate.pretrained: ${DMF_CKPT}`, and `run.py` requires it.

**No ImageNet data is needed.** WTF rolls every training sample out from noise, so the ImageNet lane trains
on random class labels only (`--synthetic-data`, on by default). The WebDataset loader in `train.py` is used only with `--no-synthetic-data`.

## Rewards and metrics

All models below download automatically on first use and are cached by Hugging Face.

| model | role |
|---|---|
| HPSv2.1 (`xswu/HPSv2`, OpenCLIP ViT-H-14) | training reward and evaluation |
| PickScore (`yuvalkirstain/PickScore_v1`) | evaluation |
| ImageReward (`THUDM/ImageReward`) | evaluation |
| CLIP ViT-L/14 (`openai/clip-vit-large-patch14`) | diversity |
| DreamSim (ensemble) | diversity |

## Hardware and run time

All reported runs used one node with 8×H100 80GB.

| run | steps | time |
|---|---|---|
| text-to-image training | 1,460 | about 6 hours (15.1 s per step) |
| ImageNet training | 12,500 | about 3 hours |
| text-to-image evaluation, one NFE, one GPU | 1,000 + 512 images | a few minutes at NFE 1 to 8, about 40 minutes at NFE 50 |
| ImageNet evaluation, one NFE, one GPU | 512 images | about 3 minutes at NFE 1, about 12 minutes at NFE 250 |



## Logging

Training logs to [Weights & Biases](https://wandb.ai). Set `WANDB_MODE=offline` to keep logs local, or
`WANDB_MODE=disabled` to turn logging off.
