import os
from PIL import Image
import io
import numpy as np
import time
import requests

import torch
import torch.distributed as dist
from ..distributed import get_local_rank
from lib.utils import freeze


short_names = {
    "aesthetic_score": "aes",
    "imagereward": "imgr",
    "hpscore": "hps",
    "pickscore": "pick"
}
use_prompt = {
    "aesthetic_score": False,
    "imagereward": True,
    "hpscore": True,
    "pickscore": True
}

def aesthetic_score(dtype=torch.float32, device="cuda", distributed=True):
    from lib.reward_func.aesthetic_scorer import AestheticScorer
    scorer = AestheticScorer(dtype=torch.float32, distributed=distributed).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            pass  # assume float tensor in [0, 1]
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn

# For ImageReward
def imagereward(dtype=torch.float32, device="cuda"):
    # transformers >=4.5x relocated a group of helpers (apply_chunking_to_forward,
    # find_pruneable_heads_and_indices, prune_linear_layer, Conv1D, ...) from modeling_utils to
    # pytorch_utils. ImageReward still imports them from the old location, so on transformers
    # 4.57.6 `import ImageReward` raises ImportError and any eval requesting imagereward dies.
    # Alias the whole moved set back rather than one symbol at a time.
    import transformers.modeling_utils as _mu, transformers.pytorch_utils as _pu
    for _n in dir(_pu):
        if not _n.startswith("_") and not hasattr(_mu, _n):
            setattr(_mu, _n, getattr(_pu, _n))

    import ImageReward as RM
    from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
    try:
        from torchvision.transforms import InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC
    except ImportError:
        BICUBIC = Image.BICUBIC

    if get_local_rank() == 0:  # only download once
        reward_model = RM.load("ImageReward-v1.0")
    dist.barrier()
    reward_model = RM.load("ImageReward-v1.0")
    reward_model.to(dtype).to(device)

    rm_preprocess = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _fn(images, prompts, metadata):
        dic = reward_model.blip.tokenizer(prompts,
                padding='max_length', truncation=True,  return_tensors="pt",
                max_length=reward_model.blip.tokenizer.model_max_length) # max_length=512
        device = images.device
        input_ids, attention_mask = dic.input_ids.to(device), dic.attention_mask.to(device)
        reward = reward_model.score_gard(input_ids, attention_mask, rm_preprocess(images)) # differentiable
        return reward.reshape(images.shape[0]).float(), {} # bf16 -> f32

    return _fn


# For HPSv2 reward
# https://github.com/tgxs002/HPSv2/blob/master/hpsv2/img_score.py
def hpscore(dtype=torch.float32, device=torch.device('cuda')):
    import huggingface_hub
    import torchvision.transforms.functional as F
    from torchvision.transforms import Compose  # missing in upstream vggflow rewards.py
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    from hpsv2.utils import root_path, hps_version_map
    from hpsv2.src.open_clip.transform import MaskAwareNormalize, ResizeMaxSize

    hps_version = "v2.1"
    model_dict = {}
    if not model_dict:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )

    # initialize_model()
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to(device)
    model.eval()

    def _fn(images, prompts, metadata):
        image_size = model.visual.image_size[0]
        transforms = Compose([
            ResizeMaxSize(image_size, fill=0),
            MaskAwareNormalize(mean=model.visual.image_mean, std=model.visual.image_std),
        ])

        images = torch.stack([transforms(img) for img in images])
        texts = tokenizer(prompts).to(device=device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=dtype):
            outputs = model(images, texts)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits_per_image = image_features @ text_features.T # (bs, bs)
            hps_score = torch.diagonal(logits_per_image) # (bs,)

        return hps_score, {}

    return _fn

# For HPSv2 reward
# https://github.com/tgxs002/HPSv2/blob/master/hpsv2/img_score.py
def pickscore(dtype=torch.float32, device=torch.device('cuda')):
    import huggingface_hub
    from torchvision.transforms import Compose  # missing in upstream vggflow rewards.py
    from transformers import AutoProcessor, AutoModel
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    from hpsv2.src.open_clip.transform import MaskAwareNormalize, ResizeMaxSize

    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)
    tokenizer = get_tokenizer('ViT-H-14')

    def _fn(images, prompts, metadata):
        image_size = 224
        transforms = Compose([
            ResizeMaxSize(image_size, fill=0),
            MaskAwareNormalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ])

        images = torch.stack([transforms(img) for img in images])
        texts = tokenizer(prompts).to(device=device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=dtype):
            image_embs = model.get_image_features(images)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = model.get_text_features(texts)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
            pick_score = model.logit_scale.exp() * (image_embs * text_embs).sum(dim=1)
            assert len(pick_score.shape) == 1

        return pick_score, {}

    return _fn
