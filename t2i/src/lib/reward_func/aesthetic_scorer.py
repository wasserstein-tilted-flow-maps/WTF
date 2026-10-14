import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

import torch.distributed as dist
from lib.distributed import get_local_rank

import sys
if sys.version_info < (3, 9):
    from importlib_resources import files
else:
    from importlib.resources import files
ASSETS_PATH = files("lib.assets")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self, dtype, distributed=True):
        super().__init__()
        if distributed:
            if get_local_rank() == 0: # only download once
                self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            dist.barrier()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        self.mlp = MLP()
        state_dict = torch.load(
            ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth")
        )
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()


        self.OPENAI_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        self.OPENAI_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


    def __call__(self, images):
        device = next(self.parameters()).device

        # images should be in [0, 1]
        images = torch.nn.functional.interpolate(images, size=(224, 224), mode="bicubic")
        images = (images - self.OPENAI_CLIP_MEAN.to(device)) / self.OPENAI_CLIP_STD.to(device)
        inputs = {"pixel_values": images}

        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)