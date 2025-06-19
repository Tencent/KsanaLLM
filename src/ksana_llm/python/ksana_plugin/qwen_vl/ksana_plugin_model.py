# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import sys
from typing import List

import torch
from transformers import AutoConfig

import requests
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# parent_dir: ./KsanaLLM/src/ksana_llm/python/ksana_plugin
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from plugin_utils import free_cache, load_safetensors, get_module, build_trt, get_weight_map
from plugin_model import BaseVITModel


class VITModel(BaseVITModel):

    def __init__(self, model_path):
        # read config
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.precision = self.config.torch_dtype

        self.image_size = self.config.visual.get("image_size")
        self.output_dim = self.config.visual.get("output_dim")
        self.dim = 3
        self.token_num = 256

        # trt build config
        self.min_batch = 1
        self.opt_batch = 1
        self.max_batch = 4

        # Using torch infer does not require defining encode
        self.image_pre_obj = None

        # Initialize and set the model device, assume on GPU
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

    def get_preprocess(self):
        if self.image_pre_obj is not None:
            return
        self.image_pre_obj = Preprocss(self.image_size)

    def get_model(self, model_path, precision=None):
        if precision is None:
            precision = self.precision

        vision_transformer = get_module("VITTorch", f'{model_path}/visual.py', "VisionTransformer")
        visual = vision_transformer(**self.config.visual)

        # read weight
        weight_map_files = get_weight_map(model_path, "transformer.visual")
        visual_weights = {}
        for weight_map_file in weight_map_files:
            weight_file = os.path.join(model_path, weight_map_file)
            if os.path.splitext(weight_file)[1] == ".safetensors":
                weights = load_safetensors(weight_file)
            else:
                weights = torch.load(weight_file, map_location=torch.device('cpu'))
            for name, tensor in weights.items():
                if "transformer.visual." in name:
                    visual_weights[name.replace("transformer.visual.", "")] = tensor

        # assign gpu and precision
        visual = visual.to(dtype=precision)
        visual.load_state_dict(visual_weights)
        visual = visual.to(device=self.device)

        free_cache()
        return visual


class Preprocss:

    def __init__(self, image_size: int):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def encode(self, image_paths: List[str]):
        images = []
        for image_path in image_paths:
            try:
                if image_path.startswith("http://") or image_path.startswith(
                        "https://"):
                    image = Image.open(requests.get(image_path, stream=True).raw)
                else:
                    image = Image.open(image_path)
                image = image.convert("RGB")
                images.append(self.image_transform(image))
            except Exception:  # pylint: disable=broad-except
                image = torch.zeros((3, 448, 448))
                images.append(image)

        images = torch.stack(images, dim=0)
        return images


if __name__ == '__main__':

    model_path = sys.argv[1]
    model = VITModel(model_path)

    build_trt(model, model_path)
