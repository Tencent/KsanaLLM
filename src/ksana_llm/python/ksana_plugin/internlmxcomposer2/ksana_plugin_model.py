# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import sys
import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

# parent_dir: ./KsanaLLM/src/ksana_llm/python/ksana_plugin
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from plugin_utils import free_cache, load_safetensors, get_module, build_trt, get_weight_map
from plugin_model import BaseVITModel, CUDA_0


class IXCModel:
    def __init__(self, model_path):
        # read config
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # 为了对齐 with torch.autocast(device_type='cuda', dtype=torch.float16
        # precision = self.config.torch_dtype
        self.precision = torch.float16
        self.max_length = self.config.max_length

    def get_processor(self, model_path):
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def get_model(self, model_path, precision=None, is_trt=False):
        if precision is None:
            precision = self.precision

        # get ixc model
        internlm_img = get_module("IXCTorch", f'{model_path}/visual.py', "InternLMXComposerImg")
        ixc = internlm_img(model_path)

        # get weight files
        weight_map_files = get_weight_map(model_path)
        def get_weights(weight_map_files, filter_key, is_replace=False):
            model_weights = {}
            # get weight files
            filtered_values = {value for key, value in weight_map_files.items() if filter_key in key}
            weight_map_files = list(filtered_values)
            # read weight
            for weight_map_file in weight_map_files:
                weight_file = os.path.join(model_path, weight_map_file)
                if os.path.splitext(weight_file)[1] == ".safetensors":
                    weights = load_safetensors(weight_file)
                else:
                    weights = torch.load(weight_file, map_location=torch.device('cpu'))
                for name, tensor in weights.items():
                    if filter_key in name:
                        if is_replace:
                            model_weights[name.replace(filter_key, "")] = tensor
                        else:
                            model_weights[name] = tensor
            return model_weights

        # assign gpu and precision
        ixc.plora_glb_GN.to(dtype=precision)
        ixc.plora_sub_GN.to(dtype=precision)
        ixc.load_state_dict(get_weights(weight_map_files, "plora_", False))
        ixc.plora_glb_GN.to(CUDA_0)
        ixc.plora_sub_GN.to(CUDA_0)

        #ixc.vision_proj.to(dtype=precision)
        ixc.vision_proj.load_state_dict(get_weights(weight_map_files, "vision_proj.",  True))
        ixc.vision_proj.to(CUDA_0)

        ixc.vit.to(dtype=precision)
        if not is_trt:
            ixc.vit.load_state_dict(get_weights(weight_map_files, "vit.", True))
            ixc.vit.to(CUDA_0).eval()

        free_cache()
        return ixc


class VitInferWrapper(nn.Module):
    def __init__(self, clip_vision_tower):
        super().__init__()
        self.clip_vision_tower = clip_vision_tower

    def forward(self, input_imgs):
        return self.clip_vision_tower.vit_infer(input_imgs)


class VITModel(BaseVITModel):
    def __init__(self, model_path):
        # read config
        vision_tower = model_path + '/internlm-xcomposer2d5-clip/'
        self.config = AutoConfig.from_pretrained(vision_tower, trust_remote_code=True)
        self.precision = self.config.torch_dtype

        self.image_size = self.config.image_size   # 560
        self.output_dim = self.config.hidden_size  # 1024
        self.dim = 3
        self.token_num = 1600

        # trt build config
        self.min_batch = 1
        self.opt_batch = 6
        self.max_batch = 10

    def get_model(self, model_path, precision=None):
        if precision is None:
            precision = self.precision

        clip_vision_tower = get_module("VITTorch", f'{model_path}/visual.py', "CLIPVisionTower")
        vision_tower = model_path + '/internlm-xcomposer2d5-clip/'
        visual = clip_vision_tower(vision_tower)

        # read weight
        weight_map_files = get_weight_map(model_path, "vit.")
        visual_weights = {}
        for weight_map_file in weight_map_files:
            weight_file = os.path.join(model_path, weight_map_file)
            if os.path.splitext(weight_file)[1] == ".safetensors":
                weights = load_safetensors(weight_file)
            else:
                weights = torch.load(weight_file, map_location=torch.device('cpu'))
            for name, tensor in weights.items():
                if "vit." in name:
                    visual_weights[name.replace("vit.", "")] = tensor

        # assign gpu and precision
        visual = visual.to(dtype=precision)
        visual.load_state_dict(visual_weights)
        visual = visual.to(device=CUDA_0)

        vit_infer_wrapper = VitInferWrapper(visual)
        vit = vit_infer_wrapper.eval()

        free_cache()
        return vit


if __name__ == '__main__':

    model_path = sys.argv[1]
    model = VITModel(model_path)

    build_trt(model, model_path)
