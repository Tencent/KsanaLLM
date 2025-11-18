# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import sys

from transformers import AutoConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

import torch

# parent_dir: ./KsanaLLM/src/ksana_llm/python/ksana_plugin
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from plugin_utils import free_cache, load_safetensors, get_weight_map


class VITModel:

    def __init__(self, model_path):
        # read config
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.precision = self.config.torch_dtype

        # Initialize the model device, assume on GPU
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

    def get_model(self, model_path, precision=None):
        if precision is None:
            precision = self.precision

        # init vit model
        # NOTE: qwen2.5_vl differs from qwen2_vl only in the visual encoder.
        # We therefore branch here: qwen2.5_vl uses its own VisionTransformer
        # (Qwen2_5_VisionTransformerPretrainedModel), while every other
        # qwen2_vl variant keeps the original Qwen2VisionTransformerPretrainedModel.
        if self.config.model_type == "qwen2_5_vl":
            visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(
                self.config.vision_config,
                torch_dtype=precision)
        else:
            visual = Qwen2VisionTransformerPretrainedModel._from_config(
                self.config.vision_config,
                attn_implementation="flash_attention_2",
                torch_dtype=precision)

        # read weight
        weight_map_files = get_weight_map(model_path, "visual.")
        visual_weights = {}
        for weight_map_file in weight_map_files:
            weight_file = os.path.join(model_path, weight_map_file)
            if os.path.splitext(weight_file)[1] == ".safetensors":
                weights = load_safetensors(weight_file)
            else:
                weights = torch.load(weight_file, map_location=torch.device('cpu'))
            for name, tensor in weights.items():
                if "visual." in name:
                    visual_weights[name.replace("visual.", "")] = tensor

        # assign gpu and precision
        visual = visual.to(dtype=precision)
        visual.load_state_dict(visual_weights)
        visual = visual.to(device=self.device)
        visual.get_dtype = lambda: next(visual.parameters()).dtype

        free_cache()
        return visual
