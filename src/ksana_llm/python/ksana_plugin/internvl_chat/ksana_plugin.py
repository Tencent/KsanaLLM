# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import sys

import importlib
from typing import List
import torch
from transformers import AutoTokenizer
from transformers import AutoConfig

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(current_dir)
sys.path.append(parent_dir)

from internvl_chat.InternVL.ksana_plugin_model import IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN
from plugin_utils import adjust_device_memory_ratio


class KsanaPlugin:
    """
    Define a class named KsanaPlugin
    """
    def __init__(self):
        self.messages = []

    # Plugin initialization is automatically invoked upon service startup.
    def init_plugin(self, **kwargs):
        if "preprocess" in kwargs:
            self.tokenizer = AutoTokenizer.from_pretrained(kwargs['model_path'], trust_remote_code=True, use_fast=False)
            self.config = AutoConfig.from_pretrained(kwargs['model_path'], trust_remote_code=True).to_dict()
            image_size = self.config['force_image_size'] or self.config['vision_config']['image_size']
            if kwargs['vit_model_type'] == "InternVL2_5":
                from internvl_chat.InternVL2_5.ksana_plugin_model import InternVLTwoPointFivePreprocess
                self.preprocessor = InternVLTwoPointFivePreprocess(kwargs['model_path'], image_size)
            else:
                from internvl_chat.InternVL.ksana_plugin_model import InternVLPreprocess
                self.preprocessor = InternVLPreprocess(kwargs['model_path'], image_size)
            sys.path.append(kwargs['model_path'])
            self.conv_module = importlib.import_module('conversation')
            self.conv_template = self.conv_module.get_conv_template(self.config['template'])        
            self.system_message = self.conv_template.system_message
            patch_size = self.config['vision_config']['patch_size']
            self.num_image_token = int((image_size // patch_size) ** 2 * (self.config['downsample_ratio'] ** 2))
            print(f'self.num_image_token: {self.num_image_token}')
            self.hidden_size = self.config['llm_config']['hidden_size']
            self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)            
            adjust_device_memory_ratio(kwargs["config_file"], 0.15)

                
    # Method for pre-processing
    def preprocess(self, **kwargs):
        if not self.check_input(['ksana_python_input', 'messages'], **kwargs):
            print(f"[E] Check input failed.")
            return
        # Generate internvl template when there is no LLMServing
        if os.getenv("KSANA_HTTP_PORT") is None:
            self.messages = kwargs['messages'][0]
            pixel_values, user_text = self.preprocessor.set_image_message(self.messages)
            print(f'pixel_values shape: {pixel_values.shape}')
            question = user_text[0]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
            template = self.conv_module.get_conv_template(self.config['template'])
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            for num_patches in num_patches_list:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)
            inputs = self.tokenizer(query, return_tensors='pt')        
            input_token_ids = inputs["input_ids"]
        else:
            self.messages = kwargs['messages'][0]
            pixel_values = self.preprocessor.get_images(self.messages)
            input_token_ids = kwargs['ksana_python_input'].input_tokens

        vit_embeds = None
        visual_features = None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.preprocessor.extract_feature(pixel_values)
        if isinstance(input_token_ids, list):
            input_token_ids = torch.tensor(input_token_ids)
        input_ids_squeeze = input_token_ids.squeeze().cpu()
        ksana_python_input = kwargs['ksana_python_input']
        ksana_python_input.input_tokens = input_ids_squeeze.tolist()
        if vit_embeds is not None:
            vit_embeds = vit_embeds.reshape(-1, self.hidden_size)
            vision_srt = [int(pos + 1) for pos, id in enumerate(input_ids_squeeze) if id == self.img_context_token_id]
            ksana_python_input.input_refit_embedding.pos = vision_srt
            ksana_python_input.input_refit_embedding.embedding_tensors = torch.unbind(vit_embeds.cpu().float())
            
    # Method for post-processing
    def postprocess(self, **kwargs):
        return

    def check_input(self, input_list: List[str], **kwargs) -> bool:
        for input_name in input_list:
            if input_name not in kwargs:
                print(f"[E] Input {input_name} not found.")
                return False
        return True
