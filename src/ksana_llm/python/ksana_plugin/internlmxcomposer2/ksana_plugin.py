# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
import sys
import os
from typing import List

import torch

# parent_dir: ./KsanaLLM/src/ksana_llm/python/ksana_plugin
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from internlmxcomposer2.ksana_plugin_model import IXCModel
from plugin_utils import adjust_device_memory_ratio, build_trt_process


class KsanaPlugin:
    """
    Define a class named KsanaPlugin
    """
    def __init__(self):
        pass

    # Plugin initialization is automatically invoked upon service startup.
    def init_plugin(self, **kwargs):
        if "preprocess" in kwargs:
            model_path = kwargs["model_path"]
            enable_trt = kwargs.get('enable_trt', True)

            # Initializing a model instance
            ixc_model = IXCModel(model_path)

            self.max_length = ixc_model.max_length
            self.tokenizer = ixc_model.get_processor(model_path)

            self.trt = False
            if enable_trt:
                try:
                    self._init_trt(model_path)
                    self.trt = True
                    print(f"[I] Initializing the TensorRT model successfully!")
                except Exception as e:  # pylint: disable=broad-except
                    print(f"[E] Failed to initialize TensorRT model : {e}")

            self.ixc = ixc_model.get_model(model_path, is_trt=self.trt)

            if self.trt:
                # hack vit_infer
                self.ixc.vit.vit_infer = self._infer_trt.__get__(self.ixc.vit)

            adjust_device_memory_ratio(kwargs["config_file"], 0.08 if self.trt else 0.15)

            # Ensure the result is a dictionary
            return {
                       'plugin_trt' : self.trt,
                   }

        if "postprocess" in kwargs:
            return

    # Method for pre-processing
    def preprocess(self, **kwargs):
        if not self.check_intput(['ksana_python_input', 'messages'], **kwargs):
            raise RuntimeError(f"Plugin preprocess wrong input.")
        messages: Optional[List[Dict]] = kwargs['messages']
        if messages is None:
            return
        prompt, images = KsanaPlugin.convert_openai_messages(messages)
        ksana_python_input = kwargs['ksana_python_input']

        # 1.添加模版
        prompt = f"""[UNUSED_TOKEN_146]user\n{prompt}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"""
        image_nums = len(images)
        if image_nums == 0:
            return 
        elif image_nums == 1 and prompt.find('<ImageHere>') == -1:
            prompt = '<ImageHere>' + prompt

        parts = prompt.split('<ImageHere>')
        need_bos = True
        input_tokens = []
        wrap_embeds = []
        image_pad_id = 0
        url_srt = []

        # 2.开始解析
        for idx, part in enumerate(parts):
            if len(part) > 0:
                part_tokens = self.tokenizer(
                    part,
                    padding='longest',
                    add_special_tokens=need_bos)
                input_tokens.extend(part_tokens.input_ids)
                if need_bos:
                    need_bos = False
            if idx < image_nums:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    img = self.ixc.encode_img(images[idx])
                    # img = self.ixc.encode_img(images[idx], hd_num=16, num_frames=8) 
                torch.cuda.synchronize()
                url_srt.append(len(input_tokens))
                wrap_embeds.append(img.cpu().float())
                input_tokens.extend([image_pad_id] * img.shape[1])

        # fixed input_tokens
        # 如果 input_tokens > max_length 需要截断
        input_tokens = input_tokens[:self.max_length]

        ksana_python_input.input_tokens = input_tokens
        ksana_python_input.input_refit_embedding.pos = url_srt
        ksana_python_input.input_refit_embedding.embedding_tensors = wrap_embeds

    # Method for post-processing
    def postprocess(self, **kwargs):
        return

    def check_intput(self, input_list: List[str], **kwargs) -> bool:
        for input_name in input_list:
            if input_name not in kwargs:
                print(f"[E] Input {input_name} not found.")
                return False
        return True

    def _init_trt(self, model_path):
        from internlmxcomposer2.ksana_plugin_model import VITModel
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

        from trt_engine import Engine
        model = VITModel(model_path)

        trt_path = model.get_trt_path(model_path)
        trt_engine = Engine(trt_path)


        # If there is no TRT engine, Start model convert
        if not os.path.exists(trt_path):
            build_trt_process(current_dir, model_path)

        # Load trt
        trt_engine.load()
        self.stream = torch.cuda.current_stream().cuda_stream

        self.visual = trt_engine
        self.model = model

    def _infer_trt(self, input_imgs):
        # TRT engine can split the input according to the engine's maximum batch size
        split_size = self.model.max_batch
        images_list = [input_imgs]
        if input_imgs.size(0) > split_size:
            images_list = torch.split(input_imgs, split_size)

        outs_list = []
        for image in images_list:
            batch_size = image.size(0)
            infer_shape = self.model.get_infer_shape(batch_size)
            self.visual.allocate_buffers(infer_shape)

            infer_data = self.model.get_infer_data(image)
            target = self.model.get_output_names()[0]
            out = self.visual.infer(infer_data, self.stream)[target]

            outs_list.append(out)
        image_features = torch.cat(outs_list, dim=0)
        return image_features.to(input_imgs.dtype)

    @staticmethod
    def convert_openai_messages(messages):
        # Convert `messages` in OpenAI Chat Completion format to IXC format
        text = None
        image_list = []
        for message in messages:
            for part in message["content"]:
                if "text" in part:
                    text = part["text"] 
                if "zip_url" in part:
                    image_url = part["zip_url"]["url"]
                    image_list.append(image_url)
                if "image_url" in part:
                    image_url = part["image_url"]["url"]
                    image_list.append(image_url)
        return text, image_list
