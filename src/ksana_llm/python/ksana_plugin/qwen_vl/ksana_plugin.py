# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import sys

import torch

# parent_dir: ./KsanaLLM/src/ksana_llm/python/ksana_plugin
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from qwen_vl.ksana_plugin_model import VITModel
from plugin_utils import free_cache, adjust_device_memory_ratio, build_trt_process


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
            self.model = VITModel(model_path)
            self.visual = None

            self.trt = False
            if enable_trt:
                try:
                    self.visual = self._init_trt(model_path)
                    self.trt = True
                    print(f"[I] Initializing the TensorRT model successfully!")
                except Exception as e:  # pylint: disable=broad-except
                    print(f"[E] Failed to initialize TensorRT model : {e}")

            if not self.trt:
                self.visual = self._init_torch(model_path)
                print(f"[I] Initializing the Torch model successfully!")

            free_cache()

            adjust_device_memory_ratio(kwargs["config_file"], 0.01 if self.trt else 0.04)

            # Ensure the result is a dictionary
            return {
                       'plugin_trt' : self.trt,
                   }

        if "postprocess" in kwargs:
            pass

    # Method for pre-processing
    def preprocess(self, **kwargs):
        config = self.model.config
        if self.check_intput(**kwargs):
            raise RuntimeError(f"Check input failed.")
        ksana_python_input = kwargs['ksana_python_input']

        input_tokens = ksana_python_input.input_tokens
        url_srt = [int(pos+1) for pos, ids in enumerate(input_tokens) if ids == config.visual["image_start_id"]]
        url_end = [int(pos-1) for pos, ids in enumerate(input_tokens) if ids == config.visual["image_start_id"]+1]
        image_url = []
        for i in range(len(url_srt)):
            url = input_tokens[url_srt[i]:url_end[i]]
            url = url[:url.index(config.visual['image_start_id']+2)]
            image_url.append(bytes(url).decode('utf-8'))

        if (len(image_url) == 0):
            return

        if not self.trt:
            image_embedding  = self._infer_torch(image_url)
        else:
            image_embedding  = self._infer_trt(image_url)

        ksana_python_input.input_refit_embedding.pos = url_srt
        ksana_python_input.input_refit_embedding.embedding_tensors = torch.unbind(image_embedding.cpu().float())

    # Method for post-processing
    def postprocess(self, **kwargs):
        return

    def check_intput(self, **kwargs):
        input_list = [
            'ksana_python_input',
        ]
        for input_name in input_list:
            if input_name not in kwargs:
                print(f"input {input_name} not found.")
                return False

    def _init_torch(self, model_path):
        model = self.model.get_model(model_path)
        return model

    def _init_trt(self, model_path):
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

        from trt_engine import Engine

        trt_path = self.model.get_trt_path(model_path)
        trt_engine = Engine(trt_path)

        # If there is no TRT engine, Start model convert
        if not os.path.exists(trt_path):
            build_trt_process(current_dir, model_path)

        # Load trt
        trt_engine.load()
        self.model.get_preprocess()

        return trt_engine

    def _infer_torch(self, image_url):
        with torch.no_grad():
            image_embedding = self.visual.encode(image_url)
        return image_embedding

    def _infer_trt(self, image_url):
        images = self.model.image_pre_obj.encode(image_url).to(self.model.device).contiguous()

        # TRT engine can split the input according to the engine's maximum batch size
        split_size = self.model.max_batch
        images_list = [images]
        if images.size(0) > split_size:
            images_list = torch.split(images, split_size)

        outs_list = []
        for image in images_list:
            batch_size = image.size(0)
            infer_shape = self.model.get_infer_shape(batch_size)
            self.visual.allocate_buffers(infer_shape, device=self.model.device)

            infer_data = self.model.get_infer_data(image)
            target = self.model.get_output_names()[0]
            out = self.visual.infer(infer_data, torch.cuda.current_stream().cuda_stream)[target]

            outs_list.append(out)
        image_embedding = torch.cat(outs_list, dim=0)
        return image_embedding
