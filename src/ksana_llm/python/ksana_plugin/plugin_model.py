import torch
from torch import device
from transformers import AutoConfig

from plugin_utils import check_file_dir

CUDA_0 = device("cuda:0")


class BaseVITModel:

    def __init__(self, model_path):
        # read config
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.precision = self.config.torch_dtype

        self.image_size = self.config.get("image_size")
        self.output_dim = self.config.get("output_dim")
        self.dim = 3
        self.token_num = 256

        # trt build config
        self.min_batch = 1
        self.opt_batch = 2
        self.max_batch = 4

    def get_onnx_path(self, worker_path):
        onnx_path = f'{worker_path}/onnx/visual_encoder.onnx'
        check_file_dir(onnx_path)
        return onnx_path

    def get_trt_path(self, worker_path):
        trt_path = f'{worker_path}/trt/visual_encoder_fp16.plan'
        check_file_dir(trt_path)
        return trt_path

    def get_input_names(self):
        return ['input']

    def get_output_names(self):
        return ['output']

    def get_dynamic_axes(self):
        # Build onnx config
        return {
                'input': {0: 'B'}
            }

    def get_trt_profile(self):
        # Build trt config
        return {
                'input': [(self.min_batch, self.dim, self.image_size, self.image_size),
                          (self.opt_batch, self.dim, self.image_size, self.image_size),
                          (self.max_batch, self.dim, self.image_size, self.image_size)]
            }

    def get_sample_input(self, device=CUDA_0):
        return (
            torch.randn(self.opt_batch, self.dim, self.image_size, self.image_size).to(device)
        )

    def get_infer_shape(self, infer_batch):
        return {
                'input': (infer_batch, self.dim, self.image_size, self.image_size),
                'output': (infer_batch, self.token_num, self.output_dim),
            }

    def get_infer_data(self, image):
        return {
                'input': image.float()
            }
