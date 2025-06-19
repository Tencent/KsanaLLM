# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import sys
from typing import Dict, List, Optional, Union
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

import torch

# parent_dir: ./KsanaLLM/src/ksana_llm/python/ksana_plugin
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from qwen2_vl.ksana_plugin_model import VITModel
from plugin_utils import free_cache, adjust_device_memory_ratio


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
            # TODO(yfjin): Support inference ViT by TensorRT
            enable_trt = False  # kwargs.get('enable_trt', True)

            # init processor
            self.processor = AutoProcessor.from_pretrained(model_path)

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
            return

    # Method for pre-processing
    def preprocess(self, **kwargs):
        if not self.check_intput(['ksana_python_input', 'messages'], **kwargs):
            raise RuntimeError(f"Plugin preprocess wrong input.")

        messages: Optional[List[Dict]] = kwargs['messages']
        if messages is None:
            return
        messages = KsanaPlugin.convert_openai_messages(messages, kwargs['additional_params'])

        ksana_python_input = kwargs['ksana_python_input']
        config = self.model.config

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        input_tokens = inputs["input_ids"][0].tolist()
        ksana_python_input.input_tokens = input_tokens
        inputs = inputs.to(self.model.device)

        vision_srt = [int(pos + 1) for pos, id in enumerate(input_tokens) if id == config.vision_start_token_id]
        vision_end = [int(pos - 1) for pos, id in enumerate(input_tokens) if id == config.vision_end_token_id]

        # Used to reassemble image embeddings and video embeddings in the order of visual tokens
        permute = {}

        def infer_embeds(permute, tag="image"):
            pixel_values_tag = "pixel_values_videos" if tag == "video" else "pixel_values"
            grid_thw_tag = "video_grid_thw" if tag == "video" else "image_grid_thw"
            token_id = config.video_token_id if tag == "video" else config.image_token_id

            grid_thw = []
            if pixel_values_tag in inputs:
                pixel_values = inputs[pixel_values_tag].type(self.visual.get_dtype())
                grid_thw = inputs[grid_thw_tag]
                embeds = self._infer_torch(pixel_values, grid_thw)
                sections, idx = [], []
                for i, srt in enumerate(vision_srt):
                    if input_tokens[srt] == token_id:
                        sections.append(vision_end[i] - srt + 1)
                        idx.append(i)
                embeds = torch.split(embeds, sections, dim=0)
                for i in range(len(embeds)):
                    permute[idx[i]] = embeds[i]
            return grid_thw

        image_grid_thw = infer_embeds(permute, tag="image")
        video_grid_thw = infer_embeds(permute, tag="video")

        embeds = []
        for i in range(len(permute)):
            embeds.append(permute[i].cpu().float())
        ksana_python_input.input_refit_embedding.pos = vision_srt
        ksana_python_input.input_refit_embedding.embedding_tensors = embeds

        position_ids, mrope_position_deltas = self._get_input_positions(input_tokens, image_grid_thw, video_grid_thw)
        ksana_python_input.input_refit_embedding.additional_tensors = [
            position_ids.squeeze(0).contiguous(), mrope_position_deltas]

    # Method for post-processing
    def postprocess(self, **kwargs):
        return

    def check_intput(self, input_list: List[str], **kwargs) -> bool:
        for input_name in input_list:
            if input_name not in kwargs:
                print(f"[E] Input {input_name} not found.")
                return False
        return True

    def _init_torch(self, model_path):
        model = self.model.get_model(model_path)
        return model

    def _infer_torch(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
        with torch.no_grad():
            visual_embedding = self.visual(hidden_states, grid_thw)
        return visual_embedding

    def _get_input_positions(
        self,
        input_tokens: List[int],
        image_grid_thw: Union[List[List[int]], torch.Tensor],
        video_grid_thw: Union[List[List[int]], torch.Tensor],
    ):
        """Get mrope input positions and delta value.

        Adapted from
        https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding.py#L840
        """
        config = self.model.config
        spatial_merge_size = config.vision_config.spatial_merge_size
        image_token_id = config.image_token_id
        video_token_id = config.video_token_id
        vision_start_token_id = config.vision_start_token_id
        spatial_merge_size = config.vision_config.spatial_merge_size

        if isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = image_grid_thw.tolist()
        if isinstance(video_grid_thw, torch.Tensor):
            video_grid_thw = video_grid_thw.tolist()

        input_tokens_tensor = torch.tensor(input_tokens)
        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_videos = image_nums, video_nums

        image_index, video_index = 0, 0
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_index += 1
                remain_videos -= 1
                ed = ed_video
            llm_grid_t, llm_grid_h, llm_grid_w = \
                t, h // spatial_merge_size, w // spatial_merge_size
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).repeat_interleave(3) + st_idx)

            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(
                -1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
                llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
                llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(
                (torch.tensor(list(zip(t_index, h_index, w_index)))
                             + text_len + st_idx).view(-1))
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).repeat_interleave(3) + st_idx)

        # We set the layout of mrotary_embedding_pos be [t1, h1, w1, t2, h2, w2, ...],
        # instead of the conventional [t1, t2, ..., h1, h2, ..., w1, w2, ...].
        # This is to facilitate the implementation of our MRotaryEmbedding kernel.
        llm_positions = torch.cat(llm_pos_ids_list)
        mrope_position_delta = llm_positions.max() + 1 - len(input_tokens)

        return llm_positions, mrope_position_delta

    @staticmethod
    def convert_openai_messages(messages: List[Dict], additional_params: Dict) -> List[Dict]:
        # Convert `messages` in OpenAI Chat Completion format to Qwen2VL format,
        # so that it can be processed by qwen_vl_utils.
        # Refer to
        # https://github.com/QwenLM/Qwen2-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
        for i, message in enumerate(messages):
            for j, part in enumerate(message["content"]):
                if "image_url" in part:
                    messages[i]["content"][j]["image"] = part["image_url"]["url"]
                    for param in ["min_pixels", "max_pixels",
                                  "resized_height", "resized_width"]:
                        if param in additional_params:
                            messages[i]["content"][j][param] = additional_params[param]
                    messages[i]["content"][j].pop("image_url")
                elif "video_url" in part:
                    messages[i]["content"][j]["video"] = part["video_url"]["url"]
                    for param in ["fps", "nframes", "min_frames", "max_frames",
                                  "resized_height", "resized_width"]:
                        if param in additional_params:
                            messages[i]["content"][j][param] = additional_params[param]
                    messages[i]["content"][j].pop("video_url")
        
        for message in messages:
            if message['role'] == 'user' and 'content' in message:
                multi_modal_elements = [elem for elem in message['content'] if elem['type'] != 'text']
                text_elements = [elem for elem in message['content'] if elem['type'] == 'text']
                # 重组列表：非text元素在前，text元素在后, 保证顺序与LLMPT服务一致
                message['content'] = multi_modal_elements + text_elements
        return messages
