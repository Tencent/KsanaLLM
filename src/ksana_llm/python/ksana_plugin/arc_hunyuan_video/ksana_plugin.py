# pylint: disable=broad-except,invalid-name,unused-variable
# Copyright 2025 Tencent Inc.  All rights reserved.
#
# Adapted from https://github.com/TencentARC/ARC-Hunyuan-Video-7B/tree/master/model_vllm
# ==============================================================================

import math
import os
import sys
from typing import Dict, List, Optional, Union, Tuple

import json
from pathlib import Path
import numpy as np
import librosa
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageDraw, ImageFont
from decord import VideoReader, cpu
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from transformers import (
    WhisperFeatureExtractor,
    AutoConfig,
    ARCHunyuanVideoProcessor,
    PretrainedConfig,
)
from safetensors.torch import load_file as safetensors_load_file

import torch

# parent_dir: ./KsanaLLM/src/ksana_llm/python/ksana_plugin
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from arc_hunyuan_video.video_audio_encoder import VideoAudioEncoder
from plugin_utils import free_cache, adjust_device_memory_ratio


class VideoProcessorConfig:
    """配置视频处理相关参数"""

    def __init__(self):
        self.image_size = 640
        self.max_num_frame = 150
        self.factor = 2
        self.dtype = torch.bfloat16
        self.hunyuan_mean = (0.48145466, 0.4578275, 0.40821073)
        self.hunyuan_std = (0.26862954, 0.26130258, 0.27577711)


def load_state_dict_from_safetensors(path: str, prefixes: list[str]):
    def filter_dict_with_k_prefix(d, prefixes):
        return {
            k: v
            for k, v in d.items()
            if any(k.startswith(prefix) for prefix in prefixes)
        }

    index_path = os.path.join(path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"Index file {index_path} does not exist, loading all weights")
        pre_trained_dir = Path(path)
        weights_files = sorted(pre_trained_dir.glob("model-*.safetensors"))
    else:
        weight_map = json.load(open(index_path))["weight_map"]
        weights_files = set(filter_dict_with_k_prefix(weight_map, prefixes).values())
        weights_files = [os.path.join(path, f) for f in weights_files]

    if len(weights_files) == 0:
        raise ValueError(f"No weights files found in {path} with prefixes {prefixes}")

    state_dict = {}
    for file in weights_files:
        part_state_dict = safetensors_load_file(file)
        state_dict.update(part_state_dict)

    state_dict = filter_dict_with_k_prefix(state_dict, prefixes)
    return state_dict


class KsanaPlugin:
    """
    ARC Hunyuan Video Plugin for KsanaLLM
    用于处理混元视频模型的预处理和后处理
    """

    def __init__(self):
        self.device = "cuda:0"
        self.vision_config = None
        self.processor = None

    # Plugin initialization is automatically invoked upon service startup.
    def init_plugin(self, **kwargs):
        if "preprocess" in kwargs:
            model_path = kwargs["model_path"]
            # TODO: Support inference by TensorRT
            enable_trt = False  # kwargs.get('enable_trt', True)

            self.trt = False
            if enable_trt:
                try:
                    self.visual = self._init_trt(model_path)
                    self.trt = True
                    print("[I] Initializing the TensorRT model successfully!")
                except Exception as e:
                    print(f"[E] Failed to initialize TensorRT model : {e}")

            if not self.trt:
                self.config = AutoConfig.from_pretrained(model_path)
                self.vision_config = VideoProcessorConfig()
                self.wav_processor = WhisperFeatureExtractor.from_pretrained(model_path)
                self.mm_encoder = self._init_mm_encoder(
                    model_path, self.config, self.device
                )
                self.processor = ARCHunyuanVideoProcessor.from_pretrained(model_path)
                print("[I] Initializing the Torch model successfully!")

            free_cache()

            # Adjust device memory ratio for video model (may need more memory)
            adjust_device_memory_ratio(
                kwargs["config_file"], 0.01 if self.trt else 0.06
            )

            # Ensure the result is a dictionary
            return {
                "plugin_trt": self.trt,
            }

        if "postprocess" in kwargs:
            return

    # Method for pre-processing
    def preprocess(self, **kwargs):
        """
        预处理视频输入

        Args:
            ksana_python_input: KsanaLLM输入对象
            messages: 消息列表（OpenAI格式）
            additional_params: 额外参数
        """
        if not self.check_input(["ksana_python_input", "messages"], **kwargs):
            raise RuntimeError(f"Plugin preprocess wrong input.")

        messages: Optional[List[Dict]] = kwargs["messages"]
        if messages is None:
            return

        # Convert OpenAI format messages to model-specific format
        question, video_path, audio_path, task = KsanaPlugin.convert_openai_messages(
            messages, kwargs["additional_params"]
        )

        # 解析视频和音频
        pixel_values, num_patches = self._load_video_frames(video_path)
        audio_features, duration = self._process_audio(video_path, audio_path)

        # 超长裁剪
        if duration < pixel_values.shape[0]:
            pixel_values = pixel_values[:duration]
        if duration <= self.vision_config.max_num_frame:
            duration = pixel_values.shape[0]
        else:
            assert pixel_values.shape[0] == self.vision_config.max_num_frame

        # 生成prompt
        prompt = "<|startoftext|>" + self._build_prompt(
            pixel_values.shape[0], question, task
        )

        # 多模态encoder
        embeddings = self._forward_mm_encoder(pixel_values, audio_features, duration)

        # grid构建
        num_patches = self.config.vision_config.force_image_size // 32 // 2
        image_grid_thw = torch.tensor([[1, num_patches, num_patches + 1]])
        image_grid_thw = image_grid_thw.repeat(embeddings.shape[0], 1)

        # prompt替换 & token生成
        image_replace = self._get_image_replace(embeddings.shape[1])
        prompt = prompt.replace("<image>", image_replace)
        inputs = self.processor(text=prompt, return_tensors="pt")

        # 类型转换
        input_tokens = inputs["input_ids"].tolist()
        video_embeds = [embedding.cpu().float() for embedding in embeddings]

        # 查找多模态token位置
        video_start = [
            int(pos + 1)
            for pos, id in enumerate(input_tokens)
            if id == self.config.text_config.im_start_id
        ]

        # 计算xdrope的position信息
        llm_positions, mrope_position_delta = self._get_input_positions_tensor(
            input_tokens=input_tokens,
            hf_config=self.config.text_config,
            image_grid_thw=image_grid_thw,
            video_grid_thw=[],
            second_per_grid_ts=[],
            context_len=0,
            seq_len=None,
        )

        # 绑定数据
        ksana_python_input = kwargs["ksana_python_input"]
        ksana_python_input.input_tokens = input_tokens
        ksana_python_input.input_refit_embedding.pos = video_start
        ksana_python_input.input_refit_embedding.embedding_tensors = video_embeds
        ksana_python_input.input_refit_embedding.additional_tensors = [
            llm_positions,
            mrope_position_delta,
        ]

    # Method for post-processing
    def postprocess(self, **kwargs):
        """
        后处理生成的输出
        目前暂未实现特殊的后处理逻辑
        """
        return

    def check_input(self, input_list: List[str], **kwargs) -> bool:
        """检查必需的输入参数是否存在"""
        for input_name in input_list:
            if input_name not in kwargs:
                print(f"[E] Input {input_name} not found.")
                return False
        return True

    # https://github.com/TencentARC/ARC-Hunyuan-Video-7B/model_vllm/hunyuan.py
    def _get_input_positions_tensor(
        self,
        input_tokens: List[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[List[List[int]], torch.Tensor],
        video_grid_thw: Union[List[List[int]], torch.Tensor],
        second_per_grid_ts: List[float],
        context_len: int = 0,
        seq_len: Optional[int] = None,
        audio_feature_lengths: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        return self._vl_get_input_positions_tensor(
            input_tokens=input_tokens,
            hf_config=hf_config,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            context_len=context_len,
            seq_len=seq_len,
        )

    def _vl_get_input_positions_tensor(
        self,
        input_tokens: List[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[List[List[int]], torch.Tensor],
        video_grid_thw: Union[List[List[int]], torch.Tensor],
        second_per_grid_ts: List[float],
        context_len: int = 0,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Get xdrope input positions following get_xdrope_position_ids pattern."""

        image_token_id = hf_config.image_token_id
        vision_start_token_id = hf_config.im_start_id

        input_tokens_tensor = torch.tensor(input_tokens)

        # Initialize 4D position embeddings (following xdrope pattern)
        seq_length = len(input_tokens)
        position_ids_seq = torch.arange(seq_length)  # Sequential positions
        position_ids_t = position_ids_seq.clone()
        position_ids_x = position_ids_seq.clone()
        position_ids_y = position_ids_seq.clone()

        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id
        ).squeeze(1)

        if len(vision_start_indices) == 0:
            # No vision tokens, return 4D sequential positions
            llm_positions = torch.stack(
                [position_ids_seq, position_ids_x, position_ids_y, position_ids_t]
            )
            mrope_position_delta = 0
            llm_positions = llm_positions[:, context_len:seq_len]
            return llm_positions, mrope_position_delta

        # Process vision tokens using image_grid_thw information
        image_index, video_index = 0, 0
        current_pos = 0

        for start_idx in vision_start_indices:
            start_idx = start_idx.item()

            # Determine if this is image or video token
            if start_idx + 1 < len(input_tokens):
                next_token = input_tokens[start_idx + 1]
                is_image = next_token == image_token_id

                if is_image and image_index < len(image_grid_thw):
                    t, h, w = image_grid_thw[image_index]
                    image_index += 1
                else:
                    continue

                # Calculate grid dimensions
                llm_grid_t, llm_grid_h, llm_grid_w = (t, h, w)

                # Find end of vision tokens (approximate)
                vision_token_count = llm_grid_t * llm_grid_h * llm_grid_w
                end_idx = min(
                    start_idx + vision_token_count + 2, seq_length
                )  # +2 for start/end tokens

                # Apply xdrope position assignment pattern
                if end_idx > start_idx + 2:  # Ensure we have vision tokens
                    # Reset time dimension for vision tokens (following get_xdrope_position_ids)
                    position_ids_t[start_idx + 2 : end_idx] = current_pos
                    current_pos += 1

                    # Calculate row and column for 2D layout
                    vision_tokens_between = (end_idx - start_idx - 2)
                    if llm_grid_h > 0:
                        tokens_per_row = llm_grid_w
                        num_rows = llm_grid_h

                        # Assign x,y coordinates following the pattern
                        idx_xy = 0
                        for rr in range(num_rows):
                            for cc in range(tokens_per_row):
                                if start_idx + 2 + idx_xy < end_idx:
                                    position_ids_x[start_idx + 2 + idx_xy] = cc
                                    position_ids_y[start_idx + 2 + idx_xy] = rr
                                    idx_xy += 1

        # Stack into 4D positions
        llm_positions = torch.stack(
            [position_ids_seq, position_ids_x, position_ids_y, position_ids_t]
        )
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
        llm_positions = llm_positions[:, context_len:seq_len]

        # llm_positions 原始shape是(4,num_token)为了方便算子计算转置成(num_token,4)
        # mrope_position_delta 在解析的时候按照torch.Tensor解析的，额外包装一下
        return llm_positions.t().contiguous(), torch.Tensor([mrope_position_delta])

    def _get_image_replace(self, num_image_token):
        IMG_START = "<img>"
        IMG_END = "</img>"
        IMG_CONTEXT = "<IMG_CONTEXT>"
        replace_features = IMG_CONTEXT * num_image_token
        replace_full = IMG_START + replace_features + IMG_END
        return replace_full

    def _init_mm_encoder(self, model_path, config, device):
        multi_modal_state_dict = load_state_dict_from_safetensors(
            model_path, ("vision_model.", "mlp2.", "speech_encoder.")
        )

        multi_modal_encoder = VideoAudioEncoder(
            config,
            max_num_frames=config.max_num_frame,
        )

        missing, unexpected = multi_modal_encoder.load_state_dict(
            multi_modal_state_dict, strict=False
        )
        assert len(missing) == 0, f"Missing keys in mm encoder: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys in mm encoder: {unexpected}"

        multi_modal_encoder.eval()
        multi_modal_encoder.to(device)

        return multi_modal_encoder

    def _forward_mm_encoder(self, pixel_values, audio_values, duration):
        with torch.no_grad(), torch.autocast(self.device, torch.bfloat16):
            pixel_values = pixel_values.to(
                device=self.device, dtype=torch.bfloat16, non_blocking=True
            )
            audio_values = audio_values.to(
                device=self.device, dtype=torch.bfloat16, non_blocking=True
            )

            mixed_embeds = self.mm_encoder(pixel_values, audio_values, duration)

            mixed_embeds = mixed_embeds.to(device="cpu").float().share_memory_()

        return mixed_embeds

    def _build_prompt(self, num_frames: int, question: str, task: str) -> str:
        video_prefix = "<image>" * num_frames

        if task == "MCQ":
            return (
                f"{video_prefix}\n{question}\n"
                f"Output the thinking process in <think> </think> and "
                f"final answer (only option index) in <answer> </answer> tags, "
                f"i.e., <think> reasoning process here </think>"
                f"<answer> answer here </answer>.<sep>"
            )
        elif task == "Grounding":
            return (
                f"{video_prefix}\n{question}\n"
                f"Output the thinking process in <think> </think> and "
                f"final answer (only time range) in <answer> </answer> tags, "
                f"i.e., <think> reasoning process here </think>"
                f"<answer> answer here </answer>.<sep>"
            )
        else:  # QA、summary、segment
            return (
                f"{video_prefix}\n{question}\n"
                f"Output the thinking process in <think> </think> and "
                f"final answer in <answer> </answer> tags, "
                f"i.e., <think> reasoning process here </think>"
                f"<answer> answer here </answer>.<sep>"
            )

    def _load_video_frames(self, video_path: str) -> tuple[torch.Tensor, list]:
        video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=4)
        vlen = len(video_reader)
        input_fps = video_reader.get_avg_fps()
        duration = vlen / input_fps

        transform = self._build_video_transform()
        frame_indices, intervals_sec = self._calculate_frame_indices(
            vlen, input_fps, duration
        )

        pixel_values = []
        for i, idx in enumerate(frame_indices):
            frame = Image.fromarray(video_reader[idx].asnumpy())
            start_sec, end_sec = intervals_sec[i]
            frame = self._add_timestamp_to_frame(frame, start_sec, end_sec)
            pixel_values.append(transform(frame))

        return torch.stack(pixel_values).to(self.vision_config.dtype), [1] * len(
            pixel_values
        )

    def _process_audio(self, video_path: str, audio_path: str) -> torch.Tensor:
        video = VideoFileClip(video_path)
        try:
            video.audio.write_audiofile(audio_path, logger=None)
            audio, sr = self._cut_audio_with_librosa(
                audio_path,
                max_num_frame=150,
                segment_sec=2,
                max_total_sec=300,
                sr=16000,
            )
        except Exception as ex:
            # when no audios
            duration = min(math.ceil(video.duration), 300)
            silent_audio = AudioSegment.silent(duration=duration * 1000)
            silent_audio.export(audio_path, format="mp3")
            print("no audio", audio_path)
            audio, sr = librosa.load(audio_path, sr=16000)

        audio = self._pad_audio(audio, sr)
        duration = math.ceil(len(audio) / sr)

        return self._extract_spectrogram(audio, sr), duration

    def _extract_spectrogram(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        segment_length = sr * 30
        spectrograms = []

        for i in range(0, len(audio), segment_length):
            segment = audio[i : i + segment_length]
            spectrograms.append(
                self.wav_processor(segment, sampling_rate=sr, return_tensors="pt")[
                    "input_features"
                ]
            )
        return torch.cat(spectrograms).to(self.vision_config.dtype)

    def _pad_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if len(audio.shape) == 2:
            audio = audio[:, 0]
        if len(audio) < sr:
            sil = np.zeros(sr - len(audio), dtype=float)
            audio = np.concatenate((audio, sil), axis=0)
        return audio

    def _cut_audio_with_librosa(
        self, audio_path, max_num_frame=150, segment_sec=2, max_total_sec=300, sr=16000
    ):
        audio, _ = librosa.load(audio_path, sr=sr)
        total_samples = len(audio)
        total_sec = total_samples / sr

        if total_sec <= max_total_sec:
            return audio, sr

        segment_length = total_samples // max_num_frame
        segment_samples = int(segment_sec * sr)
        segments = []
        for i in range(max_num_frame):
            start = i * segment_length
            end = min(start + segment_samples, total_samples)
            segments.append(audio[start:end])
        new_audio = np.concatenate(segments)
        return new_audio, sr

    def _build_video_transform(self) -> T.Compose:
        return T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (self.vision_config.image_size, self.vision_config.image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(
                    mean=self.vision_config.hunyuan_mean,
                    std=self.vision_config.hunyuan_std,
                ),
            ]
        )

    def _sec2hms(self, seconds):
        seconds = int(round(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _add_timestamp_to_frame(self, frame, start_sec, end_sec, font_size=40):
        draw = ImageDraw.Draw(frame)
        font_size = int(frame.height * 0.05)
        font = ImageFont.truetype(os.path.join(current_dir, "ARIAL.TTF"), font_size)
        text = f"{self._sec2hms(start_sec)}-{self._sec2hms(end_sec)}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = frame.width - text_w - 20
        y = 20
        draw.rectangle(
            [x - 10, y - 10, x + text_w + 10, y + text_h + 10], fill=(0, 0, 0, 180)
        )
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        return frame

    def _calculate_frame_indices(self, vlen: int, fps: float, duration: float) -> list:
        frames_per_second = fps

        if duration <= self.vision_config.max_num_frame:
            interval = 1
            intervals = [
                (
                    int(i * interval * frames_per_second),
                    int((i + 1) * interval * frames_per_second),
                )
                for i in range(math.ceil(duration))
            ]
            intervals_sec = [
                (int(i * interval), int((i + 1) * interval))
                for i in range(math.ceil(duration))
            ]
        else:
            num_segments = self.vision_config.max_num_frame
            segment_duration = duration / num_segments
            intervals = [
                (
                    int(i * segment_duration * frames_per_second),
                    int((i + 1) * segment_duration * frames_per_second),
                )
                for i in range(num_segments)
            ]
            intervals_sec = [
                (round(i * segment_duration), round((i + 1) * segment_duration))
                for i in range(num_segments)
            ]

        frame_indices = []
        for start, end in intervals:
            if end > vlen:
                end = vlen
            frame_indices.append((start + end) // 2)

        return frame_indices, intervals_sec

    @staticmethod
    def convert_openai_messages(
        messages: List[Dict], additional_params: Dict
    ) -> List[Dict]:
        if not messages or len(messages) == 0:
            raise ValueError("messages不能为空")

        # 获取第一条消息的content
        content = messages[0].get("content", [])

        text = None
        video = None
        audio = None

        # 遍历content提取各个字段
        for item in content:
            item_type = item.get("type")

            if item_type == "text":
                text = item.get("text")
            elif item_type == "video":
                video = item.get("video_url", {}).get("url")
            elif item_type == "audio":
                audio = item.get("audio_url", {}).get("url")

        # 检查必需字段
        if text is None:
            raise ValueError("缺少必需的text字段")
        if video is None:
            raise ValueError("缺少必需的video字段")

        # 获取task，默认值为"summary"
        if additional_params is None:
            additional_params = {}
        task = additional_params.get("task", "summary")

        return text, video, audio, task
