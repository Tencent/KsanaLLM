# Adapted from https://github.com/TencentARC/ARC-Hunyuan-Video-7B/blob/master/model_vllm/video_audio_encoder.py

import math

import torch
import torch.nn as nn
from transformers.modeling_utils import no_init_weights
from transformers import ARCHunyuanVideoVisionModel, ARCHunyuanVideoAudioEncoder


class VideoAudioEncoder(nn.Module):
    def __init__(self, config, max_num_frames=150):
        super().__init__()
        self.max_num_frames = max_num_frames

        config.vision_config._attn_implementation = "sdpa"
        config.audio_config._attn_implementation = "sdpa"

        with no_init_weights():
            # Initialize vision model
            self.vision_model = ARCHunyuanVideoVisionModel(
                vision_config=config.vision_config,
                text_config=config.text_config,
            )

            self.speech_encoder = ARCHunyuanVideoAudioEncoder(
                config=config.audio_config,
            )

        self.speech_dim = config.audio_config.d_model

        llm_hidden_size = config.text_config.hidden_size

        self.mlp2 = nn.Sequential(
            nn.LayerNorm(self.speech_dim),
            nn.Linear(self.speech_dim, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    @torch.no_grad()
    def extract_image_feature(self, pixel_values):
        """Extract features from image tensors using vision model"""
        vit_embeds = self.vision_model(pixel_values)
        return vit_embeds

    @torch.no_grad()
    def extract_audio_feature(self, audio_values):
        """Extract features from audio tensors using speech encoder"""
        audio_values = audio_values.squeeze(0).reshape(
            -1, 128, audio_values.shape[-1]
        )
        num_segments = audio_values.shape[0]

        speech_embeds = self.speech_encoder(
            audio_values, return_dict=True
        ).last_hidden_state

        speech_embeds = speech_embeds.reshape(1, -1, speech_embeds.shape[-1])
        speech_embeds = self.mlp2(speech_embeds)
        return num_segments, speech_embeds

    def create_mixed_embeddings(self, vit_embeds, audio_embeds, duration):
        """Create mixed embeddings from visual and audio features"""
        # Reshape audio embeddings to match video frames
        audio_embeds = audio_embeds.reshape(
            audio_embeds.shape[0], -1, 50, audio_embeds.shape[-1]
        )
        audio_embeds_no_pad = audio_embeds[:, :duration].squeeze(0)

        max_num_frame = self.max_num_frames

        # Handle case where audio duration exceeds max number of frames
        if duration > max_num_frame:
            per_audio_tokens = math.ceil(
                audio_embeds_no_pad.shape[0] / max_num_frame * 50
            )
            num_audio_tokens_sum = per_audio_tokens * max_num_frame
            audio_embeds_no_pad = audio_embeds_no_pad.reshape(
                -1, audio_embeds_no_pad.shape[-1]
            )

            if num_audio_tokens_sum != audio_embeds_no_pad.shape[0]:
                zero_padding = (
                    torch.zeros(
                        num_audio_tokens_sum - audio_embeds_no_pad.shape[0],
                        audio_embeds_no_pad.shape[-1],
                    )
                    .to(audio_embeds_no_pad.dtype)
                    .to(audio_embeds_no_pad.device)
                )
                audio_embeds_no_pad = torch.cat(
                    (audio_embeds_no_pad, zero_padding), dim=0
                )

            audio_embeds_no_pad = audio_embeds_no_pad.reshape(
                max_num_frame, -1, audio_embeds_no_pad.shape[-1]
            )

        # Pad or trim to match the visual embedding shape
        padding_size = vit_embeds.shape[1] - audio_embeds_no_pad.shape[1]
        if padding_size != 0:
            zero_padding = (
                torch.zeros(
                    vit_embeds.shape[0],
                    padding_size,
                    audio_embeds_no_pad.shape[-1],
                )
                .to(audio_embeds_no_pad.dtype)
                .to(audio_embeds_no_pad.device)
            )
            audio_embeds_pad = torch.cat(
                (audio_embeds_no_pad, zero_padding), dim=1
            )
        else:
            audio_embeds_pad = audio_embeds_no_pad

        mixed_embeds = vit_embeds + audio_embeds_pad

        return mixed_embeds

    def forward(self, pixel_values, audio_values, duration):
        """
        Encode images and audio to create mixed embeddings

        Args:
            pixel_values (torch.Tensor): Batch of images from video (processed frames)
            audio_values (torch.Tensor): Processed audio features
            duration (int): Duration of the video in frames or seconds

        Returns:
            mixed_embeds (torch.Tensor): Mixed embeddings combining vision and audio
        """

        # Extract features
        vit_embeds = self.extract_image_feature(pixel_values)

        _, audio_embeds = self.extract_audio_feature(audio_values)

        # Create mixed embeddings
        mixed_embeds = self.create_mixed_embeddings(
            vit_embeds, audio_embeds, duration
        )

        return mixed_embeds
