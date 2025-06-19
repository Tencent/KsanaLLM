# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2025 Tencent Inc. All rights reserved.
# Copyright 2025 vLLM Team
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
#
# Adapted from
# [vLLM Project] https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/attention/ops/triton_decode_attention.py
# [SGLang Project] https://github.com/sgl-project/sglang/blob/9f635ea50de920aa507f486daafba26a5b837574/python/sglang/srt/layers/attention/triton_ops/decode_attention.py
# Originally from
# [LightLLM Project] https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# [LightLLM Project] https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py
#
# ==============================================================================

import argparse
import os
import json
import torch
import numpy as np

os.environ["PATH"] = "/usr/local/nvidia/lib64:" + os.environ["PATH"]

from typing import Any, Dict, List, Optional, Tuple, Union

import triton
import triton.language as tl


@triton.jit
def decode_softmax_reducev_fwd_kernel(
    Mid_O,
    o,
    B_Seqlen,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split,
                                  cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os,
                         mask=mask_d,
                         other=0.0)
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )

def decode_softmax_reducev_fwd(
    logits,
    batch,
    head_num,
    o,
    Lv,
    b_seq_len,
    num_kv_splits,
):
    BLOCK_DV = triton.next_power_of_2(Lv)

    NUM_KV_SPLITS = num_kv_splits

    extra_kargs = {}
    grid = (batch, head_num)
    kernel = decode_softmax_reducev_fwd_kernel[grid](
        logits,
        o,
        b_seq_len,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )
    return o, kernel

def str_to_bool(value):
    if value.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif value.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Invalid boolean value: {}'.format(value))

def args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--BLOCK_DV', type=int, default=512)
    parser.add_argument('--NUM_KV_SPLITS', type=int, default=4)
    parser.add_argument('--Lv', type=int, default=512)
    parser.add_argument('--input_type', type=str, required=True,
                        choices=["FP16", "BF16"])

    parser.add_argument('--output_dir', type=str, default="./")
    args = parser.parse_args()
    if args.input_type == "FP16":
        args.torch_dtype = torch.float16
    elif args.input_type == "BF16":
        args.torch_dtype = torch.bfloat16
    return args


if __name__ == "__main__":
    torch.manual_seed(0)
    args = args_config()
    num_kv_splits = args.NUM_KV_SPLITS
    batch = 1
    logit_cap = 0.0
    input_type = args.torch_dtype
    num_heads = 16
    kv_lora_rank = 512

    b_seq_len = torch.zeros([batch], device='cuda', dtype=torch.int32)
    attn_logits = torch.zeros([batch, num_heads, num_kv_splits, kv_lora_rank + 1], device='cuda', dtype=torch.float32)
    o = torch.zeros([batch, num_heads, kv_lora_rank], device='cuda', dtype=input_type)
    output_triton, kernel = decode_softmax_reducev_fwd(attn_logits, batch, num_heads, o, args.Lv, b_seq_len,
                                num_kv_splits)

    kernel_name = "decode_softmax_reducev_fwd_kernel" + \
                  f"_BLOCK_DV_{args.BLOCK_DV}" + \
                  f"_NUM_KV_SPLITS_{args.NUM_KV_SPLITS}" + \
                  f"_Lv_{args.Lv}" + \
                  f"_input_type_{args.input_type}"


    with open(f"{args.output_dir}/{kernel_name}.cubin", "wb") as _f:
        _f.write(kernel.asm['cubin'])
    with open(f"{args.output_dir}/{kernel_name}.json", "w") as _f:
        json_dict = {"shm_size": kernel.metadata.shared,
                     "num_warps": kernel.metadata.num_warps,
                     "num_stages": kernel.metadata.num_stages}
        _f.write(json.dumps(json_dict))
    with open(f"{args.output_dir}/{kernel_name}.ptx", "w") as _f:
        SHM_SIZE = 0
        try:
            SHM_SIZE = kernel.metadata["shared"]
        except TypeError:
            SHM_SIZE = kernel.metadata.shared
        KERNEL_NAME = "default"
        try:
            KERNEL_NAME = kernel.metadata["name"]
        except TypeError:
            KERNEL_NAME = kernel.metadata.name
        print("//shared_memory:", SHM_SIZE, end=", ", file=_f)
        print("kernel_name:", KERNEL_NAME, file=_f)
        print(kernel.asm['ptx'], file=_f)
        print(kernel.metadata.shared)
    exit(0)
