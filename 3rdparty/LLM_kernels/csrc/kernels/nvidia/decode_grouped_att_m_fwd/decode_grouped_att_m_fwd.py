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


@triton.jit()
def decode_grouped_att_m_fwd_kernel(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if kv_group_num > BLOCK_H:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = cur_batch

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[
        None, :]
    q = tl.load(Q + offs_q,
                mask=(mask_h[:, None]) & (mask_d[None, :]),
                other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (cur_batch * stride_qbs + cur_head[:, None] * stride_qh +
                   offs_dpe[None, :])
        qpe = tl.load(Q + off_qpe,
                      mask=(mask_h[:, None]) & (mask_dpe[None, :]),
                      other=0.0)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split,
                              cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_page_number = tl.load(
                Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx +
                offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
            ).to(tl.int64)
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
            offs_buf_k = (kv_loc[None, :] * stride_buf_kbs +
                          cur_kv_head * stride_buf_kh + offs_d[:, None])
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = (kv_loc[None, :] * stride_buf_kbs +
                                cur_kv_head * stride_buf_kh +
                                offs_dpe[:, None])
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) &
                    (mask_dpe[:, None]),
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(mask_h[:, None] & (offs_n[None, :] < split_kv_end),
                          qk, float("-inf"))

            offs_buf_v = (kv_loc[:, None] * stride_buf_vbs +
                          cur_kv_head * stride_buf_vh + offs_dv[None, :])
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (cur_batch * stride_mid_ob +
                      cur_head[:, None] * stride_mid_oh +
                      split_kv_id * stride_mid_os + offs_dv[None, :])

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh +
                        split_kv_id * stride_mid_os + Lv)

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    Req_to_tokens,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap,
):
    BLOCK = 32
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    BLOCK_H = 16
    NUM_KV_SPLITS = num_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_KV_SPLITS,
    )

    extra_kargs = {}
    kernel = decode_grouped_att_m_fwd_kernel[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        Req_to_tokens,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        k_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        num_warps=4,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
        **extra_kargs,
    )
    return att_out, kernel


def str_to_bool(value):
    if value.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif value.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(
            'Invalid boolean value: {}'.format(value))


def args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kv_group_num', type=int, default=16)
    parser.add_argument('--q_head_num', type=int, default=16)
    parser.add_argument('--BLOCK_DMODEL', type=int, default=512)
    parser.add_argument('--BLOCK_DPE', type=int, default=64)
    parser.add_argument('--BLOCK_DV', type=int, default=512)
    parser.add_argument('--BLOCK_N', type=int, default=32)
    parser.add_argument('--BLOCK_H', type=int, default=16)
    parser.add_argument('--NUM_KV_SPLITS', type=int, default=4)
    parser.add_argument('--PAGE_SIZE', type=int, default=16)
    parser.add_argument('--Lk', type=int, default=576)
    parser.add_argument('--Lv', type=int, default=512)
    parser.add_argument('--input_type',
                        type=str,
                        required=True,
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
    num_heads = args.kv_group_num
    kv_lora_rank = args.BLOCK_DMODEL
    qk_rope_head_dim = args.BLOCK_DPE
    sm_scale = 0.1147213867929261
    page_size = args.PAGE_SIZE
    logit_cap = 0.0
    block_num = 3
    max_blocks_per_seq = 2
    input_type = args.torch_dtype

    q = torch.zeros([batch, num_heads, kv_lora_rank + qk_rope_head_dim],
                    device='cuda',
                    dtype=input_type)
    k_buffer = torch.zeros(
        [block_num, page_size, 1, kv_lora_rank + qk_rope_head_dim],
        device='cuda',
        dtype=input_type)
    v_buffer = torch.zeros([block_num, page_size, 1, kv_lora_rank],
                           device='cuda',
                           dtype=input_type)
    req_to_token = torch.zeros([batch, max_blocks_per_seq],
                               device='cuda',
                               dtype=torch.int32)
    b_seq_len = torch.zeros([batch], device='cuda', dtype=torch.int32)
    att_out = torch.zeros([batch, num_heads, num_kv_splits, kv_lora_rank + 1],
                          device='cuda',
                          dtype=torch.float32)

    output_triton, kernel = decode_grouped_att_m_fwd(q, k_buffer, v_buffer,
                                                     att_out, req_to_token,
                                                     b_seq_len, num_kv_splits,
                                                     sm_scale, page_size,
                                                     logit_cap)

    kernel_name = "decode_grouped_att_m_fwd_kernel" + \
                  f"_kv_group_num_{args.kv_group_num}" + \
                  f"_q_head_num_{args.q_head_num}" + \
                  f"_BLOCK_DMODEL_{args.BLOCK_DMODEL}" + \
                  f"_BLOCK_DPE_{args.BLOCK_DPE}" + \
                  f"_BLOCK_DV_{args.BLOCK_DV}" + \
                  f"_BLOCK_N_{args.BLOCK_N}" + \
                  f"_BLOCK_H_{args.BLOCK_H}" + \
                  f"_NUM_KV_SPLITS_{args.NUM_KV_SPLITS}" + \
                  f"_PAGE_SIZE_{args.PAGE_SIZE}" + \
                  f"_Lk_{args.Lk}" + \
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
    exit(0)
