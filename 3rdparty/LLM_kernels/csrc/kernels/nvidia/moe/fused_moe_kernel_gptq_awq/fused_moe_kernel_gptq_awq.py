# Copyright 2025 Tencent Inc. All rights reserved.
# Copyright 2025 vLLM Team
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
# [vLLM Project] https://github.com/vllm-project/vllm/blob/v0.8.2/vllm/model_executor/layers/fused_moe/fused_moe.py#L36
#
# ==============================================================================

import argparse
import os
import sys
import json
import torch
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

os.environ["PATH"] = "/usr/local/nvidia/lib64:" + os.environ["PATH"]

from typing import Any, Dict, List, Optional, Tuple, Union

import triton
import triton.language as tl

current_dir = os.path.dirname(os.path.abspath(__file__))
common_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, current_dir)
sys.path.insert(0, common_dir)
from common import dump_kernel, str_to_bool, set_global_seed, performance_kernel

"""
    Fused MoE Kernel GPTQ AWQ - WxA16C16混合精度计算内核

    本kernel支持：
    - uint8、uint4权重数据类型
    - gptq、awq权重类型，gptq不包含零点、awq包含零点
"""
@triton.jit
def fused_moe_gptq_awq_kernel(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        b_scale_ptr,
        b_zp_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        # Matrix dimensions
        N,
        K,
        EM,
        num_valid_tokens,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        # stride_am,
        # stride_ak,
        # stride_be,
        # stride_bk,
        # stride_bn,
        # stride_cm,
        # stride_cn,
        # stride_bse,
        # stride_bsk,
        # stride_bsn,
        # stride_bze,
        # stride_bzk,
        # stride_bzn,
        block_k_diviable: tl.constexpr,
        group_size: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        has_zp: tl.constexpr,
        use_int4_w4a16: tl.constexpr,
        use_int8_w8a16: tl.constexpr):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    # TODO(jinxcwu): 可以仿照fused_moe直接移除
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + N * offs_token[:, None] + 1 * offs_cn[
            None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * K +
                      offs_k[None, :])

    if use_int4_w4a16:
        pack_factor = 8 // 4
        b_ptrs = b_ptr + off_experts * (N * K // pack_factor) + \
            (offs_k[:, None] // 2) + offs_bn[None, :] * \
                (K // pack_factor)
        b_shifter = (offs_k[:, None] % 2) * 4
    elif use_int8_w8a16:
        pack_factor = 8 // 8
        b_ptrs = b_ptr + off_experts * (N * K // pack_factor) + \
            offs_k[:, None] + offs_bn[None, :] * (K // pack_factor)

    stride_bse = N * K // group_size
    stride_bsn = K // group_size
    stride_bze = N // pack_factor * K // group_size
    stride_bzn = K // group_size

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = (offs_bn[None, :] % 2) * 4

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.

        if not block_k_diviable:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs)
        if use_int4_w4a16:
            b = (b >> b_shifter) & 0xF

        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + \
            offs_bn[None, :] * stride_bsn + \
            ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size)
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
        b_scale = b_scale.to(tl.float32)

        if has_zp and use_int4_w4a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = b_zp_ptr + off_experts * stride_bze + \
                (offs_bn[None, :] // 2) * stride_bzn + \
                offs_k_true
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = ((b_zp >> b_zp_shifter) & 0xF)
            b_zp = b_zp.to(tl.float32)
        elif has_zp and use_int8_w8a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = b_zp_ptr + off_experts * stride_bze + \
                offs_bn[None, :] * stride_bzn + \
                offs_k_true
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = b_zp.to(tl.float32)

        # We accumulate along the K dimension.
        if has_zp:
            b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
        else:
            b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)
        accumulator = tl.dot(a, b, acc=accumulator)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K
        if use_int4_w4a16:
            b_ptrs += (BLOCK_SIZE_K // 2)
        else:
            b_ptrs += BLOCK_SIZE_K

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + N * offs_token[:, None] + offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def fused_moe_gptq_awq(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        B_scale: Optional[torch.Tensor],
        B_zp: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        mul_routed_weight: bool,
        top_k: int,
        config: Dict[str, Any],
        compute_type: tl.dtype,
        has_zp: bool,
        use_int8_w8a16: bool,
        use_int4_w4a16: bool,
        block_shape: List[int],
    ):
    EM = sorted_token_ids.shape[0] 
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.shape[0],
                 A.shape[0] * top_k * config['BLOCK_SIZE_M'])

    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
        B.shape[1], META['BLOCK_SIZE_N']), )
    kernel = fused_moe_gptq_awq_kernel[grid](
                A,
                B,
                C,
                B_scale,
                B_zp,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                B.shape[1],
                A.shape[1],
                EM,
                topk_ids.numel(),
                # A.stride(0),
                # A.stride(1),
                # B.stride(0),
                # B.stride(2),
                # B.stride(1),
                # C.stride(1),
                # C.stride(2),
                # B_scale.stride(0),
                # B_scale.stride(2),
                # B_scale.stride(1),
                # B_zp.stride(0) if B_zp is not None else 0,
                # B_zp.stride(2) if B_zp is not None else 0,
                # B_zp.stride(1) if B_zp is not None else 0,
                block_k_diviable=A.shape[1] % config["BLOCK_SIZE_K"] == 0,
                group_size=block_shape[1],
                MUL_ROUTED_WEIGHT=mul_routed_weight,
                top_k=top_k,
                compute_type=compute_type,
                has_zp=has_zp,
                use_int4_w4a16=use_int4_w4a16,
                use_int8_w8a16=use_int8_w8a16,
                **config,
            )
    return C, kernel

def args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--BLOCK_SIZE_M', type=int, required=True)
    parser.add_argument('--BLOCK_SIZE_N', type=int, required=True)
    parser.add_argument('--BLOCK_SIZE_K', type=int, required=True)
    parser.add_argument('--GROUP_SIZE_M', type=int, required=True)

    parser.add_argument('--MUL_ROUTED_WEIGHT', type=str_to_bool, required=True)
    parser.add_argument('--top_k', type=int, required=True)

    parser.add_argument('--compute_type', type=str, required=True, choices=["FP16", "BF16"])

    parser.add_argument('--has_zp', type=str_to_bool, required=True)
    parser.add_argument('--weight_bits', type=int, required=True, choices=[4, 8])
    parser.add_argument('--group_size', type=int, required=True, choices=[64, 128])

    parser.add_argument('--m', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--n', type=int, required=True)

    parser.add_argument('--num_experts', type=int, required=True)

    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument("--tune", action="store_true")

    args = parser.parse_args()
    if args.compute_type == "FP16":
        args.torch_dtype = torch.float16
        args.triton_compute_type = tl.float16
    elif args.compute_type == "BF16":
        args.torch_dtype = torch.bfloat16
        args.triton_compute_type = tl.bfloat16
    return args

if __name__ == "__main__":
    set_global_seed(0)
    args = args_config()

    # 基础配置
    m = args.m
    k = args.k
    n = args.n
    num_experts = args.num_experts
    topk = args.top_k
    mul_routed_weight = args.MUL_ROUTED_WEIGHT
    compute_type = args.triton_compute_type
    block_shape = [0, args.group_size]
    group_size = args.group_size
    has_zp = args.has_zp
    weight_bits = args.weight_bits
    input_dtype = args.torch_dtype
    numel = m * topk
    num_iters = 20

    if weight_bits == 4:
        use_int8_w8a16 = False
        use_int4_w4a16 = True
    if weight_bits == 8:
        use_int8_w8a16 = True
        use_int4_w4a16 = False

    config = {
        "BLOCK_SIZE_M": args.BLOCK_SIZE_M,
        "BLOCK_SIZE_N": args.BLOCK_SIZE_N,
        "BLOCK_SIZE_K": args.BLOCK_SIZE_K,
        "GROUP_SIZE_M": args.GROUP_SIZE_M
    }

    em = numel + num_experts * (config["BLOCK_SIZE_M"] - 1)
    max_num_m_blocks = (em + config["BLOCK_SIZE_M"] - 1) // config["BLOCK_SIZE_M"]

    pack_factor = int(8 / weight_bits)

    A = torch.empty([m, k], device='cuda', dtype=input_dtype)
    B = torch.empty([num_experts, n, k // pack_factor], device='cuda', dtype=torch.uint8)
    C = torch.empty([m, topk, n], device='cuda', dtype=input_dtype)
    B_scale = torch.empty([num_experts, n, k // group_size], device='cuda', dtype=input_dtype)
    B_zp = torch.empty([num_experts, n // pack_factor, k // group_size], device='cuda', dtype=torch.uint8)

    topk_weights = torch.rand([m, topk], device='cuda', dtype=torch.float32)
    sorted_token_ids = torch.randint(size=[em], low=0, high=num_experts, device='cuda', dtype=torch.int32)
    expert_ids = torch.randint(size=[max_num_m_blocks], low=0, high=num_experts, device='cuda', dtype=torch.int32)
    num_tokens_post_padded = torch.randint(size=[1], low=0, high=1, device='cuda', dtype=torch.int32)
    topk_ids = torch.randint(size=[m, topk], low=0, high=num_experts, device='cuda', dtype=torch.int32)

    # NOTE(karlluo): set best config as the best config
    candidate_configs = {
        "configs": [],
        "default": {
            "config": config,
            "kernel_time": 0.0,  # us
            "kernel": None,
        }
    }

    def fn():
        return fused_moe_gptq_awq(A, B, C, B_scale, B_zp,
                                  topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
                                  mul_routed_weight, topk,
                                  config, compute_type, has_zp, use_int8_w8a16, use_int4_w4a16, block_shape)
    default_kernel, kernel_time, output_tensor = performance_kernel(fn)
    candidate_configs["default"]["kernel_time"] = kernel_time
    candidate_configs["default"]["kernel"] = default_kernel

    # using the same search space as vllm in vllm/benchmarks/kernels/benchmark_moe.py
    for num_warps in [4, 8]:
        for num_stages in [2, 3, 4, 5]:
            opt_config = {"num_warps": num_warps, "num_stages": num_stages}
            opt_config.update(config)

            def fn():
                return fused_moe_gptq_awq(A, B, C, B_scale, B_zp,
                                          topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
                                          mul_routed_weight, topk,
                                          opt_config, compute_type, has_zp, use_int8_w8a16, use_int4_w4a16, block_shape)
            kernel, kernel_time, output_tensor = performance_kernel(fn)
            candidate_configs["configs"].append({
                "config": opt_config,
                "kernel_time": kernel_time,
                "kernel": kernel
            })

    opt_best_kernel_time = sys.float_info.max
    opt_best_kerenl_config = None
    opt_best_kernel = None
    for config in candidate_configs["configs"]:
        if opt_best_kernel_time > config["kernel_time"]:
            opt_best_kernel_time = config["kernel_time"]
            opt_best_kerenl_config = config
            opt_best_kernel = config["kernel"]

    kernel_name = "fused_moe_gptq_awq_kernel" + \
                  f"_BLOCK_SIZE_M_{args.BLOCK_SIZE_M}" + \
                  f"_BLOCK_SIZE_N_{args.BLOCK_SIZE_N}" + \
                  f"_BLOCK_SIZE_K_{args.BLOCK_SIZE_K}" + \
                  f"_GROUP_SIZE_M_{args.GROUP_SIZE_M}" + \
                  f"_MUL_ROUTED_WEIGHT_{args.MUL_ROUTED_WEIGHT}" + \
                  f"_top_k_{args.top_k}" + \
                  f"_compute_type_{args.compute_type}" + \
                  f"_has_zp_{args.has_zp}" + \
                  f"_weight_bits_{args.weight_bits}" + \
                  f"_group_size_{args.group_size}"
    # dump default kernel name
    dump_kernel(default_kernel, args.output_dir, kernel_name,
                candidate_configs["default"]["config"])

    if opt_best_kernel_time > candidate_configs["default"]["kernel_time"]:
        opt_best_kernel_time = sys.float_info.max
        opt_best_kerenl_config = None

    if opt_best_kerenl_config is not None and args.tune:
        dump_kernel(opt_best_kernel, args.output_dir, kernel_name,
                    opt_best_kerenl_config)
        logging.info("Found best config after tuning")
        logging.info(opt_best_kerenl_config)
        logging.info(f"Tuned best config average latency: {opt_best_kernel_time} us")
        logging.info(f"Default config average latency: {candidate_configs['default']['kernel_time']} us")
    else:
        logging.info("Using default config")
        logging.info(candidate_configs["default"]["config"])
        logging.info(
            f"Average latency: {candidate_configs['default']['kernel_time']} us"
        )
    exit(0)
