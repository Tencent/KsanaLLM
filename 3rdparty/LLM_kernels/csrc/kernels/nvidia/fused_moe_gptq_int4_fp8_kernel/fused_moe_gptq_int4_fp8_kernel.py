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

@triton.jit
def fused_moe_gptq_int4_fp8_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        N,
        K,
        EM,
        num_valid_tokens,
        block_k_diviable: tl.constexpr,
        group_size: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr):
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
    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * K +
                      offs_k[None, :])

    b_ptrs = b_ptr + off_experts * (N * K) + \
        offs_k[:, None] + offs_bn[None, :] * K

    a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * K // 128

    stride_bse = N * K // group_size
    stride_bsn = K // group_size

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

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // 128
        a_scale = tl.load(a_scale_ptrs + offs_ks,
                    mask=token_mask,
                    other=0.0)

        b = tl.load(b_ptrs)

        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + \
            offs_bn * stride_bsn + \
            ((BLOCK_SIZE_K * k) // group_size)
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
        # TODO(jinxcwu) 注意这里b_scale的类型

        accumulator += tl.dot(a, b) * b_scale[None,:] * a_scale[:,None]

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K
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

def fused_moe_gptq_int4_fp8(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        mul_routed_weight: bool,
        top_k: int,
        config: Dict[str, Any],
        compute_type: tl.dtype,
        block_shape: List[int],
    ):
    EM = sorted_token_ids.shape[0] 
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.shape[0],
                 A.shape[0] * top_k * config['BLOCK_SIZE_M'])

    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
        B.shape[1], META['BLOCK_SIZE_N']), )
    kernel = fused_moe_gptq_int4_fp8_kernel[grid](
                A,
                B,
                C,
                A_scale,
                B_scale,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                B.shape[1],
                A.shape[1],
                EM,
                topk_ids.numel(),
                block_k_diviable=A.shape[1] % config["BLOCK_SIZE_K"] == 0,
                group_size=block_shape[1],
                MUL_ROUTED_WEIGHT=mul_routed_weight,
                top_k=top_k,
                compute_type=compute_type,
                **config,
            )
    return C, kernel

def str_to_bool(value):
    if value.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif value.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Invalid boolean value: {}'.format(value))

def args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--BLOCK_SIZE_M', type=int, required=True)
    parser.add_argument('--BLOCK_SIZE_N', type=int, required=True)
    parser.add_argument('--BLOCK_SIZE_K', type=int, required=True)
    parser.add_argument('--GROUP_SIZE_M', type=int, required=True)

    parser.add_argument('--MUL_ROUTED_WEIGHT', type=str_to_bool, required=True)
    parser.add_argument('--top_k', type=int, required=True)

    parser.add_argument('--compute_type', type=str, required=True, choices=["FP16", "BF16"])

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

def performance_fused_moe_gptq_int4_fp8(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        A_scale: Optional[torch.Tensor],
        B_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        mul_routed_weight: bool,
        topk: int,
        config: Dict[str, Any],
        compute_type: tl.dtype,
        block_shape: List[int],
):
    # for generate the jit code
    output_tensor, kernel = fused_moe_gptq_int4_fp8(A, B, C, A_scale, B_scale,
                                                    topk_weights, topk_ids, sorted_token_ids,
                                                    expert_ids, num_tokens_post_padded,
                                                    mul_routed_weight, topk, config,
                                                    compute_type, block_shape)

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            fused_moe_gptq_int4_fp8(A, B, C, A_scale, B_scale,
                                    topk_weights, topk_ids, sorted_token_ids,
                                    expert_ids, num_tokens_post_padded,
                                    mul_routed_weight, topk, config,
                                    compute_type, block_shape)
    torch.cuda.synchronize()
    # Warmup
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies: List[float] = []

    for i in range(num_iters):
        torch.cuda.synchronize()

        start_event.record()
        graph.replay()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    kernel_time = sum(latencies) / (num_iters) * 1000  # us
    graph.reset()

    return kernel, kernel_time, output_tensor

def dump_kernel(kernel, output_dir, kernel_name, config):
    with open(f"{output_dir}/{kernel_name}.cubin", "wb") as _f:
        _f.write(kernel.asm['cubin'])
    with open(f"{output_dir}/{kernel_name}.json", "w") as _f:
        json_dict = {"shm_size": kernel.metadata.shared}
        if config.get("config", {}).get("num_warps", None) is not None and config.get("config", {}).get(
                "num_stages", None) is not None:
            json_dict["num_warps"] = config.get("config").get("num_warps")
            json_dict["num_stages"] = config.get("config").get("num_stages")
        _f.write(json.dumps(json_dict))
    with open(f"{output_dir}/{kernel_name}.ptx", "w") as _f:
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


if __name__ == "__main__":
    torch.manual_seed(0)
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
    input_dtype = args.torch_dtype
    numel = m * topk
    num_iters = 20

    config = {
        "BLOCK_SIZE_M": args.BLOCK_SIZE_M,
        "BLOCK_SIZE_N": args.BLOCK_SIZE_N,
        "BLOCK_SIZE_K": args.BLOCK_SIZE_K,
        "GROUP_SIZE_M": args.GROUP_SIZE_M
    }

    em = numel + num_experts * (config["BLOCK_SIZE_M"] - 1)
    max_num_m_blocks = (em + config["BLOCK_SIZE_M"] - 1) // config["BLOCK_SIZE_M"]

    A = torch.randn([m, k], device='cuda', dtype=torch.float32).to(torch.float8_e4m3fn)
    B = torch.randn([num_experts, n, k], device='cuda', dtype=torch.float32).to(torch.float8_e4m3fn)
    C = torch.empty([m, topk, n], device='cuda', dtype=input_dtype)
    A_scale = torch.randn([m, k // 128], device='cuda', dtype=torch.float32)
    B_scale = torch.randn([num_experts, n, k // group_size], device='cuda', dtype=input_dtype)

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

    default_kernel, kernel_time, output_tensor = performance_fused_moe_gptq_int4_fp8(
                                            A, B, C, A_scale, B_scale,
                                            topk_weights, topk_ids, sorted_token_ids,
                                            expert_ids, num_tokens_post_padded,
                                            mul_routed_weight, topk, config,
                                            compute_type, block_shape)
    candidate_configs["default"]["kernel_time"] = kernel_time
    candidate_configs["default"]["kernel"] = default_kernel

    # using the same search space as vllm in vllm/benchmarks/kernels/benchmark_moe.py
    for num_warps in [4, 8]:
        for num_stages in [2, 3, 4, 5]:
            opt_config = {"num_warps": num_warps, "num_stages": num_stages}
            opt_config.update(config)

            kernel, kernel_time, output_tensor = performance_fused_moe_gptq_int4_fp8(
                                            A, B, C, A_scale, B_scale,
                                            topk_weights, topk_ids, sorted_token_ids,
                                            expert_ids, num_tokens_post_padded,
                                            mul_routed_weight, topk, opt_config,
                                            compute_type, block_shape)
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

    kernel_name = "fused_moe_gptq_int4_fp8_kernel" + \
                  f"_BLOCK_SIZE_M_{args.BLOCK_SIZE_M}" + \
                  f"_BLOCK_SIZE_N_{args.BLOCK_SIZE_N}" + \
                  f"_BLOCK_SIZE_K_{args.BLOCK_SIZE_K}" + \
                  f"_GROUP_SIZE_M_{args.GROUP_SIZE_M}" + \
                  f"_MUL_ROUTED_WEIGHT_{args.MUL_ROUTED_WEIGHT}" + \
                  f"_top_k_{args.top_k}" + \
                  f"_compute_type_{args.compute_type}" + \
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
