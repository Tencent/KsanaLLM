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
# [vLLM Project] https://github.com/vllm-project/vllm/blob/72c2b68dc9d4fb20eb135c22ee8c86caca48d28b/vllm/model_executor/layers/fused_moe/fused_moe.py#L224 and
# [Sglang Project] https://github.com/sgl-project/sglang/blob/9858113c336f4565a0a35f9a990cdada0de1988f/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py#L459
#
# ==============================================================================

import argparse
import os
import sys
import torch
import json
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

os.environ["PATH"] = "/usr/local/nvidia/lib64:" + os.environ["PATH"]

from typing import Any, Dict, List, Optional, Tuple, Union

import triton
import triton.language as tl


@triton.jit
def fused_moe_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        # Matrix dimensions
        N,
        K,
        EM,
        num_valid_tokens,
        # Block size for block-wise quantization
        group_n: tl.constexpr,
        group_k: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        use_fp8_w8a8: tl.constexpr,
        use_int8_w8a16: tl.constexpr,
        even_Ks: tl.constexpr):
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
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * K + offs_k[None, :])

    b_ptrs = b_ptr + off_experts * N * K + (offs_k[:, None] + offs_bn[None, :] * K)
    if use_int8_w8a16:
        b_scale_ptrs = b_scale_ptr + off_experts * N // 128 * K // 128 + offs_bn[
            None, :] * K // 128
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * K // 128
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (b_scale_ptr + off_experts * N // 128 * K // 128 +
                            offs_bsn * K // 128)
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if even_Ks:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
            b = tl.load(b_ptrs,
                        mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                        other=0.0)
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(a_scale_ptrs + offs_ks,
                                  mask=token_mask,
                                  other=0.0)
                b_scale = tl.load(b_scale_ptrs + offs_ks)

                accumulator += tl.dot(a, b) * a_scale[:,
                                                      None] * b_scale[None, :]
            else:
                accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]
    if use_int8_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + N * offs_token[:, None] + offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def fused_moe(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    block_shape: List[int],
):
    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.shape[0],
                 A.shape[0] * top_k * config['BLOCK_SIZE_M'])
    
    K = B.shape[2]
    if K % config["BLOCK_SIZE_K"] != 0 and args.even_Ks:
        assert False, "K must be divisible by BLOCK_SIZE_K when even_Ks is True"

    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
        B.shape[1], META['BLOCK_SIZE_N']), )
    kernel = fused_moe_kernel[grid](
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
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        even_Ks=args.even_Ks,
        **config,
    )
    return C, kernel


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
    parser.add_argument('--group_n', type=int, default=0)
    parser.add_argument('--group_k', type=int, default=0)
    parser.add_argument('--BLOCK_SIZE_M', type=int, required=True)
    parser.add_argument('--BLOCK_SIZE_N', type=int, required=True)
    parser.add_argument('--BLOCK_SIZE_K', type=int, required=True)
    parser.add_argument('--GROUP_SIZE_M', type=int, required=True)
    parser.add_argument('--num_warps', type=int, default=0)
    parser.add_argument('--num_stages', type=int, default=0)
    parser.add_argument('--MUL_ROUTED_WEIGHT', type=str_to_bool, required=True)
    parser.add_argument('--top_k', type=int, required=True)
    parser.add_argument('--compute_type',
                        type=str,
                        required=True,
                        choices=["FP16", "BF16"])
    parser.add_argument('--use_fp8_w8a8', type=str_to_bool, required=True)
    parser.add_argument('--use_int8_w8a16', type=str_to_bool, required=True)
    parser.add_argument('--m', type=int, default=128)
    parser.add_argument('--k', type=int, default=7168)
    parser.add_argument('--n', type=int, default=4096)
    parser.add_argument('--num_experts', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default="./")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--even_Ks", type=str_to_bool, default="False")
    args = parser.parse_args()
    if args.compute_type == "FP16":
        args.torch_dtype = torch.float16
        args.triton_compute_type = tl.float16
    elif args.compute_type == "BF16":
        args.torch_dtype = torch.bfloat16
        args.triton_compute_type = tl.bfloat16
    return args


def performance_fused_moe(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    topk: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    block_shape: List[int],
):
    # for generate the jit code
    output_tensor, kernel = fused_moe(A, B, C, A_scale, B_scale, topk_weights,
                                      topk_ids, sorted_token_ids, expert_ids,
                                      num_tokens_post_padded,
                                      mul_routed_weight, topk, config,
                                      compute_type, use_fp8_w8a8,
                                      use_int8_w8a16, block_shape)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            fused_moe(A, B, C, A_scale, B_scale, topk_weights, topk_ids,
                      sorted_token_ids, expert_ids, num_tokens_post_padded,
                      mul_routed_weight, topk, config, compute_type,
                      use_fp8_w8a8, use_int8_w8a16, block_shape)
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
    torch.manual_seed(42)
    args = args_config()

    m = args.m
    k = args.k
    n = args.n
    num_experts = args.num_experts
    topk = args.top_k
    mul_routed_weight = args.MUL_ROUTED_WEIGHT
    compute_type = args.triton_compute_type
    block_shape = [args.group_n, args.group_k]
    use_fp8_w8a8 = args.use_fp8_w8a8
    use_int8_w8a16 = args.use_int8_w8a16
    numel = m * topk
    num_iters = 20

    config = {
        "BLOCK_SIZE_M": args.BLOCK_SIZE_M,
        "BLOCK_SIZE_N": args.BLOCK_SIZE_N,
        "BLOCK_SIZE_K": args.BLOCK_SIZE_K,
        "GROUP_SIZE_M": args.GROUP_SIZE_M
    }
    if args.num_warps > 0:
        config["num_warps"] = args.num_warps
    if args.num_stages > 0:
        config["num_stages"] = args.num_stages

    em = numel + num_experts * (config["BLOCK_SIZE_M"] - 1)
    max_num_m_blocks = (em + config["BLOCK_SIZE_M"] -
                        1) // config["BLOCK_SIZE_M"]

    if args.use_fp8_w8a8:
        input_dtype = torch.float8_e4m3fn
    else:
        input_dtype = args.torch_dtype

    A = torch.rand([m, k], device='cuda', dtype=torch.float32).to(input_dtype)
    B = torch.rand([num_experts, n, k], device='cuda', dtype=torch.float32).to(input_dtype)
    C = torch.rand([m, topk, n], device='cuda', dtype=args.torch_dtype)
    A_scale = torch.rand([m, k // 128], device='cuda', dtype=torch.float32)
    B_scale = torch.rand([num_experts, n // 128, k // 128],
                          device='cuda',
                          dtype=torch.float32)
    topk_weights = torch.rand([m, topk], device='cuda', dtype=torch.float32)
    topk_ids = torch.randint(size=[m, topk],
                             low=0,
                             high=num_experts,
                             device='cuda',
                             dtype=torch.int32)
    sorted_token_ids = torch.randint(size=[em],
                                     low=0,
                                     high=num_experts,
                                     device='cuda',
                                     dtype=torch.int32)
    expert_ids = torch.randint(size=[max_num_m_blocks],
                               low=0,
                               high=num_experts,
                               device='cuda',
                               dtype=torch.int32)
    num_tokens_post_padded = torch.randint(size=[1],
                                           low=0,
                                           high=1,
                                           device='cuda',
                                           dtype=torch.int32)

    # NOTE(karlluo): set best config as the best config
    candidate_configs = {
        "configs": [],
        "default": {
            "config": config,
            "kernel_time": 0.0,  # us
            "kernel": None,
        }
    }

    default_kernel, kernel_time, output_tensor = performance_fused_moe(
        A, B, C, A_scale, B_scale, topk_weights, topk_ids, sorted_token_ids,
        expert_ids, num_tokens_post_padded, mul_routed_weight, topk, config,
        compute_type, use_fp8_w8a8, use_int8_w8a16, block_shape)
    candidate_configs["default"]["kernel_time"] = kernel_time
    candidate_configs["default"]["kernel"] = default_kernel

    if args.tune:
        # using the same search space as vllm in vllm/benchmarks/kernels/benchmark_moe.py
        for num_warps in [4, 8]:
            for num_stages in [2, 3, 4, 5]:
                opt_config = config.copy()
                opt_config.update({"num_warps": num_warps, "num_stages": num_stages})

                kernel, kernel_time, output_tensor = performance_fused_moe(
                    A, B, C, A_scale, B_scale, topk_weights, topk_ids,
                    sorted_token_ids, expert_ids, num_tokens_post_padded,
                    mul_routed_weight, topk, opt_config, compute_type,
                    use_fp8_w8a8, use_int8_w8a16, block_shape)
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

    kernel_name = "fused_moe_kernel" + \
                  f"_group_n_{args.group_n}" + \
                  f"_group_k_{args.group_k}" + \
                  f"_BLOCK_SIZE_M_{args.BLOCK_SIZE_M}" + \
                  f"_BLOCK_SIZE_N_{args.BLOCK_SIZE_N}" + \
                  f"_BLOCK_SIZE_K_{args.BLOCK_SIZE_K}" + \
                  f"_GROUP_SIZE_M_{args.GROUP_SIZE_M}" + \
                  f"_MUL_ROUTED_WEIGHT_{args.MUL_ROUTED_WEIGHT}" + \
                  f"_top_k_{args.top_k}" + \
                  f"_compute_type_{args.compute_type}" + \
                  f"_use_fp8_w8a8_{args.use_fp8_w8a8}" + \
                  f"_use_int8_w8a16_{args.use_int8_w8a16}"
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
