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
from transformers import AutoConfig
from tqdm import tqdm

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


def two_step_fused_moe(
    up_gate_proj_dict: Dict[str, torch.Tensor],
    down_proj_dict: Dict[str, torch.Tensor],
    gate_dict: Dict[str, torch.Tensor],
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    block_shape: List[int],
):
    fused_moe(**up_gate_proj_dict, **gate_dict,
              mul_routed_weight=False,
              top_k=top_k, config=config,
              compute_type=compute_type,
              use_fp8_w8a8=use_fp8_w8a8,
              use_int8_w8a16=use_int8_w8a16,
              block_shape=block_shape)
    fused_moe(**down_proj_dict, **gate_dict,
              mul_routed_weight=True,
              top_k=1, config=config,
              compute_type=compute_type,
              use_fp8_w8a8=use_fp8_w8a8,
              use_int8_w8a16=use_int8_w8a16,
              block_shape=block_shape)


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
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to model configuration file. If provided, input settings"
        "(k, n, num_experts, topk, block_shape) will be overridden by the model configuration. "
    )
    parser.add_argument(
        "--deep_tune",
        action="store_true",
        help=(
            "Deep_tune will enable two-step fused_moe_kernel and "
            "automatically tune BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, and GROUP_SIZE_M for optimal performance. "
            "Note: No Triton kernel will be generated in this mode; instead, a JSON file with the best configuration will be saved."
        )
    )
    parser.add_argument("--num_iters", type=int, default=5)
    parser.add_argument("--tp_size", type=int, default=1)
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


def performance_two_step_fused_moe(
    up_gate_proj_dict: Dict[str, torch.Tensor],
    down_proj_dict: Dict[str, torch.Tensor],
    gate_dict: Dict[str, torch.Tensor],
    mul_routed_weight: bool,
    topk: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    block_shape: List[int],
):
    two_step_fused_moe(up_gate_proj_dict, down_proj_dict, gate_dict,
                       mul_routed_weight, topk, config, compute_type,
                       use_fp8_w8a8, use_int8_w8a16, block_shape)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            two_step_fused_moe(up_gate_proj_dict, down_proj_dict, gate_dict,
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

    return kernel_time


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


def get_weight_block_size_safety(config, default_value=None):
    quantization_config = getattr(config, 'quantization_config', {})
    if isinstance(quantization_config, dict):
        return quantization_config.get('weight_block_size', default_value)
    return default_value

# Adapted from vLLM: removed some unneeded code
# https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py#L287
def if_skip_config(M, N, config):
    large_gemm = False
    if M >= 2048 and N >= 2048:
        large_gemm = True

    BLOCK_SIZE_M = config["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = config["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = config["BLOCK_SIZE_K"]
    GROUP_M = config["GROUP_SIZE_M"]
    num_warps = config["num_warps"]
    
    # some layouts could not work properly in case
    # number elements per thread is less 1
    if BLOCK_SIZE_M * BLOCK_SIZE_N < 64:
        return True
    # Skip BLOCK_SIZE that is too large compare to M/N
    # unless BLOCK_SIZE is already small enough
    if M * 2 < BLOCK_SIZE_M and BLOCK_SIZE_M != 16:
        return True
    if N * 2 < BLOCK_SIZE_N and BLOCK_SIZE_N != 16:
        return True
    # skip large GROUP_M
    if GROUP_M * BLOCK_SIZE_M > M and GROUP_M != 1:
        return True
    # Skip small block sizes and num_warps for large gemm
    # For fp16 and f8, we want to only use BLOCK_SIZE >= 64
    if large_gemm:
        if BLOCK_SIZE_M < 64 or BLOCK_SIZE_N < 64:
            return True
        if BLOCK_SIZE_K < 64:
            return True
        if num_warps < 4:
            return True

    return False

# get expert_ids, sorted_ids, token_post_pad according to block_size_m, token_post_pad will influence the MOE execution latency
def moe_align_block_cpu(topk_ids, expert_ids, sorted_ids, token_post_pad, token_num, topk, expert_num, block_size):
    cumsum = [0] * (expert_num + 1)
    token_cnts = [0] * (expert_num + 1)
    numel = token_num * topk
    sorted_ids.fill_(numel)
    expert_ids.fill_(-1)

    for i in range(numel):
        expert_id = topk_ids[i]
        if expert_id >= expert_num:
            continue
        token_cnts[expert_id] += 1

    for i in range(expert_num):
        cumsum[i + 1] = cumsum[i] + \
            (token_cnts[i] + block_size - 1) // block_size
        token_cnts[i] = 0
        for j in range(cumsum[i], cumsum[i + 1]):
            expert_ids[j] = i

    token_post_pad[0] = cumsum[expert_num] * block_size

    for i in range(numel):
        expert_id = topk_ids[i]
        if expert_id >= expert_num:
            continue
        idx = cumsum[expert_id] * block_size + token_cnts[expert_id]
        sorted_ids[idx] = i
        token_cnts[expert_id] += 1


if __name__ == "__main__":
    torch.manual_seed(42)
    args = args_config()
    if not args.tune and args.deep_tune:
        logging.error("can not set deep_tune=true when tune=false")
        sys.exit(1)

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
    num_iters = args.num_iters

    if args.model_path != "":
        model_config = AutoConfig.from_pretrained(
            args.model_path, trust_remote_code=True)
        if (model_config.architectures[0] == "DeepseekV3ForCausalLM"
                or model_config.architectures[0] == "DeepseekV2ForCausalLM"):
            block_shape_override = get_weight_block_size_safety(model_config)
            if block_shape_override is None:
                block_shape_override = [0, 0]
            logging.info(
                "Load model config - these inputs will be ignored and have been overridden: \n"
                "k:{}->{}, n:{}->{}, num_experts:{}->{}, topk:{}->{}, block_shape(groupn,groupk):[{},{}]-> [{},{}].".format(
                    k, model_config.hidden_size,
                    n, model_config.moe_intermediate_size,
                    num_experts, model_config.n_routed_experts,
                    topk, model_config.num_experts_per_tok,
                    block_shape[0], block_shape[1], block_shape_override[0], block_shape_override[1]
                )
            )
            k = model_config.hidden_size
            n = model_config.moe_intermediate_size
            num_experts = model_config.n_routed_experts
            topk = model_config.num_experts_per_tok
            block_shape = block_shape_override
        else:
            logging.error("no support for this model")
            exit(1)
    numel = m * topk

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

    up_gate_proj_dict = {}
    up_gate_proj_dict["A"] = torch.rand(
        [m, k], device='cuda', dtype=torch.float32).to(input_dtype)
    up_gate_proj_dict["B"] = torch.rand(
        [num_experts, n * 2 // args.tp_size, k], device='cuda', dtype=torch.float32).to(input_dtype)
    up_gate_proj_dict["C"] = torch.rand(
        [m, topk, n * 2 // args.tp_size], device='cuda', dtype=args.torch_dtype)
    up_gate_proj_dict["A_scale"] = torch.rand(
        [m, k // 128], device='cuda', dtype=torch.float32)
    up_gate_proj_dict["B_scale"] = torch.rand([num_experts, n * 2 // args.tp_size // 128, k // 128],
                                              device='cuda',
                                              dtype=torch.float32)
    down_proj_dict = {}
    if args.deep_tune:
        down_proj_dict["A"] = torch.rand(
            [m * topk, n // args.tp_size], device='cuda', dtype=torch.float32).to(input_dtype)
        down_proj_dict["B"] = torch.rand(
            [num_experts, k, n // args.tp_size], device='cuda', dtype=torch.float32).to(input_dtype)
        down_proj_dict["C"] = torch.rand(
            [m, topk, k], device='cuda', dtype=args.torch_dtype)
        down_proj_dict["A_scale"] = torch.rand(
            [m * topk, n // args.tp_size // 128], device='cuda', dtype=torch.float32)
        down_proj_dict["B_scale"] = torch.rand([num_experts, k // 128, n // args.tp_size // 128],
                                               device='cuda',
                                               dtype=torch.float32)

    topk_weights = torch.rand([m, topk], device='cuda', dtype=torch.float32)
    torch.manual_seed(42)
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
# todo(pengfeilei): moe_align_block_cpu initializes num_tokens_post_padded, which determines MOE execution latency.
# Behavior depends on whether deep_tune is enabled:
# 1. If deep_tune is not enabled (default behavior):
#    - num_tokens_post_padded = 0 (tuning is disabled)
# 2. If deep_tune is enabled:
#    - num_tokens_post_padded is properly set for tuning
    if args.deep_tune:
        moe_align_block_cpu(topk_ids.flatten(), expert_ids, sorted_token_ids,
                            num_tokens_post_padded, m, topk, num_experts, config["BLOCK_SIZE_M"])

    gate_dict = {
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "sorted_token_ids": sorted_token_ids,
        "expert_ids": expert_ids,
        "num_tokens_post_padded": num_tokens_post_padded,
    }

    # NOTE(karlluo): set best config as the best config
    candidate_configs = {
        "configs": [],
        "default": {
            "config": config,
            "kernel_time": 0.0,  # us
            "kernel": None,
        }
    }

    if args.deep_tune:
        kernel_time = performance_two_step_fused_moe(
            up_gate_proj_dict, down_proj_dict, gate_dict,
            mul_routed_weight, topk, config, compute_type, use_fp8_w8a8,
            use_int8_w8a16, block_shape)
        candidate_configs["default"]["kernel_time"] = kernel_time
        candidate_configs["default"]["kernel"] = None
    else:
        default_kernel, kernel_time, output_tensor = performance_fused_moe(
            **up_gate_proj_dict, **gate_dict, mul_routed_weight=mul_routed_weight, topk=topk, config=config,
            compute_type=compute_type, use_fp8_w8a8=use_fp8_w8a8, use_int8_w8a16=use_int8_w8a16, block_shape=block_shape)
        candidate_configs["default"]["kernel_time"] = kernel_time
        candidate_configs["default"]["kernel"] = default_kernel

    # vllm search space
    # block_m_range = [16, 32, 64, 128, 256]
    # block_n_range = [32, 64, 128, 256]
    # block_k_range = [64, 128, 256]
    # group_m_range = [1, 16, 32, 64]
    # num_warps_range = [4, 8]
    # num_stage_range = [2, 3, 4, 5]
    if args.deep_tune:
        block_size_m_list = [16, 32, 64, 128]
        block_size_n_list = [64, 128]
        block_size_k_list = [64, 128]
        group_size_m_list = [1, 16, 32, 64]
    else:
        block_size_m_list = [args.BLOCK_SIZE_M]
        block_size_n_list = [args.BLOCK_SIZE_N]
        block_size_k_list = [args.BLOCK_SIZE_K]
        group_size_m_list = [args.GROUP_SIZE_M]
    num_warps_list = [4, 8]
    num_stages_list = [2, 3, 4, 5]
    total_loops = len(block_size_m_list) * len(block_size_n_list) * len(block_size_k_list) * \
        len(group_size_m_list) * len(num_warps_list) * len(num_stages_list)
    if args.tune:
        # using the same search space as vllm in vllm/benchmarks/kernels/benchmark_moe.py
        with tqdm(total=total_loops, desc="deep tune", disable=not args.deep_tune) as pbar:
            for block_size_m in block_size_m_list:
                em = numel + num_experts * (block_size_m - 1)
                max_num_m_blocks = (em + block_size_m - 1) // block_size_m
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
                num_tokens_post_padded = torch.empty(
                    (1,), dtype=torch.int32, device='cuda').flatten()
                if args.deep_tune:
                    moe_align_block_cpu(topk_ids.flatten(
                    ), expert_ids, sorted_token_ids, num_tokens_post_padded, m, topk, num_experts, block_size_m)
                gate_dict["expert_ids"] = expert_ids
                gate_dict["sorted_token_ids"] = sorted_token_ids
                gate_dict["num_tokens_post_padded"] = num_tokens_post_padded
                for block_size_n in block_size_n_list:
                    for block_size_k in block_size_k_list:
                        for group_size_m in group_size_m_list:
                            for num_warps in num_warps_list:
                                for num_stages in num_stages_list:
                                    opt_config = {
                                        "BLOCK_SIZE_M": block_size_m,
                                        "BLOCK_SIZE_N": block_size_n,
                                        "BLOCK_SIZE_K": block_size_k,
                                        "GROUP_SIZE_M": group_size_m,
                                        "num_warps": num_warps,
                                        "num_stages": num_stages,
                                    }
                                    if if_skip_config(m, n, opt_config):
                                        continue
                                    try:
                                        if args.deep_tune:
                                            kernel_time = performance_two_step_fused_moe(
                                                up_gate_proj_dict,
                                                down_proj_dict,
                                                gate_dict,
                                                mul_routed_weight=mul_routed_weight,
                                                topk=topk,
                                                config=opt_config,
                                                compute_type=compute_type,
                                                use_fp8_w8a8=use_fp8_w8a8,
                                                use_int8_w8a16=use_int8_w8a16,
                                                block_shape=block_shape)
                                            kernel = None
                                            output_tensor = None
                                        else:
                                            kernel, kernel_time, output_tensor = performance_fused_moe(
                                                **up_gate_proj_dict, **gate_dict,
                                                mul_routed_weight=mul_routed_weight,
                                                topk=topk,
                                                config=opt_config,
                                                compute_type=compute_type,
                                                use_fp8_w8a8=use_fp8_w8a8,
                                                use_int8_w8a16=use_int8_w8a16,
                                                block_shape=block_shape)
                                        candidate_configs["configs"].append({
                                            "config": opt_config,
                                            "kernel_time": kernel_time,
                                            "kernel": kernel
                                        })
                                    except Exception as e:
                                        logging.error(
                                            f"Error in config {opt_config}: {e}")
                                        continue
                                    pbar.update(1)
    opt_best_kernel_time = sys.float_info.max
    opt_best_kerenl_config = None
    opt_best_kernel = None
    for config in candidate_configs["configs"]:
        if opt_best_kernel_time > config["kernel_time"]:
            opt_best_kernel_time = config["kernel_time"]
            opt_best_kerenl_config = config
            opt_best_kernel = config["kernel"]

    if args.deep_tune:
        best_config_path = os.path.join(args.output_dir, "best_config.json")
        if os.path.exists(best_config_path):
            with open(best_config_path, "r") as f:
                best_config = json.load(f)
        else:
            best_config = {}
        best_config[m] = opt_best_kerenl_config["config"]
        sorted_best_config = {k: best_config[k] for k in sorted(
            best_config, key=lambda x: int(x))}
        with open(best_config_path, "w") as f:
            json.dump(sorted_best_config, f, indent=4)
        exit(0)

    # block_shape[groupn,groupk]
    kernel_name = "fused_moe_kernel" + \
                  f"_group_n_{block_shape[0]}" + \
                  f"_group_k_{block_shape[1]}" + \
                  f"_BLOCK_SIZE_M_{args.BLOCK_SIZE_M}" + \
                  f"_BLOCK_SIZE_N_{args.BLOCK_SIZE_N}" + \
                  f"_BLOCK_SIZE_K_{args.BLOCK_SIZE_K}" + \
                  f"_GROUP_SIZE_M_{args.GROUP_SIZE_M}" + \
                  f"_MUL_ROUTED_WEIGHT_{args.MUL_ROUTED_WEIGHT}" + \
                  f"_top_k_{topk}" + \
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
