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
# [vLLM Project] https://github.com/vllm-project/vllm/blob/f35f8e2242db224a92a14e084d502eec67d56da9/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L179
#
# ==============================================================================

import argparse
import os
import json
import torch

os.environ["PATH"] = "/usr/local/nvidia/lib64:" + os.environ["PATH"]

from typing import Any, Dict, List, Optional, Tuple, Union

import triton
import triton.language as tl


@triton.jit
def _per_token_group_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size: tl.constexpr,
    # Num columns of y
    y_num_columns,
    # y_row_stride,
    # Avoid to divide zero
    eps: tl.constexpr,
    # Information for float8
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size

    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr += (row * y_num_columns) + (row_g_id * group_size)
    y_q_ptr += g_id * group_size
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_colmajor(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size: tl.constexpr,
    # Num columns of y
    y_num_columns,
    # y_row_stride,
    # Stride from one column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps: tl.constexpr,
    # Information for float8
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size

    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr += (row * y_num_columns) + (row_g_id * group_size)
    y_q_ptr += g_id * group_size

    # Convert g_id the flattened block coordinate to 2D so we can index
    # into the output y_scales matrix
    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    y_s_ptr += scale_col * y_s_col_stride + scale_row

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int = 128,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.
    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `torch.float8_e4m3fn`
        is supported for now.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
        scaling factor for quantization.
    """
    if dtype is None:
        dtype = torch.float8_e4m3fn
    assert (x.shape[-1] % group_size == 0), (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}")
    assert x.is_contiguous(), "`x` must be contiguous"

    fp8_min = -448.0
    fp8_max = 448.0

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    M = x.numel() // group_size  #
    N = group_size  # 128
    if column_major_scales:
        shape = (x.shape[-1] // group_size, ) + x.shape[:-1]
        x_s = torch.empty(shape, device=x.device,
                          dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size, )
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    BLOCK = triton.next_power_of_2(N)

    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1

    if column_major_scales:
        kernel = _per_token_group_quant_fp8_colmajor[(M, )](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            # x.stride(0),
            x_s.stride(1),
            eps,
            fp8_min,
            fp8_max,
            BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        kernel = _per_token_group_quant_fp8[(M, )](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            # x.stride(0),
            eps,
            fp8_min,
            fp8_max,
            BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return x_q, x_s, kernel


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
    parser.add_argument('--compute_type',
                        type=str,
                        required=True,
                        choices=["FP16", "BF16"])
    parser.add_argument('--colmajor', type=str_to_bool, required=True)
    parser.add_argument('--output_dir', type=str, default="./")
    args = parser.parse_args()
    if args.compute_type == "FP16":
        args.torch_dtype = torch.float16
        args.triton_compute_type = tl.float16
    elif args.compute_type == "BF16":
        args.torch_dtype = torch.bfloat16
        args.triton_compute_type = tl.bfloat16
    return args


if __name__ == "__main__":
    torch.manual_seed(0)
    args = args_config()
    data = torch.rand([66, 2560], device='cuda', dtype=args.torch_dtype)

    output_triton_a, output_triton_b, kernel = per_token_group_quant_fp8( \
        data, group_size=128, column_major_scales=args.colmajor)

    kernel_name = "_per_token_group_quant_fp8" + \
                  ("_colmajor" if args.colmajor else "") + \
                  f"_compute_type_{args.compute_type}" + \
                  f"_colmajor_{args.colmajor}"

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
