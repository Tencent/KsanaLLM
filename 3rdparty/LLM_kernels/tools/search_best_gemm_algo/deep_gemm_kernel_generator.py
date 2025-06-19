# Copyright 2024 Tencent Inc.  All rights reserved.
# Copyright (c) 2025 DeepSeek.
# Adapt from:
# https://github.com/deepseek-ai/DeepGEMM/tree/9b0dad86407609fed7b9046f9e4f298f04b2f3d5/tests/test_core.py
# ==============================================================================
import argparse
import random
import torch
import sys
import os
import glob
import shutil
import subprocess
import yaml
from typing import Tuple, Dict, Any, Optional

from cuda_data_type_enum import CudaDataType

is_deep_gemm_available = False
try:
    import deep_gemm
    from deep_gemm import bench_kineto, get_col_major_tma_aligned_tensor, ceil_div
    from deep_gemm.jit_kernels.tuner import jit_tuner
    from deep_gemm.jit_kernels.utils import get_num_sms
    from deep_gemm.jit_kernels.gemm import get_best_configs
    is_deep_gemm_available = True
except ImportError as e:
    print(e, file=sys.stderr)


def args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m',
                        type=int,
                        required=True,
                        help='GEMM M dimension')
    parser.add_argument('--n',
                        type=int,
                        required=True,
                        help='GEMM N dimension')
    parser.add_argument('--k',
                        type=int,
                        required=True,
                        help='GEMM K dimension')
    parser.add_argument('--kernel_saved_path',
                        type=str,
                        default="",
                        help='Path to save the generated kernel files')
    parser.add_argument('--tuner_device_id',
                        type=int,
                        default=0,
                        help='the card id to run the tuner on, default is 0')
    args = parser.parse_args()
    return args


def update_yaml_config(config_path: str, new_config: Dict[str, Any]) -> bool:
    existing_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            existing_config = yaml.safe_load(f) or {}

    merged_config = {**existing_config, **new_config}

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(merged_config,
                  f,
                  default_flow_style=False,
                  allow_unicode=True,
                  sort_keys=False)


def load_yaml_config(config_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(config_path):
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config
    except yaml.YAMLError as e:
        print(f"Load {config_path} fail: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Load {config_path} fail: {e}", file=sys.stderr)
        return None


def per_token_cast_to_fp8(
        x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(
        torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(
        x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128),
                           dtype=x.dtype,
                           device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (
        x_amax / 448.0).view(x_view.size(0), x_view.size(2))


def construct_deep_gemm_fp8fp8bf16nt_input(m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()

    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out


def copy_files(src_dir: str, dst_dir: str):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for filename in os.listdir(src_dir):
        file_path = os.path.join(src_dir, filename)
        if os.path.isfile(file_path):
            shutil.copy(file_path, dst_dir)


# DeepGEMM kernel only support fp8 x fp8 => bf16
def generate_deep_gemm_kernel(m: int, n: int, k: int, input_dtype: str,
                              output_dtype: str, op_type: int,
                              config_path: str, kernel_saved_path: str = "",
                              tuner_device_id: int = 0):
    torch.cuda.set_device(tuner_device_id)

    original_kernel_conf = load_yaml_config(config_path)
    if original_kernel_conf is None:
        original_kernel_conf = {}

    items = torch.version.cuda.split(".")
    cuda_ver = int(items[0]) * 1000 + int(items[1]) * 10
    props = torch.cuda.get_device_properties(0)
    sm = props.major * 10 + props.minor

    if input_dtype != "fp8" or output_dtype != "bf16" or op_type != 4:
        return None

    lhs, rhs, out, ref_out = construct_deep_gemm_fp8fp8bf16nt_input(m, k, n)
    # generate kernel file
    deep_gemm.gemm_fp8_fp8_bf16_nt(lhs, rhs, out)

    # noinspection PyShadowingNames
    def test_func():
        # Construct new tensors every time to avoid L2 cache acceleration
        x_fp8, y_fp8, out, ref_out = construct_deep_gemm_fp8fp8bf16nt_input(
            m, k, n)
        deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)

    latency_in_ms = bench_kineto(
        test_func, 'fp8_gemm', suppress_kineto_output=True) * 1e3

    # NOTE(karlluo): following script adapted from https://github.com/deepseek-ai/DeepGEMM/tree/9b0dad86407609fed7b9046f9e4f298f04b2f3d5/deep_gemm/jit/compiler.py
    # fetch generated kernel file
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    n, _ = rhs.shape
    num_sms = get_num_sms()
    name = "gemm_fp8_fp8_bf16_nt"
    arg_defs = (('lhs', torch.float8_e4m3fn), ('lhs_scales', torch.float),
                ('rhs', torch.float8_e4m3fn), ('rhs_scales', torch.float),
                ('out', torch.bfloat16), ('m', int),
                ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size',
                                                                  int))

    num_min_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config = get_best_configs(
        m, n, k, 1, num_sms)
    smem_size = smem_config[0]
    keys = {'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n,
            'SWIZZLE_D_MODE': smem_config[1],
            'BLOCK_N_PADDING': smem_config[2],
            'NUM_STAGES': num_stages,
            'NUM_TMA_MULTICAST': tma_multicast_config[0],
            'IS_TMA_MULTICAST_ON_A': tma_multicast_config[1]}
    keys = {k: keys[k] for k in sorted(keys.keys())}
    signature = (name, f'{keys}')
    kernel_dir = jit_tuner.tuned[signature].path
    command = ["cuobjdump", "-sass", os.path.join(kernel_dir, "kernel.so")]
    cubin_file_path = os.path.join(kernel_dir, "kernel.cubin")
    with open(cubin_file_path, "w") as _cubin_file:
        return_code = subprocess.call(command, stdout=_cubin_file)
        assert return_code == 0, f'Failed to generate DeepGEMM kernel cubin with command {" ".join(command)}'

    gemm_algo_config = {
        str(sm): [{
            str(cuda_ver): [{
                "batch_size": 1,
                "m": m,
                "n": n,
                "k": k,
                "a_data_type": int(CudaDataType.CUDA_R_8F_E4M3),
                "b_data_type": int(CudaDataType.CUDA_R_8F_E4M3),
                "c_data_type": int(CudaDataType.CUDA_R_16BF),
                "compute_type": int(CudaDataType.CUDA_R_8F_E4M3),
                "trans_a": 1,
                "trans_b": 1,
                "op_type": 4,
                "gpu_elapsed_ms": latency_in_ms,
                "smem_size": smem_size,
                "num_sms": num_sms,
            }]
        }],
    }
    kernel_config = {
        "m": m,
        "n": n,
        "k": k,
        "smem_size": smem_size,
        "num_sms": num_sms
    }
    with open(os.path.join(kernel_dir, "config.yaml"), 'w') as config_file:
        yaml.dump(kernel_config, config_file, default_flow_style=False)
    # merge config
    merged_config = {**original_kernel_conf, **gemm_algo_config}
    with open(config_path, "w", encoding="utf-8") as _f:
        yaml.dump(merged_config, _f, allow_unicode=True, sort_keys=False)
    if kernel_saved_path:
        copy_files(kernel_dir, kernel_saved_path)
    else:
        print(f"Kernel saved to {kernel_dir}, please copy it to your desired location.")


if __name__ == "__main__":
    args = args_config()
    generate_deep_gemm_kernel(args.m, args.n, args.k, "fp8",
                              "bf16", 4, "gemm_algo_map.yaml", args.kernel_saved_path, args.tuner_device_id)