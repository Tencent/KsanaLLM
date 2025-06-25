# Copyright 2024 Tencent Inc.  All rights reserved.

import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--type", help="inference type", type=str)
parser.add_argument("--variance_epsilon", help="variance epsilon", type=float)

if __name__ == "__main__":
    args = parser.parse_args()

    inference_data_type = torch.float32
    if args.type == "half":
        inference_data_type = torch.float16
    elif args.type == "bfloat16":
        inference_data_type = torch.bfloat16

    #  Since NumPy lacks native bf16 support, we store bf16 data as float16 (same binary representation, different type
    #  interpretation). Note: When loading such npy files, you must reinterpret the data to the correct type.
    input = torch.from_numpy(
        np.load("add_norm_test_input.npy")).view(inference_data_type).cuda()
    weight = torch.from_numpy(
        np.load("add_norm_test_weight.npy")).view(inference_data_type).cuda()
    residual = torch.from_numpy(
        np.load("add_norm_test_residual.npy")).view(inference_data_type).cuda()


    input_dtype = input.dtype
    residual = input + residual
    hidden_states = residual.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + args.variance_epsilon)
    rmsnorm_output = weight * hidden_states.to(input_dtype)

    #  Since NumPy lacks native bf16 support, we store bf16 data as float16 (same binary representation, different type
    #  interpretation). Note: When loading such npy files, you must reinterpret the data to the correct type.
    if args.type == "bfloat16":
        rmsnorm_output = rmsnorm_output.view(torch.float16)
        residual = residual.view(torch.float16)

    np.save("add_norm_test_output.npy", rmsnorm_output.cpu().numpy())
    np.save("add_norm_test_residual.npy", residual.cpu().numpy())
    
