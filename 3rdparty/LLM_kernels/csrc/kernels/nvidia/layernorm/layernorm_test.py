# Copyright 2024 Tencent Inc.  All rights reserved.

import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--type", help="inference type", type=str)
parser.add_argument("--variance_epsilon", help="variance epsilon", type=float)
parser.add_argument("--test_mode", type=str,
                        default="default",
                        choices=[
                            'default', 'use_layernorm_3d', 'use_layernorm_fused_qkv',
                        ],
                        help='test mode from: default, use_layernorm_3d, use_layernorm_fused_qkv')

if __name__ == "__main__":
    args = parser.parse_args()

    inference_data_type = torch.float32
    if args.type == "half":
        inference_data_type = torch.float16
    elif args.type == "bfloat16":
        inference_data_type = torch.bfloat16

    if args.test_mode == "default":
        #  Since NumPy lacks native bf16 support, we store bf16 data as float16 (same binary representation, different type
        #  interpretation). Note: When loading such npy files, you must reinterpret the data to the correct type.
        input = torch.from_numpy(
            np.load("layernorm_test_input.npy")).view(inference_data_type).cuda()
        weight = torch.from_numpy(
            np.load("layernorm_test_weight.npy")).view(inference_data_type).cuda()
        bias = torch.from_numpy(
            np.load("layernorm_test_bias.npy")).view(inference_data_type).cuda()

        layernorm = torch.nn.LayerNorm(normalized_shape=input.shape[1], eps=args.variance_epsilon)
        layernorm.weight.data = weight
        layernorm.bias.data = bias
        layernorm_output = layernorm(input)

        #  Since NumPy lacks native bf16 support, we store bf16 data as float16 (same binary representation, different type
        #  interpretation). Note: When loading such npy files, you must reinterpret the data to the correct type.
        if args.type == "bfloat16":
            layernorm_output = layernorm_output.view(torch.float16)
        np.save("layernorm_test_output.npy", layernorm_output.cpu().detach().numpy())

        input_dtype = input.dtype
        hidden_states = input.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + args.variance_epsilon)
        rmsnorm_output = weight * hidden_states.to(input_dtype)

        if args.type == "bfloat16":
            rmsnorm_output = rmsnorm_output.view(torch.float16)
        np.save("rmsnorm_test_output.npy", rmsnorm_output.cpu().numpy())

    elif args.test_mode == "use_layernorm_3d":
        # for 3d layer norm test 
        input_3d = torch.from_numpy(
            np.load("rmsnorm_3d_test_input.npy")).view(inference_data_type).cuda()
        weight_3d = torch.from_numpy(
            np.load("rmsnorm_3d_test_weight.npy")).view(inference_data_type).cuda()
        mask = torch.from_numpy(
            np.load("rmsnorm_3d_test_mask.npy")).view(inference_data_type).cuda()

        input_dtype = input_3d.dtype
        hidden_states = input_3d[2:, 0:2, :]
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + args.variance_epsilon)
        hidden_states = weight_3d * hidden_states.to(input_dtype)
        input_3d[2:, 0:2, :] = hidden_states
        rmsnorm_output = input_3d
        if args.type == "bfloat16":
            rmsnorm_output = rmsnorm_output.view(torch.float16)
        np.save("rmsnorm_3d_test_torch_output.npy", rmsnorm_output.cpu().numpy())
    elif args.test_mode == "use_layernorm_fused_qkv":
        # for fused qkv layer norm test 
        input_fused_qkv = torch.from_numpy(
            np.load("rmsnorm_fused_qkv_test_input.npy")).view(inference_data_type).cuda()
        weight_fused_qkv = torch.from_numpy(
            np.load("rmsnorm_fused_qkv_test_weight.npy")).view(inference_data_type).cuda()
        mask = torch.from_numpy(
            np.load("rmsnorm_fused_qkv_test_mask.npy")).view(inference_data_type).cuda()

        input_dtype = input_fused_qkv.dtype
        hidden_states = input_fused_qkv[2:, :4, :].to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + args.variance_epsilon)
        hidden_states = weight_fused_qkv * hidden_states.to(input_dtype)
        input_fused_qkv[2:, :4, :] = hidden_states
        rmsnorm_output = input_fused_qkv
        if args.type == "bfloat16":
            rmsnorm_output = rmsnorm_output.view(torch.float16)
        np.save("rmsnorm_fused_qkv_test_torch_output.npy", rmsnorm_output.cpu().numpy())        
