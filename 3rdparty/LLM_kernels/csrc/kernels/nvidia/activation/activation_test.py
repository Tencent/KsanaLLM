# Copyright 2024 Tencent Inc.  All rights reserved.

import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--type", help="inference type", type=str)
parser.add_argument("--activation", help="activation type", type=str)
parser.add_argument("--activation_kernel", help="activation kernel", type=str)

def InvokeGenericActivation(args):
    inference_data_type = torch.float32
    if args.type == "half":
        inference_data_type = torch.float16
    elif args.type == "bfloat16":
        inference_data_type = torch.bfloat16

    activation = "silu"
    if args.activation == "gelu":
        activation = "gelu"
    elif args.activation == "relu":
        activation = "relu"

    # input
    #  Since NumPy lacks native bf16 support, we store bf16 data as float16 (same binary representation, different type
    #  interpretation). Note: When loading such npy files, you must reinterpret the data to the correct type.
    input = torch.from_numpy(
        np.load("activation_test_input.npy")).view(inference_data_type).cuda()
    gated_weight = torch.from_numpy(
        np.load("activation_test_gated_weight.npy")).view(inference_data_type).cuda()

    if activation == "silu":
        act = torch.nn.SiLU()
    elif activation == "gelu":
        act = torch.nn.GELU()
    else:  # activation == "relu"
        act = torch.nn.ReLU()
    output = act(input) * gated_weight

    #  Since NumPy lacks native bf16 support, we store bf16 data as float16 (same binary representation, different type
    #  interpretation). Note: When loading such npy files, you must reinterpret the data to the correct type.
    if args.type == "bfloat16":
        output = output.view(torch.float16)
    np.save("activation_test_output.npy", output.cpu().numpy())
    
def InvokeRowBasedActivation(args):
    inference_data_type = torch.float32
    if args.type == "half":
        inference_data_type = torch.float16
    elif args.type == "bfloat16":
        inference_data_type = torch.bfloat16

    # input
    input = torch.from_numpy(
        np.load("row_based_activation_test_input.npy")).view(inference_data_type).cuda()
    _, num_cols = input.shape
    mid_col = num_cols // 2
    gate_proj_output = input[:, :mid_col]
    up_proj_output = input[:, mid_col:]
    
    act = torch.nn.SiLU()

    output = act(gate_proj_output) * up_proj_output

    if args.type == "bfloat16":
        output = output.view(torch.float16)
    np.save("row_based_activation_test_output.npy", output.cpu().numpy())

if __name__ == "__main__":
    args = parser.parse_args()
    if(args.activation_kernel == "InvokeGenericActivation"):
        InvokeGenericActivation(args)
    elif(args.activation_kernel == "InvokeRowBasedActivation"):
        InvokeRowBasedActivation(args)


