# Copyright 2024 Tencent Inc.  All rights reserved.

import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--type", help="inference type", type=str)
parser.add_argument("--test_func", type=str)
parser.add_argument("--m", type=int, default=-1)
parser.add_argument("--n", type=int, default=-1)
parser.add_argument("--input_offset", type=int, default=-1)
parser.add_argument("--output_n", type=int, default=-1)


def InvokeExtractSubMatrix(inference_data_type, args):
    input_tensor = (
        torch.from_numpy(np.load("extract_submatrix_input.npy"))
        .view(inference_data_type)
        .cuda()
    )
    
    output = input_tensor[:, args.input_offset : args.input_offset + args.output_n].contiguous()

    if args.type == "bfloat16":
        output = output.view(torch.float16)
    np.save(f"extract_submatrix_output.npy", output.cpu().numpy())


if __name__ == "__main__":
    args = parser.parse_args()
    inference_data_type = torch.float32
    if args.type == "half":
        inference_data_type = torch.float16
    elif args.type == "bfloat16":
        inference_data_type = torch.bfloat16

    if args.test_func == "InvokeExtractSubMatrix":
        InvokeExtractSubMatrix(inference_data_type, args)
    else:
        raise ValueError(f"Unknown test function: {args.test_func}")
