# Copyright 2024 Tencent Inc.  All rights reserved.

import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--type", help="inference type", type=str)
parser.add_argument('--output_n', type=int, nargs='+', help='List of split offset')


def InvokeSplit(args):
    inference_data_type = torch.float32
    if args.type == "half":
        inference_data_type = torch.float16
    elif args.type == "bfloat16":
        inference_data_type = torch.bfloat16

    input = torch.from_numpy(
        np.load("split_test_input.npy")).view(inference_data_type).cuda()
    
    start_idx = 0
    for i, n in enumerate(args.output_n):
        output = input[:, start_idx:start_idx + n]
        start_idx += n
        if args.type == "bfloat16":
            output = output.view(torch.float16)
        np.save(f"split_test_output_{i}.npy", output.cpu().numpy())

if __name__ == "__main__":
    args = parser.parse_args()
    InvokeSplit(args)
