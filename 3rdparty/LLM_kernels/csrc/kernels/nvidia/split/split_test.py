# Copyright 2024 Tencent Inc.  All rights reserved.

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--type", help="inference type", type=str)
parser.add_argument('--output_n', type=int, nargs='+', help='List of split offset')


def InvokeSplit(args):
    inference_data_type = np.float32
    if args.type == "half":
        inference_data_type = np.float16
    elif args.type == "bfloat16":
        # numpy 没有原生的 bfloat16，但可以用 float16 作为替代
        inference_data_type = np.float16

    input_data = np.load("split_test_input.npy").astype(inference_data_type)
    
    start_idx = 0
    for i, n in enumerate(args.output_n):
        output = input_data[:, start_idx:start_idx + n]
        start_idx += n
        if args.type == "bfloat16":
            output = output.astype(np.float16)
        np.save(f"split_test_output_{i}.npy", output)

if __name__ == "__main__":
    args = parser.parse_args()
    InvokeSplit(args)
