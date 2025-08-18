# Copyright 2024 Tencent Inc.  All rights reserved.

import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--type", help="inference type", type=str)
parser.add_argument("--test_func", type=str)
parser.add_argument("--n_start", type=int, default=-1)
parser.add_argument("--n_end", type=int, default=-1)
parser.add_argument("--m", type=int, default=-1)
parser.add_argument("--n", type=int, default=-1)
parser.add_argument("--group_num", type=int, default=-1)
parser.add_argument("--group_size", type=int, default=-1)
parser.add_argument(
    "--m_num_per_group", type=int, nargs="+", help="List of m numbers per group"
)
parser.add_argument("--group_info", type=int, nargs="+", help="List of group info")


def InvokeGatherSubmatrix(inference_data_type, args):
    input_tensor = (
        torch.from_numpy(np.load("gather_submatrix_input.npy"))
        .view(inference_data_type)
        .cuda()
    )

    tensors = []
    for i, m_num in enumerate(args.m_num_per_group):
        m_start_index = i * args.group_size
        tensors.append(
            input_tensor[m_start_index : m_start_index + m_num, args.n_start : args.n_end]
        )
    output = torch.cat(tensors, dim=0)

    if args.type == "bfloat16":
        output = output.view(torch.float16)
    np.save(f"gather_submatrix_output.npy", output.cpu().numpy())


def InvokeDpMapCopy(inference_data_type, args):
    input_tensor = (
        torch.from_numpy(np.load("map_copy_input.npy")).view(inference_data_type).cuda()
    )

    prefill_tensors = []
    decode_tensors = []
    group_base_id = 0
    for group_id in range(len(args.group_info) // 4):
        group_end_id = args.group_info[group_id * 4]
        prefill_num = args.group_info[group_id * 4 + 1]
        prefill_tensors.append(input_tensor[group_base_id : group_base_id + prefill_num, :])
        decode_tensors.append(input_tensor[group_base_id + prefill_num : group_end_id, :])
        group_base_id = group_end_id

    output = torch.cat(prefill_tensors + decode_tensors, dim=0)

    if args.type == "bfloat16":
        output = output.view(torch.float16)
    np.save(f"map_copy_output.npy", output.cpu().numpy())


if __name__ == "__main__":
    args = parser.parse_args()
    inference_data_type = torch.float32
    if args.type == "half":
        inference_data_type = torch.float16
    elif args.type == "bfloat16":
        inference_data_type = torch.bfloat16

    if args.test_func == "InvokeGatherSubmatrix":
        InvokeGatherSubmatrix(inference_data_type, args)
    elif args.test_func == "InvokeDpMapCopy":
        InvokeDpMapCopy(inference_data_type, args)
    else:
        raise ValueError(f"Unknown test function: {args.test_func}")
