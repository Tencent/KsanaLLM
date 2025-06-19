# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import sys

import json
import torch
import numpy as np


def check_torch_tensor_equl(tensor_name, torch_tensor, rank, dump_path):
    if torch_dtype == "bfloat16":
        torch_tensor = torch_tensor.to(torch.float32)

    tensor_data_file = dump_path + "/" + \
        str(rank) + "/" + tensor_name + ".npy"
    if not os.path.isfile(tensor_data_file):
        print("Tensor file {} is not exist.".format(tensor_data_file))
        sys.exit(1)

    read_data_arr = np.load(tensor_data_file)
    base_data_arr = torch_tensor.numpy()

    full_matched = np.allclose(read_data_arr, base_data_arr)
    if not full_matched:
        print("Tensor {} is not matched, error.".format(tensor_name))
        sys.exit(1)
    print("Tensor {} full matched, shape: {}, dtype: {}".format(
        tensor_name, base_data_arr.shape, base_data_arr.dtype))


def check_lm_head(state_dict, tensor_name, num_ranks, rank):
    torch_tensor = state_dict[tensor_name]

    # padding lm_head.weight if necessary.
    vocab_size = torch_tensor.shape[0]

    slice_size = int((vocab_size + (num_ranks - 1)) / num_ranks)
    padding_size = slice_size * num_ranks - vocab_size
    torch_tensor = torch.nn.functional.pad(
        torch_tensor, (0, 0, 0, padding_size), 'constant', 0)

    # tp
    offset_beg = rank * slice_size
    offset_end = (rank + 1) * slice_size
    torch_tensor = torch_tensor[offset_beg:offset_end, :]

    # permute
    torch_tensor = torch.permute(torch_tensor, [1, 0])
    check_torch_tensor_equl(tensor_name, torch_tensor, rank, dump_path)


def check_embed_tokens(state_dict, tensor_name, num_ranks, rank):
    torch_tensor = state_dict[tensor_name]

    slice_size = int(torch_tensor.shape[1] / num_ranks)
    offset_beg = rank * slice_size
    offset_end = (rank + 1) * slice_size
    torch_tensor = torch_tensor[:, offset_beg:offset_end]

    check_torch_tensor_equl(tensor_name, torch_tensor, rank, dump_path)


def check_model_norm(state_dict, tensor_name, num_ranks, rank):
    torch_tensor = state_dict[tensor_name]
    check_torch_tensor_equl(tensor_name, torch_tensor, rank, dump_path)


def check_attn_o_proj(state_dict, tensor_name, num_ranks, rank):
    torch_tensor = state_dict[tensor_name]

    # permute
    torch_tensor = torch.permute(torch_tensor, [1, 0])

    slice_size = int(torch_tensor.shape[0] / num_ranks)
    offset_beg = rank * slice_size
    offset_end = (rank + 1) * slice_size
    torch_tensor = torch_tensor[offset_beg:offset_end, :]

    check_torch_tensor_equl(tensor_name, torch_tensor, rank, dump_path)


def check_mlp_gate(state_dict, tensor_name, num_ranks, rank):
    torch_tensor = state_dict[tensor_name]

    # tp
    slice_size = int(torch_tensor.shape[0] / num_ranks)
    offset_beg = rank * slice_size
    offset_end = (rank + 1) * slice_size
    torch_tensor = torch_tensor[offset_beg:offset_end, :]

    # permute
    torch_tensor = torch.permute(torch_tensor, [1, 0])

    check_torch_tensor_equl(tensor_name, torch_tensor, rank, dump_path)


def check_attn_qkv(state_dict, name_prefix, num_ranks, rank):
    q_tensor_name = name_prefix + "self_attn.q_proj.weight"
    k_tensor_name = name_prefix + "self_attn.k_proj.weight"
    v_tensor_name = name_prefix + "self_attn.v_proj.weight"

    q_torch_tensor = state_dict[q_tensor_name]
    k_torch_tensor = state_dict[k_tensor_name]
    v_torch_tensor = state_dict[v_tensor_name]

    # tp
    slice_size = int(q_torch_tensor.shape[0] / num_ranks)
    offset_beg = rank * slice_size
    offset_end = (rank + 1) * slice_size

    q_torch_tensor = q_torch_tensor[offset_beg:offset_end, :]
    k_torch_tensor = k_torch_tensor[offset_beg:offset_end, :]
    v_torch_tensor = v_torch_tensor[offset_beg:offset_end, :]

    torch_tensor = torch.concat(
        [q_torch_tensor, k_torch_tensor, v_torch_tensor], 0)
    torch_tensor = torch.permute(torch_tensor, [1, 0])

    tensor_name = name_prefix + "self_attn.query_key_value.weight"
    check_torch_tensor_equl(tensor_name, torch_tensor, rank, dump_path)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(1)

    model_dir = sys.argv[1]
    num_ranks = int(sys.argv[2])
    if not os.path.isdir(model_dir):
        print("Model dir {} is not exist.".format(model_dir))
        sys.exit(1)

    dump_path = os.getcwd() + "/" + os.path.basename(model_dir.rstrip("/"))
    if not os.path.isdir(dump_path):
        print("Dump path {} is not exist.".format(dump_path))
        sys.exit(1)

    config_file = model_dir + "/config.json"
    if not os.path.isfile(config_file):
        print("Model config {} is not exist.".format(config_file))
        sys.exit(1)

    with open(config_file) as f:
        model_config = json.load(f)
    torch_dtype = model_config["torch_dtype"]
    num_hidden_layers = int(model_config["num_hidden_layers"])

    state_dict = {}

    model_files = os.listdir(model_dir)
    for model_file in model_files:
        if not model_file.endswith(".bin"):
            continue
        file_state_dict = torch.load(model_dir + "/" + model_file)
        state_dict.update(file_state_dict)

    for rank in range(num_ranks):
        # lm_head.weight
        check_lm_head(state_dict, "lm_head.weight", num_ranks, rank)

        # model.embed_tokens.weight
        check_embed_tokens(
            state_dict,
            "model.embed_tokens.weight",
            num_ranks,
            rank)

        # model.norm.weight
        check_model_norm(state_dict, "model.norm.weight", num_ranks, rank)

        for layer_idx in range(num_hidden_layers):
            NAME_PREFIX = "model.layers." + str(layer_idx) + "."

            # model.layers.N.input_layernorm.weight
            check_model_norm(state_dict,
                             NAME_PREFIX + "input_layernorm.weight",
                             num_ranks, rank)

            # model.layers.N.post_attention_layernorm.weight
            check_model_norm(state_dict,
                             NAME_PREFIX + "post_attention_layernorm.weight",
                             num_ranks, rank)

            # model.layers.N.self_attn.o_proj.weight
            check_attn_o_proj(state_dict,
                              NAME_PREFIX + "self_attn.o_proj.weight",
                              num_ranks, rank)

            # model.layers.N.mlp.down_proj.weight
            check_attn_o_proj(state_dict,
                              NAME_PREFIX + "mlp.down_proj.weight",
                              num_ranks, rank)

            # model.layers.N.mlp.gate_proj.weight
            check_mlp_gate(state_dict,
                           NAME_PREFIX + "mlp.gate_proj.weight",
                           num_ranks, rank)

            # model.layers.N.mlp.up_proj.weight
            check_mlp_gate(state_dict,
                           NAME_PREFIX + "mlp.up_proj.weight",
                           num_ranks, rank)

            # # model.layers.N.self_attn.query_key_value.weight.npy
            check_attn_qkv(state_dict, NAME_PREFIX, num_ranks, rank)

    # All check ok.
    print("All check passed.")
    open(dump_path + "/SUCCESS", 'a').close()
    sys.exit(0)
