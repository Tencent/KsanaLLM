# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import sys
import re
import copy

import json
import torch
import numpy as np
from safetensors import safe_open

PATTERN = r"layers\.(\d+).*experts\.(\d+)"


def get_layer_idx(tensor_name):
    match = re.search(PATTERN, tensor_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return -1, -1


def convert_uint16_bf16_to_fp32(u16_arr):
    """
    bf16 was save as uint16 in ksana,
    use this function to convert it to its real value as fp32
    """
    uint32_view = u16_arr.astype(np.uint32)
    bf16_padded = uint32_view << 16

    arr_float32 = bf16_padded.view(np.float32)

    return arr_float32.copy()


def check_torch_tensor_equl(tensor_name, torch_tensor, rank, dump_path):
    tensor_data_file = dump_path + "/" + \
        str(rank) + "/" + tensor_name + ".npy"
    if not os.path.isfile(tensor_data_file):
        print("Tensor file {} is not exist.".format(tensor_data_file))
        return

    read_data_arr = np.load(tensor_data_file)
    if read_data_arr.dtype == np.uint16:
        read_data_arr = convert_uint16_bf16_to_fp32(read_data_arr)
    base_data_arr = torch_tensor.to(torch.float32).numpy()

    if not read_data_arr.shape == base_data_arr.shape:
        print("Tensor {} has different shape, current shape: {}, "
              "target shape: {}, error.".format(tensor_name,
                                                read_data_arr.shape, base_data_arr.shape))
        sys.exit(1)
    full_matched = np.allclose(read_data_arr, base_data_arr)
    if not full_matched:
        print("Tensor {} is not numerically matched, error: {}.".format(
            tensor_name, np.sum(np.abs(read_data_arr - base_data_arr))))
        np.save("./input.npy", read_data_arr)
        np.save("./target.npy", base_data_arr)
        sys.exit(1)
    print("Tensor {} full matched, shape: {}, dtype: {}".format(
        tensor_name, base_data_arr.shape, base_data_arr.dtype))


def check_lm_head(state_dict, tensor_name, num_ranks, rank):
    torch_tensor = state_dict[tensor_name]

    # tp
    slice_size = int((torch_tensor.shape[1] + (num_ranks - 1)) / num_ranks)
    offset_beg = rank * slice_size
    offset_end = (rank + 1) * slice_size
    torch_tensor = torch_tensor[:, offset_beg:offset_end]

    # permute
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



def check_model_kv_a(process_weights, tensor_name, num_ranks, rank):
    torch_tensor = process_weights[tensor_name]
    check_torch_tensor_equl(tensor_name, torch_tensor, rank, dump_path)


def check_model_kv_b(process_weights, tensor_name, num_ranks, rank):
    torch_tensor = process_weights[tensor_name].contiguous()
    # pdb.set_trace()
    slice_size = int((torch_tensor.shape[1] + (num_ranks - 1)) / num_ranks)
    offset_beg = rank * slice_size
    offset_end = (rank + 1) * slice_size
    torch_tensor = torch_tensor[:, offset_beg:offset_end]
    check_torch_tensor_equl(tensor_name, torch_tensor, rank, dump_path)


def check_model_o_proj(process_weights, tensor_name, num_ranks, rank):
    torch_tensor = process_weights[tensor_name]
    slice_size = int((torch_tensor.shape[0] + (num_ranks - 1)) / num_ranks)
    offset_beg = rank * slice_size
    offset_end = (rank + 1) * slice_size
    torch_tensor = torch_tensor[offset_beg:offset_end, :]
    check_torch_tensor_equl(tensor_name, torch_tensor, rank, dump_path)


def check_model_q_b(process_weights, tensor_name, num_ranks, rank):
    torch_tensor = process_weights[tensor_name]
    slice_size = int((torch_tensor.shape[1] + (num_ranks - 1)) / num_ranks)
    offset_beg = rank * slice_size
    offset_end = (rank + 1) * slice_size
    torch_tensor = torch_tensor[:, offset_beg:offset_end]
    check_torch_tensor_equl(tensor_name, torch_tensor, rank, dump_path)


def check_model_q_a(process_weights, tensor_name, num_ranks, rank):
    torch_tensor = process_weights[tensor_name]
    check_torch_tensor_equl(tensor_name, torch_tensor, rank, dump_path)


def check_model_mlp_down_proj(process_weights, tensor_name, num_ranks, rank):
    torch_tensor = process_weights[tensor_name]
    slice_size = int((torch_tensor.shape[0] + (num_ranks - 1)) / num_ranks)
    offset_beg = rank * slice_size
    offset_end = (rank + 1) * slice_size
    torch_tensor = torch_tensor[offset_beg:offset_end, :]
    check_torch_tensor_equl(tensor_name, torch_tensor, rank, dump_path)


def check_model_mlp_gate_up_proj(process_weights, tensor_name, num_ranks, rank):
    torch_tensor = process_weights[tensor_name]
    gate_up_size  = torch_tensor.shape[0] // 2
    slice_size = int((gate_up_size + (num_ranks - 1)) / num_ranks)
    offset_beg = rank * slice_size
    offset_end = (rank + 1) * slice_size
    sliced_tensor = torch.empty([slice_size * 2,
                                 torch_tensor.shape[1]], dtype=torch_tensor.dtype)
    sliced_tensor[:slice_size, :] = torch_tensor[offset_beg:offset_end, :]
    sliced_tensor[slice_size:, :] = \
        torch_tensor[offset_beg + gate_up_size:offset_end + gate_up_size, :]
    sliced_tensor = sliced_tensor.permute(1, 0)
    check_torch_tensor_equl(tensor_name, sliced_tensor, rank, dump_path)


def check_model_mlp_experts_down_proj(process_weights, tensor_name, num_ranks, rank):
    torch_tensor = process_weights[tensor_name]
    slice_size = int((torch_tensor.shape[2] + (num_ranks - 1)) / num_ranks)
    offset_beg = rank * slice_size
    offset_end = (rank + 1) * slice_size
    torch_tensor = torch_tensor[:, :, offset_beg:offset_end]
    check_torch_tensor_equl(tensor_name, torch_tensor, rank, dump_path)


def check_model_mlp_experts_up_gate_proj(process_weights, tensor_name, num_ranks, rank):
    torch_tensor = process_weights[tensor_name]
    slice_size = int((torch_tensor.shape[1] / 2 + (num_ranks - 1)) / num_ranks)
    offset_beg = rank * slice_size
    offset_end = (rank + 1) * slice_size
    up_gate_split = int(torch_tensor.shape[1]/2)

    sliced_tensor = torch.empty([torch_tensor.shape[0],
                                 int(torch_tensor.shape[1] / num_ranks),
                                 torch_tensor.shape[2]], dtype=torch_tensor.dtype)
    for i in range(torch_tensor.shape[0]):
        sliced_tensor[i, :slice_size, :] = torch_tensor[i, :up_gate_split, :][offset_beg:offset_end, :]
        sliced_tensor[i, slice_size:, :] = torch_tensor[i, up_gate_split:, :][offset_beg:offset_end, :]

    check_torch_tensor_equl(tensor_name, sliced_tensor, rank, dump_path)


def process_model_weights(state_dict, model_config):
    processed_dict = dict()
    for weight_name, weight_tensor in state_dict.items():
        if "model.embed_tokens.weight" in weight_name:
            processed_dict[weight_name] = weight_tensor
        elif "lm_head.weight" in weight_name:
            processed_dict[weight_name] = weight_tensor.permute(1, 0)
        elif "norm.weight" in weight_name:
            processed_dict[weight_name] = weight_tensor
        elif "mlp.down_proj" in weight_name or "mlp.shared_experts.down_proj" in weight_name:
            weight_name = weight_name.replace(".shared_experts.", ".shared_expert.")
            processed_dict[weight_name] = weight_tensor.permute(1, 0)
        elif "mlp.shared_experts.gate_proj" in weight_name or \
             "mlp.shared_experts.up_proj" in weight_name or \
             "mlp.gate_proj" in weight_name or \
             "mlp.up_proj" in weight_name:
            gate_up_name = copy.deepcopy(weight_name)
            gate_up_name = gate_up_name.replace("shared_experts.", "shared_expert.")
            if "gate_proj" in weight_name:
                gate_up_name = gate_up_name.replace("gate_proj", "gate_up_proj")
            if "up_proj" in weight_name:
                gate_up_name = gate_up_name.replace("up_proj", "gate_up_proj")
            if gate_up_name not in processed_dict:
                processed_dict[gate_up_name] = torch.empty([weight_tensor.shape[0] * 2,
                                                            weight_tensor.shape[1]],
                                                            dtype=weight_tensor.dtype)
            if "gate_proj" in weight_name:
                processed_dict[gate_up_name][:weight_tensor.shape[0], :] = weight_tensor
            elif "up_proj" in weight_name:
                processed_dict[gate_up_name][weight_tensor.shape[0]:, :] = weight_tensor
        elif "mlp.experts." in weight_name:
            """
            1. cat up_proj and gate_proj to one tensor
            2. stack all expert from the same layers to one tensor
            """
            layer_idx, expert_idx = get_layer_idx(weight_name)
            if "up_proj" in weight_name or "gate_proj" in weight_name:
                up_gate_name = "model.layers." + str(layer_idx) + \
                    ".mlp.experts.up_gate_proj.weight"
                if up_gate_name not in processed_dict:
                    processed_dict[up_gate_name] = torch.empty([model_config["n_routed_experts"],
                                                                model_config["moe_intermediate_size"] * 2,
                                                                model_config["hidden_size"]])
                if "up_proj" in weight_name:
                    processed_dict[up_gate_name][expert_idx, model_config["moe_intermediate_size"]:, :] = weight_tensor
                elif "gate_proj" in weight_name:
                    processed_dict[up_gate_name][expert_idx, :model_config["moe_intermediate_size"], :] = weight_tensor
            elif "down_proj" in weight_name:
                down_proj_name = "model.layers." + str(layer_idx) + \
                    ".mlp.experts.down_proj.weight"
                if down_proj_name not in processed_dict:
                    processed_dict[down_proj_name] = torch.empty([model_config["n_routed_experts"],
                                                                  model_config["hidden_size"],
                                                                  model_config["moe_intermediate_size"]])
                processed_dict[down_proj_name][expert_idx, :, :] = weight_tensor
        elif "self_attn.kv_a_proj_with_mqa.weight" in weight_name:
            kv_a_lora_name = weight_name.replace(".kv_a_proj_with_mqa.", ".kv_a_lora_proj.")
            processed_dict[kv_a_lora_name] = weight_tensor[:model_config["kv_lora_rank"], :].permute(1, 0)

            kv_a_rope_name = weight_name.replace(".kv_a_proj_with_mqa.", ".kv_a_rope_proj.")
            processed_dict[kv_a_rope_name] = weight_tensor[model_config["kv_lora_rank"]:, :].permute(1, 0)
        elif "self_attn.kv_b_proj.weight" in weight_name:
            kv_b_nope_name = weight_name.replace(".kv_b_proj.", ".kv_b_nope_proj.")
            v_head_name = weight_name.replace(".kv_b_proj.", ".v_head_proj.")

            kv_b_nope_dim0 = model_config["num_attention_heads"] * model_config["qk_nope_head_dim"]
            v_head_dim0 = model_config["num_attention_heads"] * model_config["v_head_dim"]
            processed_dict[kv_b_nope_name] = \
                torch.empty([kv_b_nope_dim0, weight_tensor.shape[1]], dtype=weight_tensor.dtype)
            processed_dict[v_head_name] = torch.empty([v_head_dim0, weight_tensor.shape[1]], dtype=weight_tensor.dtype)

            for i in range(model_config["num_attention_heads"]):
                copy_st = i * (model_config["qk_nope_head_dim"] + model_config["v_head_dim"])
                copy_ed = (i+1)*(model_config["qk_nope_head_dim"] + model_config["v_head_dim"])
                processed_dict[kv_b_nope_name][i*model_config["qk_nope_head_dim"]:\
                                               (i+1)*model_config["qk_nope_head_dim"], :] = \
                    weight_tensor[copy_st:copy_st + model_config["qk_nope_head_dim"], :]
                processed_dict[v_head_name][i*model_config["v_head_dim"]:(i+1)*model_config["v_head_dim"], :] = \
                    weight_tensor[copy_st + model_config["qk_nope_head_dim"]:copy_ed, :]
            processed_dict[kv_b_nope_name] = processed_dict[kv_b_nope_name].permute(1, 0)
            processed_dict[v_head_name] = processed_dict[v_head_name].permute(1, 0)
        elif "self_attn.o_proj.weight" in weight_name:
            processed_dict[weight_name] = weight_tensor.permute(1, 0)
        elif "self_attn.q_b_proj.weight" in weight_name:
            q_b_nope_name = weight_name.replace(".q_b_proj.", ".q_b_nope_proj.")
            q_b_rope_name = weight_name.replace(".q_b_proj.", ".q_b_rope_proj.")

            q_b_nope_dim0 = model_config["num_attention_heads"] * model_config["qk_nope_head_dim"]
            q_b_rope_dim0 = model_config["num_attention_heads"] * model_config["qk_rope_head_dim"]
            processed_dict[q_b_nope_name] = torch.empty([q_b_nope_dim0,
                                                         weight_tensor.shape[1]],
                                                         dtype=weight_tensor.dtype)
            processed_dict[q_b_rope_name] = torch.empty([q_b_rope_dim0,
                                                         weight_tensor.shape[1]],
                                                         dtype=weight_tensor.dtype)

            for i in range(model_config["num_attention_heads"]):
                copy_st = i * (model_config["qk_nope_head_dim"] + model_config["qk_rope_head_dim"])
                copy_ed = (i+1)*(model_config["qk_nope_head_dim"] + model_config["qk_rope_head_dim"])
                processed_dict[q_b_nope_name][i*model_config["qk_nope_head_dim"]: \
                                              (i+1)*model_config["qk_nope_head_dim"], :] = \
                    weight_tensor[copy_st:copy_st + model_config["qk_nope_head_dim"], :]
                processed_dict[q_b_rope_name][i*model_config["qk_rope_head_dim"]: \
                                              (i+1)*model_config["qk_rope_head_dim"], :] = \
                    weight_tensor[copy_st + model_config["qk_nope_head_dim"]:copy_ed, :]
            processed_dict[q_b_nope_name] = processed_dict[q_b_nope_name].permute(1, 0)
            processed_dict[q_b_rope_name] = processed_dict[q_b_rope_name].permute(1, 0)
        elif "self_attn.q_a_proj.weight" in weight_name:
            processed_dict[weight_name] = weight_tensor.permute(1, 0)
    print("processed {} weights".format(len(processed_dict)))
    return processed_dict


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)

    model_dir = sys.argv[1]
    num_ranks = int(sys.argv[2])
    if not os.path.isdir(model_dir):
        print("Model dir {} is not exists.".format(model_dir))
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
    model_files = [os.path.join(os.path.abspath(model_dir), f) for f in os.listdir(model_dir) if not f.startswith('.')]
    for model_file in model_files:
        if not model_file.endswith(".safetensors"):
            continue
        print("Loading {}".format(model_file))
        with safe_open(model_file, framework = "pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    process_weights = process_model_weights(state_dict, model_config)
    for rank in range(num_ranks):
        # lm_head.weight
        check_lm_head(process_weights, "lm_head.weight", num_ranks, rank)

        # model.embed_tokens.weight
        check_embed_tokens(process_weights, "model.embed_tokens.weight", num_ranks, rank)

        # model.norm.weight
        check_model_norm(process_weights, "model.norm.weight", num_ranks, rank)

        # check all layers is time consuming, so we only check some layers here.
        for layer_idx in [0, 3]:
            NAME_PREFIX = "model.layers." + str(layer_idx) + "."

            # model.layers.N.input_layernorm.weight
            check_model_norm(process_weights, NAME_PREFIX + "input_layernorm.weight", num_ranks, rank)

            # model.layers.N.post_attention_layernorm.weight
            check_model_norm(process_weights, NAME_PREFIX + "post_attention_layernorm.weight", num_ranks, rank)

            # model.layers.N.self.attn.kv_a_layernorm.weight
            check_model_norm(process_weights, NAME_PREFIX + "self_attn.kv_a_layernorm.weight", num_ranks, rank)

            # model.layers.N.q_a_layernorm.weight
            check_model_norm(process_weights, NAME_PREFIX + "self_attn.q_a_layernorm.weight", num_ranks, rank)

            # model.layers.N.self_attn.kv_a_lora_proj.weight
            check_model_kv_a(process_weights, NAME_PREFIX + "self_attn.kv_a_lora_proj.weight", num_ranks, rank)

            # model.layers.N.self_attn.kv_a_rope_proj.weight
            check_model_kv_a(process_weights, NAME_PREFIX + "self_attn.kv_a_rope_proj.weight", num_ranks, rank)

            # model.layers.N.self_attn.kv_b_nope_proj.weight
            check_model_kv_b(process_weights, NAME_PREFIX + "self_attn.kv_b_nope_proj.weight", num_ranks, rank)

            # model.layers.N.self_attn.v_head_proj.weight
            check_model_kv_b(process_weights, NAME_PREFIX + "self_attn.v_head_proj.weight", num_ranks, rank)

            # model.layers.N.self_attn.o_proj.weight
            check_model_o_proj(process_weights, NAME_PREFIX + "self_attn.o_proj.weight", num_ranks, rank)

            # model.layers.N.self_attn.q_b_nope_proj.weight
            check_model_q_b(process_weights, NAME_PREFIX + "self_attn.q_b_nope_proj.weight", num_ranks, rank)

            # model.layers.N.self_attn.q_b_rope_proj.weight
            check_model_q_b(process_weights, NAME_PREFIX + "self_attn.q_b_rope_proj.weight", num_ranks, rank)

            # model.layers.N.self_attn.q_a_proj.weight
            check_model_q_a(process_weights, NAME_PREFIX + "self_attn.q_a_proj.weight", num_ranks, rank)

            # model.layers.0.mlp.down_proj.weight
            if layer_idx == 0:
                check_model_mlp_down_proj(process_weights, NAME_PREFIX + "mlp.down_proj.weight", num_ranks, rank)
            # model.layers.N.mlp.shared_expert.down_proj.weight
            elif layer_idx >= 1:
                check_model_mlp_down_proj(process_weights,
                                          NAME_PREFIX + "mlp.shared_expert.down_proj.weight",
                                          num_ranks, rank)

            # model.layers.0.mlp.gate_up_proj.weight
            if layer_idx == 0:
                check_model_mlp_gate_up_proj(process_weights,
                                             NAME_PREFIX + "mlp.gate_up_proj.weight",
                                             num_ranks, rank)
            # model.layers.N.mlp.shared_expert.gate_up_proj.weight
            elif layer_idx >= 1:
                check_model_mlp_gate_up_proj(process_weights,
                                             NAME_PREFIX + "mlp.shared_expert.gate_up_proj.weight",
                                             num_ranks, rank)

            # model.layers.N.mlp.experts.down_proj.weight
            if layer_idx >= 1:
                check_model_mlp_experts_down_proj(process_weights,
                                                  NAME_PREFIX + "mlp.experts.down_proj.weight",
                                                  num_ranks, rank)

            # model.layers.N.mlp.experts.up_gate_proj.weight
            if layer_idx >= 1:
                check_model_mlp_experts_up_gate_proj(process_weights,
                                                     NAME_PREFIX + "mlp.experts.up_gate_proj.weight",
                                                     num_ranks, rank)

    # All check ok.
    print("All check passed.")
    open(dump_path + "/SUCCESS", 'a').close()
    sys.exit(0)
