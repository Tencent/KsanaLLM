import argparse
import json
from typing import Any, Dict
import random
import numpy as np
import torch

def dump_kernel(kernel, output_dir, kernel_name, config):
    """Write kernel artifacts: .cubin, .json, .ptx"""
    with open(f"{output_dir}/{kernel_name}.cubin", "wb") as _f:
        _f.write(kernel.asm['cubin'])
    with open(f"{output_dir}/{kernel_name}.json", "w") as _f:
        json_dict = {"shm_size": kernel.metadata.shared}
        if config.get("config", {}).get("num_warps", None) is not None and config.get("config", {}).get("num_stages", None) is not None:
            json_dict["num_warps"] = config.get("config").get("num_warps")
            json_dict["num_stages"] = config.get("config").get("num_stages")
        _f.write(json.dumps(json_dict))
    with open(f"{output_dir}/{kernel_name}.ptx", "w") as _f:
        SHM_SIZE = 0
        try:
            SHM_SIZE = kernel.metadata["shared"]
        except TypeError:
            SHM_SIZE = kernel.metadata.shared
        KERNEL_NAME = "default"
        try:
            KERNEL_NAME = kernel.metadata["name"]
        except TypeError:
            KERNEL_NAME = kernel.metadata.name
        print("//shared_memory:", SHM_SIZE, end=", ", file=_f)
        print("kernel_name:", KERNEL_NAME, file=_f)
        print(kernel.asm['ptx'], file=_f)

def str_to_bool(value):
    """
    Convert string-like values to boolean. Raises argparse.ArgumentTypeError
    to be directly usable as an argparse `type=` function.
    Accepted truthy (case-insensitive): 'true', 't', 'yes', 'y', '1'
    Accepted falsy  (case-insensitive): 'false', 'f', 'no', 'n', '0'

    Also returns booleans unchanged.
    """
    if isinstance(value, bool):
        return value
    try:
        lowered = value.lower()
    except AttributeError:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")
    if lowered in ('true', 't', 'yes', 'y', '1'):
        return True
    elif lowered in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def moe_align_block_cpu(topk_ids, expert_ids, sorted_ids, token_post_pad, token_num, topk, expert_num, block_size):
    cumsum = [0] * (expert_num + 1)
    token_cnts = [0] * (expert_num + 1)
    numel = token_num * topk
    sorted_ids.fill_(numel)
    expert_ids.fill_(-1)

    for i in range(numel):
        expert_id = topk_ids[i]
        if expert_id >= expert_num:
            continue
        token_cnts[expert_id] += 1

    for i in range(expert_num):
        cumsum[i + 1] = cumsum[i] + \
            (token_cnts[i] + block_size - 1) // block_size
        token_cnts[i] = 0
        for j in range(cumsum[i], cumsum[i + 1]):
            expert_ids[j] = i

    token_post_pad[0] = cumsum[expert_num] * block_size

    for i in range(numel):
        expert_id = topk_ids[i]
        if expert_id >= expert_num:
            continue
        idx = cumsum[expert_id] * block_size + token_cnts[expert_id]
        sorted_ids[idx] = i
        token_cnts[expert_id] += 1

def performance_kernel(fn, num_iters: int = 20, graph_iters: int = 10, warmup_iters: int = 5):
    # 为 JIT 生成代码 / 预热 kernel
    output_tensor, kernel = fn()

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_iters):
            fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(warmup_iters):
        graph.replay()
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(num_iters):
        graph.replay()
    end_event.record()
    end_event.synchronize()

    latencies = start_event.elapsed_time(end_event)
    kernel_time = latencies / (num_iters) * 1000  # us
    graph.reset()

    return kernel, kernel_time, output_tensor