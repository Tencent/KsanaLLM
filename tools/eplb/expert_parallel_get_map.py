#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import argparse
import re
import json
from collections import defaultdict, Counter
import numpy as np


def get_step_id_from_filename(filename):
    """从文件名 topk_ids_{step}_{rank}.npy 中提取 step"""
    match = re.search(r'topk_ids_(\d+)_(\d+)\.npy', filename)
    if match:
        return int(match.group(1))
    return None


def load_topk_ids_file(filepath):
    """加载topk_ids文件并返回 numpy 数据"""
    try:
        data = np.load(filepath)
        return data
    except (IOError, ValueError, OSError) as e:
        print(f"Error loading {filepath}: {e}")
        return None


def calculate_expert_counts(topk_ids, ep_nums):
    """从topk_ids计算专家权重分布"""
    # 统计每个expert的使用频次
    expert_counts = defaultdict(int)
    flat_ids = topk_ids.reshape(-1)
    for expert_id in flat_ids:
        expert_counts[expert_id] += 1
    return expert_counts


def ep_balancer(expert_weight_array, ep_nums, nodes, gpu_per_nodes):
    """负载均衡算法，生成优化的expert到GPU映射"""
    gpus = nodes * gpu_per_nodes
    ep_per_gpu = ep_nums // gpus
    assert ep_nums % gpus == 0, "Number of experts must be divisible by the number of GPUs"

    expert_weights = expert_weight_array[0]
    sorted_indices = np.argsort(-expert_weights, kind='stable')

    gpu_weights = np.zeros(gpus)
    gpu_expert_counts = np.zeros(gpus, dtype=int)
    balance_expert_map = np.zeros(ep_nums, dtype=int)

    for expert_idx in sorted_indices:
        available_gpus = [gpu_id for gpu_id in range(
            gpus) if gpu_expert_counts[gpu_id] < ep_per_gpu]
        target_gpu = min(
            available_gpus, key=lambda gpu_id: gpu_weights[gpu_id])

        balance_expert_map[gpu_expert_counts[target_gpu] + target_gpu * ep_per_gpu] = expert_idx
        gpu_weights[target_gpu] += expert_weights[expert_idx]
        gpu_expert_counts[target_gpu] += 1
    return balance_expert_map.tolist()


def collect_expert_statistics(layer_folder, ep_nums, max_files=None):
    """收集层的expert使用统计信息"""
    print(f"Collecting statistics for {layer_folder}...")
    
    if not os.path.exists(layer_folder):
        print(f"Error: Layer folder {layer_folder} does not exist!")
        return None

    # 列举目录下所有 topk_ids 文件并按 step 排序
    files = []
    for filename in os.listdir(layer_folder):
        if filename.startswith('topk_ids_') and filename.endswith('.npy'):
            step = get_step_id_from_filename(filename)
            if step is not None:
                files.append((step, filename))
    files.sort(key=lambda x: x[0])

    if max_files:
        files = files[:max_files]

    print(f"Found {len(files)} topk_ids files")

    # 收集统计信息
    expert_global_counter = Counter()
    expert_calls = 0
    valid_files = 0
    # step_calls 存储每个 step 的累加结果（包含从第一步到当前 step 的累加计数）
    # 使用普通 dict 存储 step -> {expert_id: cumulative_count}
    step_calls = {}
    # cumulative_counter 保存当前已处理到的累计计数
    cumulative_counter = Counter()
    for i, (step, filename) in enumerate(files):
        filepath = os.path.join(layer_folder, filename)
        # 加载 numpy 数据
        topk_ids = load_topk_ids_file(filepath)
        if topk_ids is None:
            continue
        # 统计各个专家出现频次
        expert_counts = calculate_expert_counts(topk_ids, ep_nums)
        # 更新到总表中
        if expert_counts is not None:
            expert_global_counter.update(expert_counts)
            expert_calls += sum(expert_counts.values())
            valid_files += 1

            # 更新累计计数并保存当前 step 的累加结果到 step_calls
            cumulative_counter.update(expert_counts)
            # 只保留有效范围内的专家 id，并转换为普通 dict（便于后续处理和 JSON 序列化）
            step_calls[step] = {int(ep): int(cnt) for ep, cnt in cumulative_counter.items() if 0 <= ep < ep_nums}

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(files)} files...")
    print(f"  Valid files processed: {valid_files}/{len(files)}")
    print(f"  Total expert calls: {expert_calls}")
    print(f"  Unique experts used: {len(expert_global_counter)}")
    
    # 创建权重数组
    weight_array = np.zeros((1, ep_nums), dtype=np.float32)
    for ep_id, count in expert_global_counter.items():
        if 0 <= ep_id < ep_nums:
            weight_array[0, ep_id] = count

    return {
        'weight_array': weight_array,
        'counter': expert_global_counter,
        'calls': expert_calls,
        'step_calls': step_calls
    }
 

def optimize_latency_improvement(expert_layer_data, ep_balance_map, ep_nums, nodes, gpu_per_nodes, layer_name=None):
    """
    模拟命中情况并比较基线（顺序分配）与平衡映射的最大 GPU 计算量的差异。
    思路：
      - 使用 collect_expert_statistics 生成的 expert_layer_data['step_calls']
        其中每个 step 保存了该 step 的专家调用计数（{expert_id: cumulative_count}）。
      - 对于每个 step，计算基线映射和 ep_balance_map 下每个 GPU 的负载（负载 = 该 GPU 上所有专家的累计调用次数之和）。
      - 记录每个 step 的最大 GPU 负载，最后对所有 step 求和，比较基线与平衡映射的总和差异。
    返回：
      dict 包含 baseline_latency_total, balanced_latency_total, reduction
    """
    step_calls = expert_layer_data.get('step_calls', {})
    if not step_calls:
        print("  [optimize] No step_calls available for optimization.")
        return None

    gpus = nodes * gpu_per_nodes
    # 兼容性：计算 ep_per_gpu
    if gpus == 0:
        print("  [optimize] Invalid GPU count (0).")
        return None
    ep_per_gpu = ep_nums // gpus
    if ep_nums % gpus != 0:
        print("  [optimize] Warning: ep_nums is not divisible by total GPUs; using floor division for grouping.")

    # 构建平衡映射下每个 GPU 的 experts 列表
    balanced_gpu_experts = []
    for gpu_id in range(gpus):
        start = gpu_id * ep_per_gpu
        end = start + ep_per_gpu
        # ep_balance_map 的元素可能为 numpy int，需要转换为 int
        balanced_gpu_experts.append([int(x) for x in ep_balance_map[start:end]])

    # 构建基线（顺序）映射下每个 GPU 的 experts 列表
    baseline_gpu_experts = []
    for gpu_id in range(gpus):
        start = gpu_id * ep_per_gpu
        end = start + ep_per_gpu
        baseline_gpu_experts.append(list(range(start, end)))

    baseline_latency = 0
    balanced_latency = 0

    # 逐 step 计算最大 GPU 负载并累加
    for step in sorted(step_calls.keys()):
        cumulative = step_calls[step]  # dict: expert_id -> cumulative_count
        # 计算基线每个 GPU 的负载
        baseline_loads = [sum(cumulative.get(ep, 0) for ep in experts) for experts in baseline_gpu_experts]
        balanced_loads = [sum(cumulative.get(ep, 0) for ep in experts) for experts in balanced_gpu_experts]

        if baseline_loads:
            baseline_latency += max(baseline_loads)
        if balanced_loads:
            balanced_latency += max(balanced_loads)

    reduction = round((baseline_latency - balanced_latency) / baseline_latency * 100.0, 2)
    print(f"  [optimize] Layer={layer_name or 'unknown'} baseline_latency={baseline_latency}, "
          f"balanced_latency={balanced_latency}, reduction={reduction:.2f}%")

    return {
        'layer': layer_name,
        'baseline_latency': int(baseline_latency),
        'balanced_latency': int(balanced_latency),
        'reduction': f"{reduction}%"
    }


def generate_multi_layer_mappings(args):
    """生成多个层的expert映射"""
    mappings = {}
    optimize_results = {}
    
    # 开始处理前，打印用户传入的必要信息
    print(f"Generating mappings for layers {args.start_layer} to {args.end_layer}")
    print(f"Parameters: ep_nums={args.ep_nums}, nodes={args.nodes}, gpu_per_nodes={args.gpu_per_nodes}")
    if args.max_files:
        print(f"max_files={args.max_files} (testing mode)")
    
    # 逐层处理
    for layer_idx in range(args.start_layer, args.end_layer + 1):
        layer_name = f"layer_{layer_idx}"
        related_layer_path = f"{args.input_dir}/{layer_name}"

        # 获取数据文件
        expert_layer_data = collect_expert_statistics(related_layer_path, args.ep_nums, args.max_files)
        if expert_layer_data is None:
            print(f"Failed to collect statistics for {layer_name}")
            continue
        weight_array = expert_layer_data['weight_array']

        # 获取平衡分布
        ep_balance_map = ep_balancer(weight_array, args.ep_nums, args.nodes, args.gpu_per_nodes)

        # 存储平衡分布
        if ep_balance_map is not None:
            mappings[layer_name] = ep_balance_map
            if args.perf_optimize:
                # 传入完整的 expert_layer_data 以便优化函数访问 step_calls 以及元数据信息
                optimize_result = optimize_latency_improvement(expert_layer_data, ep_balance_map,
                                                               args.ep_nums, args.nodes,
                                                               args.gpu_per_nodes, layer_name)
                optimize_results[layer_name] = optimize_result
        else:
            print(f"Warning: Failed to generate mapping for {layer_name}")

    # 打印结果
    print(f"\n=== Generated Mappings ===")
    for layer_name, mapping in mappings.items():
        print(f"{layer_name}: {mapping}")

    # 保存到JSON文件
    if args.output_file:
        try:
            with open(args.output_file, 'w') as f:
                json.dump(mappings, f, ensure_ascii=False, indent=4)
            print(f"\nMappings saved to {args.output_file}")
        except (PermissionError, FileNotFoundError, OSError) as e:
            print(f"Error saving to {args.output_file}: {e}")
    if args.perf_optimize:
        try:
            total_baseline = 0
            total_balanced = 0
            for _, v in list(optimize_results.items()):
                # 如果字段缺失或不是数字，主动抛 TypeError
                b = int(float(v['baseline_latency']))
                bal = int(float(v['balanced_latency']))
                total_baseline += b
                total_balanced += bal

            total_reduction = (
                round((total_baseline - total_balanced) / total_baseline * 100, 2)
                if total_baseline > 0 else 0.0
            )
            optimize_results['total'] = {
                'baseline_latency': int(total_baseline),
                'balanced_latency': int(total_balanced),
                'reduction': f"{total_reduction}%"
            }
        except (TypeError, ValueError, ZeroDivisionError) as e:
            print(f"Error while computing totals: {type(e).__name__}: {e}")
            raise

        try:
            with open("perf_optimize.json", 'w', encoding='utf-8') as f:
                json.dump(optimize_results, f, ensure_ascii=False, indent=4)
        except (PermissionError, FileNotFoundError, OSError) as e:
            print(f"Error saving to perf_optimize.json: {type(e).__name__}: {e}")
        else:
            print("\n=== Optimize Result ===")
            for layer_name, optimize_result in optimize_results.items():
                print(f"{layer_name}: {optimize_result}")
            print("\nOptimize perf result saved to perf_optimize.json")
    return mappings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate expert mappings for multiple layers")
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        default="",
        help="Path to dump topk_ids"
    )
    parser.add_argument(
        "--start_layer",
        "-s",
        type=int,
        default=0,
        help="Start layer number"
    )
    parser.add_argument(
        "--end_layer",
        "-e",
        type=int,
        default=128,
        help="End layer number"
    )
    parser.add_argument(
        "--ep_nums",
        type=int,
        default=256,
        help="Number of experts"
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes"
    )
    parser.add_argument(
        "--gpu_per_nodes",
        type=int,
        default=8,
        help="Number of GPUs per node"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing)"
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default="eplb_config.json",
        help="Output JSON file path."
    )
    parser.add_argument(
        "--perf_optimize",
        "-p",
        action="store_true",
        help="Optimize the latency improvement."
    )
    args = parser.parse_args()

    args.input_dir = os.getenv('DUMP_EPLB_PATH', "") if args.input_dir == "" else args.input_dir
    if args.input_dir == "":
        args.input_dir = os.getenv('HOME', "") + "/.cache/KsanaLLM/EPLB/"
    assert os.path.exists(args.input_dir), f"Failed to generate {args.output_file}, cause input_dir is empty."

    # 生成映射
    mappings = generate_multi_layer_mappings(args)
    print(f"\nCompleted! Generated mappings for {len(mappings)} layers.")
