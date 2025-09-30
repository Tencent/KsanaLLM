#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""
生成每层的 expert 激活热力图

用法示例：
  python expert_activation_heatmap.py -i /path/to/dump -s 0 -e 3 --ep_nums 256

输出：一个 PNG 图片，x=expert_id, y=layer_num, 颜色=被激活次数
"""

import os
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_step_id_from_filename(filename):
    """从文件名 topk_ids_{step}_{rank}.npy 中提取 step"""
    match = re.search(r'topk_ids_(\d+)_(\d+)\.npy', filename)
    if match:
        return int(match.group(1))
    return None


def load_topk_ids_file(filepath):
    """加载 .npy 文件，返回 numpy array 或 None"""
    try:
        data = np.load(filepath)
        return data
    except (IOError, ValueError, OSError) as e:
        print(f"  [load] Failed to load {filepath}: {e}")
        return None


# 新增：一次性扫描所有层，返回 (layers, experts) 的累加矩阵
# 现在只保留实际有激活的 layer（避免为被 continue 跳过的 layer 留空列）
def collect_global_matrix(input_dir, start_layer, end_layer, ep_nums):
    """
    返回:
        mat: 形状 (ep_nums, num_active_layers)  int64
        layers: list[str] 如 ['layer_0', 'layer_3', ...]（只包含有激活的层）
    """
    layers = []
    counts_list = []

    for idx in range(start_layer, end_layer + 1):
        layer_name = f"layer_{idx}"
        layer_folder = os.path.join(input_dir, layer_name)
        if not os.path.exists(layer_folder):
            # 层目录不存在，视为未激活 / 被跳过
            continue

        total_counts = np.zeros((ep_nums,), dtype=np.int64)
        found_any = False

        # 把该层所有 topk_ids_*.npy 累加
        for fn in os.listdir(layer_folder):
            if fn.startswith('topk_ids_') and fn.endswith('.npy'):
                path = os.path.join(layer_folder, fn)
                arr = load_topk_ids_file(path)
                if arr is None:
                    continue
                flat = arr.reshape(-1)
                mask = (flat >= 0) & (flat < ep_nums)
                if np.any(mask):
                    filtered = flat[mask].astype(np.int64)
                    counts = np.bincount(filtered, minlength=ep_nums)[:ep_nums]
                    total_counts += counts
                    found_any = True

        # 仅当该层确有激活（总计 > 0）时才加入结果，避免为被 continue 跳过或空层预留列
        if found_any and total_counts.sum() > 0:
            layers.append(layer_name)
            counts_list.append(total_counts)

    if len(counts_list) == 0:
        # 没有任何激活层，返回空矩阵和空层列表
        return np.zeros((ep_nums, 0), dtype=np.int64), []

    # 将 list -> ndarray，形状为 (ep_nums, num_active_layers)
    mat = np.stack(counts_list, axis=1)
    return mat, layers


def plot_heatmap(mat, steps, layer_name, output_path, cmap='Reds', log_scale=False, dpi=150):
    """
    绘制并保存热力图（横纵轴交换后）
    mat: (ep_nums, num_steps)  原始：rows=expert, cols=step
    steps: list of step ids 或 layer names (len == num_steps) -> 将作为 y 轴标签
    现在：
      - 交换横纵坐标：x = expert id, y = step/layer
      - 尽量使每个 cell 接近正方形（aspect='equal'），通过动态计算 figsize
      - 减少图四周留白（subplots_adjust + bbox_inches='tight'）
    """
    if mat is None or steps is None:
        print(f"  [plot] Nothing to plot for {layer_name}")
        return False

    data = mat.astype(np.float64)
    if log_scale:
        data = np.log1p(data)

    # 原始 mat: (ep_nums, num_steps)
    ep_nums, num_steps = data.shape

    if num_steps == 0 or ep_nums == 0:
        print(f"  [plot] Empty matrix for {layer_name}, skipping.")
        return False

    # 交换坐标以便 x 为 expert，y 为 step
    # imshow 中的形状应为 (nrows, ncols) -> (num_steps, ep_nums)
    data = data.T
    nrows, ncols = data.shape  # nrows == num_steps, ncols == ep_nums

    # 基于单元格大小自动计算 figure 大小，使输出尽量接近正方形
    desired_cell = 0.12
    min_dim = 3.0   # 最小 figure 边长（英寸），稍微减少以减少空白
    max_dim = 24.0  # 最大 figure 边长（英寸）

    width = ncols * desired_cell
    height = nrows * desired_cell

    if width > max_dim or height > max_dim:
        desired_cell = max_dim / max(ncols, nrows)
    elif width < min_dim or height < min_dim:
        desired_cell = min_dim / max(ncols, nrows)

    fig_w = ncols * desired_cell
    fig_h = nrows * desired_cell

    # clamp
    fig_w = max(min_dim, min(max_dim, fig_w))
    fig_h = max(min_dim, min(max_dim, fig_h))

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)

    # 绘制：x=expert id, y=step/layer
    # 将 origin 设为 'upper'，使得索引较小的 layer 在上方（layer 小的在上面）
    im = ax.imshow(data, aspect='equal', origin='upper', interpolation='nearest', cmap=cmap)

    ax.set_xlabel('expert id')
    ax.set_ylabel('layer')

    # x ticks -> experts
    if ncols <= 40:
        ax.set_xticks(np.arange(ncols))
        ax.set_xticklabels([str(i) for i in range(ncols)], rotation=90, fontsize=8)
    else:
        xticks = np.linspace(0, ncols - 1, 10, dtype=int)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(i) for i in xticks], rotation=90, fontsize=8)

    # y ticks -> steps (labels passed in steps)
    if nrows <= 50:
        ax.set_yticks(np.arange(nrows))
        ax.set_yticklabels([str(s) for s in steps], fontsize=6)
    else:
        yticks = np.linspace(0, nrows - 1, 10, dtype=int)
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(steps[int(i)]) for i in yticks], fontsize=6)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('log1p(counts)' if log_scale else 'counts')

    plt.title(f'{layer_name} expert activations (experts={ep_nums}, steps={num_steps})')

    # 尽量减小边距并保留刻度/标签可读性
    plt.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.06)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 使用 bbox_inches='tight' 和小的 pad_inches 来进一步去除多余留白
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"  [plot] Saved heatmap to {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate expert activation heatmaps per layer")
    parser.add_argument('--input_dir', '-i', type=str, default='/workspace/tmp/dump/', help='Path to dump topk_ids')
    parser.add_argument('--start_layer', '-s', type=int, default=0, help='Start layer index')
    parser.add_argument('--end_layer', '-e', type=int, default=128, help='End layer index (inclusive)')
    parser.add_argument('--ep_nums', type=int, default=256, help='Number of experts')
    parser.add_argument('--output_dir', type=str, default='heatmaps', help='Directory to save heatmap PNGs')
    parser.add_argument('--log_scale', action='store_true', help='Apply log1p to color scale')
    parser.add_argument('--cmap', type=str, default='Reds', help='Matplotlib colormap name')
    args = parser.parse_args()

    args.input_dir = os.getenv('DUMP_EPLB_PATH', "") if args.input_dir == "" else args.input_dir
    if args.input_dir == "":
        args.input_dir = os.getenv('HOME', "") + "/.cache/KsanaLLM/EPLB/"
    assert os.path.exists(args.input_dir), f"Failed to generate {args.output_file}, cause input_dir is empty."

    print("Accumulating counts across all steps ...")
    mat, layers = collect_global_matrix(args.input_dir, args.start_layer, args.end_layer, args.ep_nums)

    out_file = os.path.join(args.output_dir, "layer_expert_heatmap.png")
    # 直接复用原来的 plot_heatmap，把 steps 换成 layers 即可
    plot_heatmap(mat, layers, "All layers", out_file,
                 cmap=args.cmap, log_scale=args.log_scale)

    # 可选：保存原始矩阵
    npy_out = os.path.join(args.output_dir, "layer_expert_matrix.npy")
    os.makedirs(os.path.dirname(npy_out), exist_ok=True)
    np.save(npy_out, mat)
    print(f"Saved matrix to {npy_out}")
    print("Done.")


if __name__ == '__main__':
    main()
