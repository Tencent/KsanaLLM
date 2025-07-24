# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""数据库节点解析"""

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def get_available_nodes_from_db(db_storage, cluster_name: str) -> Dict[str, List]:
    """从数据库获取可用的 prefill 和 decode 节点（node_rank=0）"""
    nodes = {"prefill": [], "decode": []}

    try:
        cluster = db_storage.get_cluster(cluster_name)

        # 查找 prefill 节点
        for group_name, group in cluster.prefill_groups.items():
            if not group.is_ready:
                continue

            for _, node in group.nodes.items():
                if node.is_online and node.node_rank == 0:
                    nodes["prefill"].append((group_name, f"{node.inference_addr}"))

        # 查找 decode 节点
        for group_name, group in cluster.decode_groups.items():
            if not group.is_ready:
                continue

            for _, node in group.nodes.items():
                if node.is_online and node.node_rank == 0:
                    nodes["decode"].append((group_name, f"{node.inference_addr}"))

    except Exception as e:  # pylint: disable=broad-except
        logger.exception(f"Error getting available nodes from database: {str(e)}")

    return nodes
