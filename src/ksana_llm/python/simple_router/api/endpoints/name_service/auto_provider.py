# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""数据库节点解析"""
import random
from typing import Dict, List, Tuple
import logging
from config import settings
from db import db
from .name_service import NameServiceProvider

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


class AutoNameServiceProvider(NameServiceProvider):
    """Auto 模式 - 保持原有逻辑"""

    def __init__(self):
        super().__init__("auto")

    def get_available_nodes(
        self, **kwargs
    ) -> Tuple[Tuple[str, str], Tuple[str, str], None, None]:
        """使用原有的 auto 模式逻辑"""

        # 从数据库获取可用节点
        available_nodes = get_available_nodes_from_db(db.storage, settings.cluster_name)

        # 选择节点
        if not available_nodes["prefill"]:
            raise ValueError("No prefill nodes available")
        if not available_nodes["decode"]:
            raise ValueError("No decode nodes available")

        prefill_node = self._pick_random_node(available_nodes["prefill"])
        decode_node = self._pick_random_node(available_nodes["decode"])

        return prefill_node, decode_node, None, None

    def _pick_random_node(self, nodes: List[Tuple[str, str]]) -> Tuple[str, str]:
        """从节点列表中随机选择一个节点"""
        if not nodes:
            raise ValueError("No available nodes for processing")
        return random.choice(nodes)


def create_provider() -> NameServiceProvider:
    """创建 Auto 模式提供者"""
    return AutoNameServiceProvider()
