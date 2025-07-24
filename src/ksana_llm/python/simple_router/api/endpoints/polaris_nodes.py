# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""Polaris节点解析"""

from typing import Dict, Tuple, Any

G_CONSUMER_API = None


def get_consumer_api():
    raise NotImplementedError("get_available_nodes_with_polaris is not implemented yet.")


def get_available_nodes_with_polaris(namespace: str, 
                                     prefill_service: str, 
                                     decode_service: str) -> Tuple[Dict, Any, Any]:
    """从 Polaris 获取可用节点"""
    raise NotImplementedError("get_available_nodes_with_polaris is not implemented yet.")


def update_polaris_service_call_result(namespace: str, prefill_service: str, decode_service: str,
                                      prefill_instance: Any, decode_instance: Any,
                                      prefill_success: bool, decode_success: bool):
    """更新 Polaris 服务调用结果"""
    raise NotImplementedError("get_available_nodes_with_polaris is not implemented yet.")
