# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""Polaris Name Service 提供者"""

import logging
from typing import Tuple, Any
from config import settings
from polaris.api.consumer import (
    GetOneInstanceRequest,
    ServiceCallResult,
    create_consumer_by_default_config_file,
)
from polaris.wrapper import POLARIS_CALL_RET_OK, POLARIS_CALL_RET_ERROR
from .name_service import NameServiceProvider

logger = logging.getLogger(__name__)


def update_polaris_service_call_result(
    namespace: str,
    prefill_service: str,
    decode_service: str,
    prefill_instance: Any,
    decode_instance: Any,
    prefill_success: bool,
    decode_success: bool,
):
    """更新 Polaris 服务调用结果"""
    try:
        consumer_api = create_consumer_by_default_config_file()

        if prefill_instance:
            prefill_call_result = ServiceCallResult(
                namespace,
                prefill_service,
                prefill_instance.get_id(),
            )
            prefill_call_result.set_ret_status(
                POLARIS_CALL_RET_OK if prefill_success else POLARIS_CALL_RET_ERROR
            )
            consumer_api.update_service_call_result(prefill_call_result)

        if decode_instance:
            decode_call_result = ServiceCallResult(
                namespace,
                decode_service,
                decode_instance.get_id(),
            )
            decode_call_result.set_ret_status(
                POLARIS_CALL_RET_OK if decode_success else POLARIS_CALL_RET_ERROR
            )
            consumer_api.update_service_call_result(decode_call_result)

    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"Error updating Polaris service call result: {str(e)}")


class PolarisNameServiceProvider(NameServiceProvider):
    """Polaris Name Service 提供者"""

    def __init__(self):
        super().__init__("polaris")

    def get_available_nodes(
        self, **kwargs
    ) -> Tuple[Tuple[str, str], Tuple[str, str], Any, Any]:
        """从 Polaris 获取可用节点并选择一组节点"""
        namespace = settings.namespace
        prefill_service = settings.prefill_service
        decode_service = settings.decode_service

        nodes = {"prefill": [], "decode": []}
        prefill_instance = None
        decode_instance = None

        try:
            consumer_api = create_consumer_by_default_config_file()

            # 获取 prefill 节点
            request = GetOneInstanceRequest(
                namespace=namespace, service=prefill_service
            )
            prefill_instance = consumer_api.get_one_instance(request)
            prefill_node = "{host}:{port}".format(
                host=prefill_instance.get_host(), port=prefill_instance.get_port()
            )

            # 获取 decode 节点
            request = GetOneInstanceRequest(namespace=namespace, service=decode_service)
            decode_instance = consumer_api.get_one_instance(request)
            decode_node = "{host}:{port}".format(
                host=decode_instance.get_host(), port=decode_instance.get_port()
            )

            nodes["prefill"].append((prefill_node, f"{prefill_node}"))
            nodes["decode"].append((decode_node, f"{decode_node}"))
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Error getting available nodes from Polaris: {str(e)}")

        return (
            (prefill_node, f"{prefill_node}"),
            (decode_node, f"{decode_node}"),
            prefill_instance,
            decode_instance,
        )

    def update_nodes_call_result(
        self,
        prefill_instance: Any,
        decode_instance: Any,
        prefill_success: bool,
        decode_success: bool,
        **kwargs,
    ):
        """更新 Polaris 服务调用结果"""
        update_polaris_service_call_result(
            settings.namespace,
            settings.prefill_service,
            settings.decode_service,
            prefill_instance,
            decode_instance,
            prefill_success,
            decode_success,
        )


def create_provider() -> NameServiceProvider:
    """创建 Polaris 提供者"""
    return PolarisNameServiceProvider()
