# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""Name Service 抽象基类和注册机制"""

from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional
import logging
import importlib

logger = logging.getLogger(__name__)


class NameServiceProvider(ABC):
    """Name Service 提供者抽象基类"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_available_nodes(
        self, **kwargs
    ) -> Tuple[Tuple[str, str], Tuple[str, str], Optional[Any], Optional[Any]]:
        """
        获取可用节点并选择一组节点

        Returns:
            Tuple containing:
            - prefill_node: (name, address) tuple for selected prefill node
            - decode_node: (name, address) tuple for selected decode node
            - prefill_instance: Optional prefill instance (for service discovery systems like Polaris)
            - decode_instance: Optional decode instance (for service discovery systems like Polaris)
        """
        pass

    def update_nodes_call_result(
        self,
        prefill_instance: Optional[Any],
        decode_instance: Optional[Any],
        prefill_success: bool,
        decode_success: bool,
        **kwargs,
    ):
        """
        更新节点调用结果

        Args:
            prefill_instance: prefill instance from get_available_nodes
            decode_instance: decode instance from get_available_nodes
            prefill_success: whether prefill call was successful
            decode_success: whether decode call was successful
        """
        pass


class NameServiceRegistry:
    """Name Service 注册器"""

    @classmethod
    def get_provider_from_config(
        cls, module_path: str
    ) -> Optional[NameServiceProvider]:
        """从配置的模块路径动态加载提供者"""
        try:
            # 动态导入模块
            module = importlib.import_module(module_path)
            # 假设每个模块都有一个 create_provider 函数返回 provider 实例
            if hasattr(module, "create_provider"):
                provider = module.create_provider()
                if not isinstance(provider, NameServiceProvider):
                    logger.error(
                        f"Module {module_path} did not return a valid NameServiceProvider instance"
                    )
                    return None
                logger.info(f"Loaded name service provider: {provider.name}")
                return provider
            else:
                logger.error(
                    f"Module {module_path} does not have create_provider function"
                )
                return None
        except ImportError as e:
            logger.exception(
                f"Failed to import name service provider module {module_path}: {e}"
            )
            return None
        except Exception as e:  # pylint: disable=broad-except
            logger.exception(
                f"Failed to create name service provider from {module_path}: {e}"
            )
            return None
