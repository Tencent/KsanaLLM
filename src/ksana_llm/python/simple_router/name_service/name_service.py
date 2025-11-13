# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


class NameServiceProvider(ABC):
    """Abstract base class for discovering serving nodes."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_available_nodes(
        self, **kwargs: Any
    ) -> Tuple[Tuple[str, str], Tuple[str, str], Optional[Any], Optional[Any]]:
        """Return a selected prefill/decode node pair."""

    def update_nodes_call_result(
        self,
        prefill_instance: Optional[Any],
        decode_instance: Optional[Any],
        prefill_success: bool,
        decode_success: bool,
        **kwargs: Any,
    ) -> None:
        """Optional hook invoked after requests are sent."""
        # Default implementation does nothing.


class NameServiceRegistry:
    """Helper to load providers via dotted path string."""

    @classmethod
    def get_provider_from_config(
        cls, module_path: str
    ) -> Optional[NameServiceProvider]:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            logger.exception(
                "Failed to import name service provider module %s", module_path
            )
            return None

        if not hasattr(module, "create_provider"):
            logger.error("Module %s does not expose create_provider()", module_path)
            return None

        provider = module.create_provider()
        if not isinstance(provider, NameServiceProvider):
            logger.error("Provider from %s is not a NameServiceProvider", module_path)
            return None

        logger.info("Loaded name service provider: %s", provider.name)
        return provider
