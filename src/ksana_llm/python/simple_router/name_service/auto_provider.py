# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
from __future__ import annotations

import logging
import random
from typing import Any, Optional, Tuple

from sqlalchemy import select

import config
from database import session_scope
from models import CommGroupPair, NodeInfo
from .name_service import NameServiceProvider

logger = logging.getLogger(__name__)


class AutoNameServiceProvider(NameServiceProvider):
    """Selects node pairs directly from the router database."""

    def __init__(self) -> None:
        super().__init__("auto")

    def get_available_nodes(
        self, **kwargs: Any
    ) -> Tuple[Tuple[str, str], Tuple[str, str], Optional[Any], Optional[Any]]:
        cluster_name = kwargs.get("cluster_name") or config.get_settings().cluster_name

        with session_scope() as session:
            stmt = select(CommGroupPair).where(CommGroupPair.is_active.is_(True))
            if cluster_name:
                stmt = stmt.where(CommGroupPair.cluster_name == cluster_name)

            pairs = list(session.scalars(stmt))
            random.shuffle(pairs)

            for pair in pairs:
                prefill_node = session.scalar(
                    select(NodeInfo).where(
                        NodeInfo.inference_addr == pair.prefill_addr,
                        NodeInfo.role == "prefill",
                        NodeInfo.node_rank == 0,
                        NodeInfo.is_online.is_(True),
                    )
                )
                decode_node = session.scalar(
                    select(NodeInfo).where(
                        NodeInfo.inference_addr == pair.decode_addr,
                        NodeInfo.role == "decode",
                        NodeInfo.node_rank == 0,
                        NodeInfo.is_online.is_(True),
                    )
                )

                if prefill_node and decode_node:
                    prefill_name, decode_name = pair.comm_key.split("__", 1)
                    logger.debug("Selected comm pair %s", pair.comm_key)
                    return (
                        (prefill_name, pair.prefill_addr),
                        (decode_name, pair.decode_addr),
                        None,
                        None,
                    )

        raise RuntimeError("No available nodes for processing")


def create_provider() -> NameServiceProvider:
    return AutoNameServiceProvider()
