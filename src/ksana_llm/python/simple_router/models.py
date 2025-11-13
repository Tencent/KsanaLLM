# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    func,
)

from database import Base


NODE_ROLE = Enum("prefill", "decode", name="node_role")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class NodeInfo(Base):
    """Physical compute node table."""

    __tablename__ = "node_info"

    id = Column(Integer, primary_key=True, autoincrement=True)
    node_id = Column(String(64), nullable=False, unique=True)
    cluster_name = Column(String(128), nullable=False)
    inference_addr = Column(String(128), nullable=False)
    coordinator_addr = Column(String(128), nullable=False)
    role = Column(NODE_ROLE, nullable=False)
    node_rank = Column(Integer, nullable=False, default=0)
    world_size = Column(Integer, nullable=False)
    device_num = Column(Integer, nullable=False)
    last_heartbeat = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    is_online = Column(Boolean, nullable=False, default=True)
    start_time = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    __table_args__ = (
        UniqueConstraint(
            "cluster_name",
            "inference_addr",
            "role",
            "node_rank",
            name="uq_cluster_addr_role_rank",
        ),
        Index("ix_node_cluster_role_online", "cluster_name", "role", "is_online"),
        Index("ix_node_last_heartbeat", "last_heartbeat"),
    )


class CommGroupPair(Base):
    """Communication meta between prefill and decode addresses."""

    __tablename__ = "comm_group_pair"

    id = Column(Integer, primary_key=True, autoincrement=True)
    comm_key = Column(String(256), nullable=False, unique=True)
    is_active = Column(Boolean, nullable=False, default=False)
    cluster_name = Column(String(128), nullable=False)
    prefill_addr = Column(String(128), nullable=False)
    decode_addr = Column(String(128), nullable=False)
    control_channel_meta = Column(JSON, nullable=False, default=list)
    data_channel_meta = Column(JSON, nullable=False, default=str)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        Index("ix_comm_cluster_active", "cluster_name"),
        Index("ix_comm_prefill_addr", "prefill_addr"),
        Index("ix_comm_decode_addr", "decode_addr"),
    )


class InferenceGroupStatus(Base):
    """Aggregated readiness for each inference address/role."""

    __tablename__ = "inference_group_status"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_name = Column(String(128), nullable=False)
    inference_addr = Column(String(128), nullable=False)
    role = Column(NODE_ROLE, nullable=False)
    world_size_expected = Column(Integer, nullable=False, default=0)
    online_device_sum = Column(Integer, nullable=False, default=0)
    online_node_count = Column(Integer, nullable=False, default=0)
    is_ready = Column(Boolean, nullable=False, default=False)
    last_computed_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "cluster_name", "inference_addr", "role", name="uq_group_status"
        ),
        Index("ix_status_role_ready", "role", "is_ready"),
    )


__all__ = ["NodeInfo", "CommGroupPair", "InferenceGroupStatus", "Base"]
