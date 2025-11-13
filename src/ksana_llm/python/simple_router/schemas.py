# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class NodeRegisterRequest(BaseSchema):
    inference_addr: str = Field(..., description="host:port of inference service")
    coordinator_addr: str = Field(..., description="host:port of control channel")
    cluster_name: str = Field(..., description="Cluster identifier")
    role: str = Field(..., pattern="^(prefill|decode)$")
    node_rank: int = Field(0, ge=0)
    world_size: int = Field(..., gt=0)
    device_num: int = Field(..., gt=0)
    start_time: datetime
    node_id: Optional[str] = Field(
        None, description="Optional externally provided node id"
    )


class SimpleNodeResponse(BaseSchema):
    node_id: str
    is_online: bool
    last_heartbeat: datetime


class HeartbeatRequest(BaseSchema):
    node_id: str


class HeartbeatResponse(BaseSchema):
    node_id: str
    role: str
    node_rank: int
    inference_addr: str
    coordinator_addr: str
    is_online: bool
    timestamp: datetime
    comm_group_to_control_meta: Dict[str, list[list[str]]]
    comm_group_to_data_meta: Dict[str, str]


class RegisterCommIDRequest(BaseSchema):
    node_id: str
    comm_key: str
    comm_id: str


class RegisterCommIDResponse(BaseSchema):
    status: str = "OK"
    comm_id: str
