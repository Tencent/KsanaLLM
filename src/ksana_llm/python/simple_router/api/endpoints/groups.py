# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
from datetime import datetime
from typing import Dict, Optional, Any
from fastapi import APIRouter, HTTPException, Query, Path
from db import db
from pydantic import BaseModel


# Request and response models
class CreateGroupRequest(BaseModel):
    """Create group request"""

    cluster_name: str
    group_name: str
    group_role: str  # "prefill" or "decode"


class GroupResponse(BaseModel):
    """Group information response"""

    group_id: str
    group_name: str
    group_role: str
    cluster_name: str
    is_ready: bool
    node_count: int
    world_size: Optional[int]
    created_at: datetime
    last_updated: datetime


class GroupDetailResponse(GroupResponse):
    """Detailed group information response"""

    comm_id: Optional[str] = None
    nodes: Dict[str, Any] = {}


# Create router
router = APIRouter()


@router.get(
    "/{cluster_name}/{group_role}/{group_name}", response_model=GroupDetailResponse
)
async def get_group(
    cluster_name: str = Path(..., description="Cluster name"),
    group_role: str = Path(..., description="Group type (prefill/decode)"),
    group_name: str = Path(..., description="Group name"),
    include_comm_id: bool = Query(
        False,
        description="Include Communication ID (only for ready prefill groups or any decode group)",
    ),
):
    """Get specific group detailed information"""
    # Validate group type
    if group_role not in ["prefill", "decode"]:
        raise HTTPException(
            status_code=400, detail="Group type must be 'prefill' or 'decode'"
        )

    # Get group
    group = db.get_group(cluster_name, group_name, group_role)
    if not group:
        raise HTTPException(
            status_code=404, detail=f"{group_role} group '{group_name}' does not exist"
        )

    # Decide whether to include Communication ID
    comm_id = None
    if include_comm_id and (
        group_role == "decode" or (group_role == "prefill" and group.is_ready)
    ):
        comm_id = group.comm_id

    # Convert nodes to dictionary
    nodes_dict = {}
    for node_id, node in group.nodes.items():
        nodes_dict[node_id] = node.dict(exclude={"gpus"})
        nodes_dict[node_id]["gpu_count"] = len(node.gpus)

    return GroupDetailResponse(
        group_id=group.group_id,
        group_name=group.group_name,
        group_role=group.group_role,
        cluster_name=group.cluster_name,
        is_ready=group.is_ready,
        node_count=len(group.nodes),
        world_size=group.world_size,
        created_at=group.created_at,
        last_updated=group.last_updated,
        comm_id=comm_id,
        nodes=nodes_dict,
    )
