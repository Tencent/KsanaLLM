# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

"""API endpoints for node management in the KsanaLLM Router service.

This module contains the API endpoints for registering, updating and
monitoring compute nodes in the router service.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Path

from config import settings
from db import db
from models import NodeInfo, RegisterNodeRequest, HeartbeatRequest
from models import (
    RegisterCommIDRequest,
    SimpleNodeResponse,
    HeartbeatResponse,
    NodeResponse,
)


# Configure logging
logger = logging.getLogger("comm_coordinator")


# Create router
router = APIRouter()


@router.post("/", response_model=SimpleNodeResponse, status_code=201)
async def register_node(request: RegisterNodeRequest):
    """Register a new node in the system.

    Args:
        request: The node registration request.

    Returns:
        Information about the registered node.

    Raises:
        HTTPException: If registration fails.
    """
    try:
        # Validate group role
        if request.group_role not in ["prefill", "decode"]:
            raise HTTPException(
                status_code=400, detail="Group role must be 'prefill' or 'decode'"
            )

        # Handle hostname - if not provided, use IP address from inference_addr
        hostname = request.hostname or request.inference_addr.split(":")[0].replace(
            ".", "-"
        )

        # Create node
        cluster_name = (
            request.cluster_name if request.cluster_name else settings.cluster_name
        )

        if request.world_size < len(request.devices):
            raise HTTPException(
                status_code=400,
                detail="World size must match the number of devices",
            )

        node = NodeInfo(
            hostname=hostname,
            inference_addr=request.inference_addr,
            coordinator_port=request.coordinator_port,
            cluster_name=cluster_name,
            group_name=request.group_name,
            group_role=request.group_role,
            node_rank=request.node_rank,
            devices=request.devices,
            world_size=request.world_size,
            job_id=request.job_id,
            start_time=request.start_time,
            last_heartbeat=datetime.now(),
            is_online=True,
        )

        # Register node
        node_id = db.register_node(node)

        # Get registered node
        registered_node = db.get_node(node_id)
        if not registered_node:
            raise HTTPException(status_code=500, detail="Node registration failed")

        # Return simplified response
        return SimpleNodeResponse(
            node_id=registered_node.node_id,
            is_online=registered_node.is_online,
            last_heartbeat=registered_node.last_heartbeat,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Node registration failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Node registration failed: {str(e)}"
        )


@router.post("/heartbeat", response_model=HeartbeatResponse)
async def node_heartbeat(request: HeartbeatRequest):
    """Update node heartbeat and get cluster information.

    Args:
        request: The heartbeat request.

    Returns:
        Updated information about the node and the cluster.

    Raises:
        HTTPException: If the node does not exist or heartbeat update fails.
    """
    try:
        # Update heartbeat
        success = db.update_heartbeat(request.node_id)
        if not success:
            raise HTTPException(status_code=404, detail="Node does not exist")

        # Get node
        node = db.get_node(request.node_id)
        if not node:
            raise HTTPException(status_code=404, detail="Node does not exist")

        # Get fresh heartbeat response data
        heartbeat_response = db.get_heartbeat_info(request.node_id)
        if not heartbeat_response:
            raise HTTPException(
                status_code=500, detail="Failed to get heartbeat response"
            )

        return heartbeat_response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process node heartbeat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{node_id}", response_model=NodeResponse)
async def get_node(node_id: str = Path(..., description="Node ID")):
    """Get information about a specific node.

    Args:
        node_id: The ID of the node.

    Returns:
        Detailed information about the node.

    Raises:
        HTTPException: If the node does not exist.
    """
    node = db.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' does not exist")

    return NodeResponse(
        node_id=node.node_id,
        hostname=node.hostname,
        inference_addr=node.inference_addr,
        cluster_name=node.cluster_name,
        group_name=node.group_name,
        group_role=node.group_role,
        node_rank=node.node_rank,
        devices=node.devices,
        is_online=node.is_online,
        last_heartbeat=node.last_heartbeat,
        job_id=node.job_id,
        start_time=node.start_time,
        world_size=node.world_size,
    )


@router.post("/registerCommId", status_code=200)
async def register_comm_id(request: RegisterCommIDRequest):
    """Register Communication ID for a communication group.

    This endpoint is used to update the Communication ID of an existing communication group,
    typically called by the rank 0 of a prefill node.

    Args:
        request: The Communication ID registration request.

    Returns:
        A status message and the registered Communication ID.

    Raises:
        HTTPException: If registration fails or the node is not authorized.
    """
    if not request.node_id:
        raise HTTPException(status_code=400, detail="Node ID cannot be empty")

    node_info = db.get_node(request.node_id)
    if not node_info:
        raise HTTPException(
            status_code=404, detail=f"Node '{request.node_id}' does not exist"
        )

    if node_info.group_role != "prefill":
        raise HTTPException(
            status_code=403, detail="Only prefill nodes can register Communication ID"
        )

    if node_info.node_rank != 0:
        raise HTTPException(
            status_code=403,
            detail="Only rank 0 of prefill nodes can register Communication ID",
        )

    if not request.comm_id:
        raise HTTPException(status_code=400, detail="Communication ID cannot be empty")

    try:
        success = db.register_comm_id(request.comm_key, request.comm_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Communication group '{request.comm_key}' does not exist",
            )

        # Return response in specified format
        return {"status": "OK", "comm_id": request.comm_id}
    except Exception as e:
        logger.error(f"Failed to register Communication ID: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to register Communication ID: {str(e)}"
        )
