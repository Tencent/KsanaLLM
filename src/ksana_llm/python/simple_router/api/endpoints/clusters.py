# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

"""API endpoints for cluster management in the KsanaLLM Router service.

This module contains the API endpoints for listing and managing clusters
in the router service.
"""

import logging
from typing import List, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from db import db

logger = logging.getLogger(__name__)


class ClusterResponse(BaseModel):
    """Cluster information response model.

    Attributes:
        cluster_name: The name of the cluster.
        prefill_groups: The number of prefill groups in the cluster.
        decode_groups: The number of decode groups in the cluster.
        group_info: List of group information dictionaries.
    """

    cluster_name: str
    prefill_groups: int
    decode_groups: int
    group_info: Optional[List[Dict]] = None


# Create router
router = APIRouter()


@router.get("/", response_model=List[ClusterResponse])
async def list_clusters(
    active_only: bool = Query(False, description="Filter clusters")
):
    """List all clusters in the system.

    Args:
        active_only: If True, only active clusters are returned.

    Returns:
        A list of cluster information.

    Raises:
        HTTPException: If the default cluster does not exist.
    """
    result = []
    cluster = db.storage.get_cluster("default-cluster")
    if not cluster:
        raise HTTPException(status_code=404, detail="Default cluster does not exist")

    group_info = []
    for group in cluster.prefill_groups.values():
        group_info.append(
            {
                "group_name": group.group_name,
                "group_role": "prefill",
                "group_ready": str(group.is_ready).lower(),
            }
        )
    for group in cluster.decode_groups.values():
        group_info.append(
            {
                "group_name": group.group_name,
                "group_role": "decode",
                "group_ready": str(group.is_ready).lower(),
            }
        )
    result.append(
        ClusterResponse(
            cluster_name=cluster.cluster_name,
            prefill_groups=len(cluster.prefill_groups),
            decode_groups=len(cluster.decode_groups),
            group_info=group_info,
        )
    )

    return result
