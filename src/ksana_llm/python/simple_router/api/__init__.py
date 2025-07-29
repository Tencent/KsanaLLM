# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================

"""API router initialization for the KsanaLLM Router service.

This module initializes the main API router and includes sub-routers for 
clusters, groups, nodes, and generation endpoints.
"""

from fastapi import APIRouter

from api.endpoints import clusters, groups, nodes, generate

# Create main router
api_router = APIRouter()

# Register sub-routers
api_router.include_router(clusters.router, prefix="/clusters", tags=["clusters"])
api_router.include_router(groups.router, prefix="/groups", tags=["groups"])
api_router.include_router(nodes.router, prefix="/nodes", tags=["nodes"])
