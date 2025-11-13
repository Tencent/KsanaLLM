# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
from __future__ import annotations

import ipaddress
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

import config
from database import get_session
from models import InferenceGroupStatus, NodeInfo
from schemas import (
    HeartbeatRequest,
    RegisterCommIDRequest,
    RegisterCommIDResponse,
    SimpleNodeResponse,
)
from services import (
    InvalidCommGroupError,
    NodeNotFoundError,
    get_comm_metadata,
    get_node,
    register_comm_group,
    register_node,
    update_inference_group_status,
    collect_comm_keys,
    check_and_filter_metadata,
)
from generate import forward_request


logger = logging.getLogger(__name__)
router = APIRouter()


class DeviceInfo(BaseModel):
    """Hardware description reported by an inference node."""

    device_id: int = Field(..., description="Device rank on the node")
    device_type: Optional[str] = Field(None, description="Optional device type")
    device_ip: str = Field(..., description="Device IP address")


class RegisterNodeCompatRequest(BaseModel):
    """Request payload understood by legacy registerNode clients."""

    inference_addr: str = Field(..., description="host:port of inference service")
    cluster_name: Optional[str] = Field(None, description="Cluster identifier")
    group_role: str = Field(..., description="Group role, prefill or decode")
    node_rank: int = Field(..., ge=0, description="Rank of the node inside the group")
    hostname: Optional[str] = Field(None, description="Optional host name")
    coordinator_addr: Optional[str] = Field(
        None, description="Coordinator service address"
    )
    world_size: Optional[int] = Field(
        None, gt=0, description="Total process world size"
    )
    devices: Optional[List[DeviceInfo]] = Field(
        None, description="Device list of the node"
    )
    job_id: Optional[str] = Field(None, description="Job identifier")
    start_time: Optional[str] = Field(None, description="Start time in ISO8601 format")
    comm_id: Optional[str] = Field(None, description="Optional communication id")
    node_id: Optional[str] = Field(
        None, description="Optional externally provided node id"
    )

    @field_validator("group_role")
    @classmethod
    def validate_group_role(cls, value: str) -> str:
        if value not in {"prefill", "decode"}:
            raise ValueError("group_role must be 'prefill' or 'decode'")
        return value

    @field_validator("inference_addr")
    @classmethod
    def validate_inference_addr(cls, value: str) -> str:
        try:
            _ensure_host_port(value, "inference_addr")
        except HTTPException as exc:
            raise ValueError(exc.detail) from exc
        return value


class SimpleRouterHeartbeatResponse(BaseModel):
    """Aggregated view returned to nodes during heartbeat calls."""

    node_id: str
    is_online: bool
    group_ready: bool
    coordinator_addr: str
    node_role: str
    timestamp: datetime
    comm_group_to_address: Dict[str, List[Tuple[int, int, str]]]
    comm_group_to_id: Dict[str, Optional[str]]


def _ensure_host_port(value: str, field_name: str) -> Tuple[str, str]:
    """Split `host:port` pairs and raise consistent HTTP errors when invalid."""

    if ":" not in value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} must be in 'host:port' format",
        )
    host, port = value.rsplit(":", 1)
    if not host or not port:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} must be in 'host:port' format",
        )
    return host, port


def _validate_coordinator_addr(value: Optional[str]) -> str:
    """Ensure the coordinator address is a clean IP:port pair."""

    if not value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="coordinator_addr is required",
        )

    candidate = value.strip()
    if candidate != value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="coordinator_addr must not contain leading or trailing whitespace",
        )

    if "://" in candidate or any(delim in candidate for delim in ("/", "?", "#")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="coordinator_addr must be in 'ip:port' format without prefixes",
        )

    host, port_str = _ensure_host_port(candidate, "coordinator_addr")

    try:
        ipaddress.ip_address(host)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="coordinator_addr host must be a valid IP address",
        ) from exc

    try:
        port = int(port_str)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="coordinator_addr port must be an integer",
        ) from exc

    if not 1 <= port <= 65535:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="coordinator_addr port must be between 1 and 65535",
        )

    return candidate


def _build_comm_group_addresses(
    session: Session,
    control_meta: Dict[str, List[List[str]]],
) -> Dict[str, List[Tuple[int, int, str]]]:
    """Expand legacy metadata format into flat tuples used by clients."""

    if not control_meta:
        return {}

    all_addresses = {
        addr
        for groups in control_meta.values()
        for addr_list in groups
        for addr in addr_list or []
    }

    if not all_addresses:
        return {key: [] for key in control_meta}

    nodes = session.scalars(
        select(NodeInfo).where(NodeInfo.coordinator_addr.in_(all_addresses))
    ).all()
    addr_to_node = {node.coordinator_addr: node for node in nodes}

    result: Dict[str, List[Tuple[int, int, str]]] = {}
    for comm_key, (prefill_addrs, decode_addrs) in control_meta.items():
        tuples: List[Tuple[int, int, str]] = []
        for addr in prefill_addrs or []:
            tuples.extend(_expand_address(addr, addr_to_node))
        for addr in decode_addrs or []:
            tuples.extend(_expand_address(addr, addr_to_node))
        result[comm_key] = tuples
    return result


def _expand_address(
    addr: str, registry: Dict[str, NodeInfo]
) -> List[Tuple[int, int, str]]:
    """Materialise device-level tuples for a recorded node."""

    node = registry.get(addr)
    if not node:
        logger.debug("Coordinator addr %s has no matching node record", addr)
        return [(0, 0, addr)]

    device_count = max(node.device_num or 1, 1)
    return [(node.node_rank, dev_id, addr) for dev_id in range(device_count)]


def _resolve_group_ready(session: Session, node: NodeInfo) -> bool:
    """Check whether the node's inference group currently satisfies readiness."""

    status_row = session.scalar(
        select(InferenceGroupStatus)
        .where(
            InferenceGroupStatus.cluster_name == node.cluster_name,
            InferenceGroupStatus.inference_addr == node.inference_addr,
            InferenceGroupStatus.role == node.role,
        )
        .limit(1)
    )
    return bool(status_row and status_row.is_ready)


@router.post(
    "/RegisterNode",
    response_model=SimpleNodeResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_node_endpoint(
    request: RegisterNodeCompatRequest,
    session: Session = Depends(get_session),
) -> SimpleNodeResponse:
    """Register a node or refresh its metadata if it already exists.
    
    Validate and reject requests from clusters other than the configured one.
    """
    cluster_name = request.cluster_name or config.get_settings().cluster_name
    if cluster_name != config.get_settings().cluster_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cluster '{cluster_name}' is not allowed; expecting '{config.get_settings().cluster_name}'",
        )
    devices = request.devices or []
    device_count = len(devices)
    world_size = request.world_size
    coordinator_addr = _validate_coordinator_addr(request.coordinator_addr)

    if world_size is None or device_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="world_size or devices must be provided",
        )

    if world_size < device_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="World size must match the number of devices",
        )

    try:
        node = register_node(
            session,
            node_id=request.node_id,
            cluster_name=cluster_name,
            inference_addr=request.inference_addr,
            coordinator_addr=coordinator_addr,
            role=request.group_role,
            node_rank=request.node_rank,
            world_size=world_size,
            device_num=device_count,
        )
    except Exception as exc:  # pragma: no cover - unexpected error path
        session.rollback()
        logger.exception("Failed to register node: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SimpleNodeResponse.model_validate(node)


@router.post("/Heartbeat", response_model=SimpleRouterHeartbeatResponse)
def node_heartbeat(
    request: HeartbeatRequest,
    session: Session = Depends(get_session),
) -> SimpleRouterHeartbeatResponse:
    """Handle heartbeat updates from nodes and return live metadata."""

    logger.debug("Received heartbeat for node_id=%s", request.node_id)
    try:
        node = update_inference_group_status(session, request.node_id)
    except NodeNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node '{exc.args[0]}' not found",
        ) from exc

    response = SimpleRouterHeartbeatResponse(
        node_id=node.node_id,
        is_online=node.is_online,
        group_ready=_resolve_group_ready(session, node),
        coordinator_addr=node.coordinator_addr,
        node_role=node.role,
        timestamp=datetime.now(),
        comm_group_to_address={},
        comm_group_to_id={},
    )
    comm_keys = collect_comm_keys(session, node)
    if comm_keys is None or len(comm_keys) == 0:
        return response
    all_control_meta, all_data_meta = get_comm_metadata(session, node, comm_keys)

    control_meta, data_meta = check_and_filter_metadata(
        session, node, all_control_meta, all_data_meta
    )
    response.comm_group_to_address = control_meta
    response.comm_group_to_id = data_meta

    return response


@router.post("/RegisterCommId", response_model=RegisterCommIDResponse)
def register_comm_id_endpoint(
    request: RegisterCommIDRequest,
    session: Session = Depends(get_session),
) -> RegisterCommIDResponse:
    """Assign or refresh a communication ID for a prefill group."""

    node = get_node(session, request.node_id)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node '{request.node_id}' not found",
        )

    try:
        pair = register_comm_group(
            session,
            node=node,
            comm_key=request.comm_key,
            comm_id=request.comm_id,
        )
    except InvalidCommGroupError as exc:
        message = str(exc)
        status_code_override = (
            status.HTTP_403_FORBIDDEN
            if "rank-0" in message
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=status_code_override, detail=message) from exc

    return RegisterCommIDResponse(comm_id=pair.data_channel_meta)


@router.post("/generate")
async def generate(req: Request):
    """Proxy `/generate` requests without altering payloads."""

    return await forward_request(req, "/generate")


@router.api_route("/v1/{path:path}", methods=["POST"])
async def proxy_v1_endpoints(req: Request, path: str):
    """Proxy OpenAI v1 compatible calls to the underlying services."""

    return await forward_request(req, f"/v1/{path}")


@router.api_route("/v2/{path:path}", methods=["POST"])
async def proxy_v2_endpoints(req: Request, path: str):
    """Proxy OpenAI v2 style routes."""

    return await forward_request(req, f"/v2/{path}")
